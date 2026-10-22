from fractions import Fraction
import itertools
import os
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import torch

import comfy.utils
import folder_paths

from ..fastfacefusion.pp_fastfaceswap_models import (
    list_available_enhance_models,
    list_available_swap_models,
)

try:
    from comfy_api.latest import InputImpl, Types
except ImportError:
    class DummyVideo:
        def __init__(self, path: str):
            self.path = path

        def get_stream_source(self) -> str:
            return self.path

    InputImpl = type("InputImpl", (), {"VideoFromFile": DummyVideo})
    Types = None


def _extract_audio_from_file(video_path: str) -> Optional[Dict[str, Any]]:
    try:
        import av
        with av.open(video_path, mode="r") as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                return None
            audio_stream = audio_streams[0]
            resampler = av.audio.resampler.AudioResampler(format="fltp")
            audio_frames = []
            for packet in container.demux(audio_stream):
                for frame in itertools.chain.from_iterable(map(resampler.resample, packet.decode())):
                    audio_frames.append(frame.to_ndarray())
            if not audio_frames:
                return None
            audio_data = np.concatenate(audio_frames, axis=1)
            return {
                "waveform": torch.from_numpy(audio_data).unsqueeze(0),
                "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 44100,
            }
    except Exception:
        return None


def _get_engine_cls():
    from ..fastfacefusion.pp_fastfaceswap_engine import FastFaceSwapEngine, create_box_mask
    return FastFaceSwapEngine, create_box_mask


def get_input_videos() -> List[str]:
    try:
        input_dir = folder_paths.get_input_directory()
        files = os.listdir(input_dir)
        return sorted(folder_paths.filter_files_content_types(files, ["video"]))
    except Exception:
        return []


def get_input_images() -> List[str]:
    try:
        input_dir = folder_paths.get_input_directory()
        files = os.listdir(input_dir)
        return sorted(folder_paths.filter_files_content_types(files, ["image"]))
    except Exception:
        return []


PADDING_PRESETS = {
    "none": (0, 0, 0, 0),
    "tight (5%)": (5, 5, 5, 5),
    "moderate (10%)": (10, 10, 10, 10),
    "chin_guard": (0, 5, 20, 5),
    "forehead_guard": (20, 5, 0, 5),
    "cheek_guard": (0, 15, 0, 15),
}


class MonolithFaceBoost:
    """
    Pre-cursor face enhancer/restoration node that feeds into MonolithFaceSwap.
    Applies to BOTH Image and Video outputs during the swap pass.
    Defaults match the proven Dune 3.0.1 configuration (gfpgan_1.4 @ blend 80).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "enhance_model": (list_available_enhance_models(), {"default": "gfpgan_1.4"}),
                "blend": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MONOLITH_FACE_ENHANCER",)
    RETURN_NAMES = ("enhancer",)
    FUNCTION = "get_options"
    CATEGORY = "Monolith/FaceSwap"

    def get_options(self, enabled: bool, enhance_model: str, blend: int):
        return ({"enabled": enabled, "model": enhance_model if enabled else "none", "blend": blend},)


class MonolithFaceSwapLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "swap_model": (list_available_swap_models(), {"default": "hyperswap_1a_256"}),
                "enhance_model": (["none"] + list_available_enhance_models(), {"default": "gfpgan_1.4"}),
                "pixel_boost": (["none", "128x128", "256x256", "384x384", "512x512"], {"default": "256x256"}),
                "provider": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MONOLITH_FACESWAP_ENGINE",)
    RETURN_NAMES = ("engine",)
    FUNCTION = "load"
    CATEGORY = "Monolith/FaceSwap"

    def load(self, swap_model: str, enhance_model: str, pixel_boost: str, provider: str):
        FastFaceSwapEngine, _ = _get_engine_cls()
        enhance = enhance_model if enhance_model != "none" else None
        p_boost = pixel_boost if pixel_boost != "none" else None
        engine = FastFaceSwapEngine(
            swap_model=swap_model,
            enhance_model=enhance,
            pixel_boost=p_boost,
            execution_providers=[provider],
        )
        if not engine.pre_check():
            raise RuntimeError("Failed to load Monolith FaceSwap models. Check server console for details.")
        return (engine,)


class MonolithFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        videos = get_input_videos()
        images = get_input_images()
        presets = ["custom", "none", "tight (5%)", "moderate (10%)", "chin_guard", "forehead_guard", "cheek_guard"]
        return {
            "required": {
                "source_image": ("IMAGE",),
            },
            "optional": {
                "target_image": ("IMAGE",),
                "target_video": ("VIDEO",),
                "enhancer": ("MONOLITH_FACE_ENHANCER",),
                "image_file": (["none"] + images, {"default": "none"}),
                "video_file": (["none"] + videos, {"default": "none"}),
                "engine": ("MONOLITH_FACESWAP_ENGINE",),
                "swap_model": (list_available_swap_models(), {"default": "hyperswap_1a_256"}),
                "enhance_model": (["none"] + list_available_enhance_models(), {"default": "gfpgan_1.4"}),
                "enhancer_blend": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
                "face_swapper_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "pixel_boost": (["from_engine", "none", "128x128", "256x256", "384x384", "512x512"], {"default": "from_engine"}),
                "face_selector_mode": (["reference", "one", "many"], {"default": "reference"}),
                "face_selector_order": (["large-small", "left-right", "right-left", "top-bottom", "bottom-top", "small-large", "best-worst", "worst-best"], {"default": "large-small"}),
                "reference_face_distance": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.05}),
                "mask_blur": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mask_padding_preset": (presets, {"default": "none"}),
                "mask_padding_top": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "mask_padding_bottom": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "mask_padding_left": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "mask_padding_right": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "save_video": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI"}),
                "video_quality": ("INT", {"default": 10, "min": 1, "max": 51, "step": 1}),
                "use_gpu_encode": ("BOOLEAN", {"default": True}),
                "max_video_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "Monolith/FaceSwap"

    def process(
        self,
        source_image: torch.Tensor,
        target_image: Optional[torch.Tensor] = None,
        target_video: Optional[Any] = None,
        enhancer: Optional[Dict[str, Any]] = None,
        image_file: str = "none",
        video_file: str = "none",
        engine: Optional[Any] = None,
        swap_model: str = "inswapper_128_fp16",
        enhance_model: str = "gfpgan_1.4",
        enhancer_blend: int = 80,
        face_swapper_weight: float = 0.5,
        pixel_boost: str = "from_engine",
        face_selector_mode: str = "reference",
        face_selector_order: str = "large-small",
        reference_face_distance: float = 0.3,
        mask_blur: float = 0.3,
        mask_padding_preset: str = "none",
        mask_padding_top: int = 0,
        mask_padding_bottom: int = 0,
        mask_padding_left: int = 0,
        mask_padding_right: int = 0,
        save_video: bool = False,
        filename_prefix: str = "video/ComfyUI",
        video_quality: int = 10,
        use_gpu_encode: bool = True,
        max_video_frames: int = 0,
    ):
        import cv2
        FastFaceSwapEngine, _ = _get_engine_cls()

        if enhancer is not None:
            if enhancer.get("enabled", True):
                actual_enhance_model = enhancer.get("model", "none")
                actual_enhancer_blend = enhancer.get("blend", 80)
            else:
                actual_enhance_model = "none"
                actual_enhancer_blend = 0
        else:
            actual_enhance_model = enhance_model
            actual_enhancer_blend = enhancer_blend

        if engine is None:
            enhance = actual_enhance_model if actual_enhance_model != "none" else None
            p_boost = pixel_boost if (pixel_boost not in ["from_engine", "none"]) else "128x128"
            engine = FastFaceSwapEngine(
                swap_model=swap_model,
                enhance_model=enhance,
                pixel_boost=p_boost,
                execution_providers=["cuda"],
            )
            if not engine.pre_check():
                raise RuntimeError(f"MonolithFaceSwap: Failed to initialize swap model '{swap_model}'")
        else:
            engine.set_enhancer(actual_enhance_model)

        if mask_padding_preset != "custom" and mask_padding_preset in PADDING_PRESETS:
            pt, pr, pb, pl = PADDING_PRESETS[mask_padding_preset]
        else:
            pt, pr, pb, pl = mask_padding_top, mask_padding_right, mask_padding_bottom, mask_padding_left

        engine.face_swapper_weight = face_swapper_weight
        engine.face_selector_mode = face_selector_mode
        engine.face_selector_order = face_selector_order
        engine.reference_face_distance = reference_face_distance
        engine.mask_blur = mask_blur
        engine.mask_padding = [pt, pr, pb, pl]
        if pixel_boost != "from_engine":
            engine.pixel_boost = pixel_boost if pixel_boost != "none" else None

        src_frames = [(source_image[i].cpu().numpy() * 255.0).astype(np.uint8)[:, :, ::-1] for i in range(source_image.shape[0])]
        source_face = engine.extract_source_face(src_frames)
        if not source_face:
            raise ValueError("MonolithFaceSwap: No face found in source image!")
        source_embedding = engine._prepare_source_embedding(source_face)

        ui_images = []
        ui_videos = []
        out_image: Optional[torch.Tensor] = None
        out_video: Optional[Any] = None

        input_dir = os.path.realpath(folder_paths.get_input_directory())

        actual_target_image = target_image
        if actual_target_image is None and image_file != "none":
            input_img_path = os.path.realpath(os.path.join(input_dir, image_file))
            if input_img_path.startswith(input_dir + os.sep) and os.path.exists(input_img_path):
                img_pil = Image.open(input_img_path).convert("RGB")
                img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).unsqueeze(0)
                actual_target_image = img_tensor

        video_input_path: Optional[str] = None
        is_comp_video = (
            target_video is not None
            and hasattr(InputImpl, "VideoFromComponents")
            and isinstance(target_video, InputImpl.VideoFromComponents)
        )
        if target_video is not None and not is_comp_video:
            if hasattr(target_video, "get_stream_source"):
                src = target_video.get_stream_source()
                if isinstance(src, str) and os.path.exists(src):
                    video_input_path = src
            elif isinstance(target_video, str) and os.path.exists(target_video):
                video_input_path = target_video
            elif isinstance(target_video, dict) and "video" in target_video:
                video_input_path = target_video["video"]
            elif isinstance(target_video, (list, tuple)) and len(target_video) > 0 and isinstance(target_video[0], str):
                video_input_path = target_video[0]

        if not video_input_path and video_file != "none":
            candidate = os.path.realpath(os.path.join(input_dir, video_file))
            if candidate.startswith(input_dir + os.sep) and os.path.exists(candidate):
                video_input_path = candidate

        has_component_video = (
            video_input_path is None
            and is_comp_video
            and hasattr(target_video, "get_components")
        )

        if actual_target_image is None and not video_input_path and not has_component_video:
            raise ValueError("MonolithFaceSwap: Please provide either a target_image or target_video (or select from dropdowns).")

        do_enhance = engine.enhance_model_name is not None and actual_enhancer_blend > 0

        if actual_target_image is not None:
            out_frames = []
            ref_face_img = None
            for i in range(actual_target_image.shape[0]):
                frame_bgr = (actual_target_image[i].cpu().numpy() * 255.0).astype(np.uint8)[:, :, ::-1]
                if ref_face_img is None and engine.face_selector_mode == "reference":
                    f0_faces = engine.sort_faces_by_order(engine.get_many_faces([frame_bgr]), engine.face_selector_order)
                    ref_face_img = engine.get_one_face(f0_faces, 0)
                result_bgr = engine.process_frame(
                    source_embedding=source_embedding,
                    frame=frame_bgr,
                    enhance=do_enhance,
                    enhancer_blend=actual_enhancer_blend,
                    reference_face=ref_face_img,
                )
                result_rgb = result_bgr[:, :, ::-1].astype(np.float32) / 255.0
                out_frames.append(torch.from_numpy(result_rgb))
            out_image = torch.stack(out_frames)

            full_out_dir, base_fn, counter, subf, _ = folder_paths.get_save_image_path("monolith_faceswap_preview", folder_paths.get_temp_directory())
            os.makedirs(full_out_dir, exist_ok=True)
            preview_fn = f"{base_fn}_{counter:05}_.png"
            preview_path = os.path.join(full_out_dir, preview_fn)
            Image.fromarray((out_image[0].cpu().numpy() * 255.0).astype(np.uint8)).save(preview_path)
            ui_images.append({"filename": preview_fn, "subfolder": subf, "type": "temp"})

        if video_input_path:
            cap = cv2.VideoCapture(video_input_path)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            pbar = comfy.utils.ProgressBar(total_vid_frames if (max_video_frames == 0 or max_video_frames > total_vid_frames) else max_video_frames)

            def pbar_update(curr, tot):
                pbar.update(1)

            if save_video:
                prefix = (filename_prefix or "video/ComfyUI").strip() or "video/ComfyUI"
                full_out_dir, base_fn, counter, subf, _ = folder_paths.get_save_image_path(prefix, folder_paths.get_output_directory())
                os.makedirs(full_out_dir, exist_ok=True)
                out_video_fn = f"{base_fn}_{counter:05}_.mp4"
                out_video_path = os.path.join(full_out_dir, out_video_fn)

                ok, err = engine.process_video(
                    source=source_embedding,
                    target_path=video_input_path,
                    output_path=out_video_path,
                    video_quality=video_quality,
                    use_gpu_encode=use_gpu_encode,
                    enhance=do_enhance,
                    enhancer_blend=actual_enhancer_blend,
                    generate_thumbnail=True,
                    max_frames=max_video_frames,
                    progress_callback=pbar_update,
                )

                if not ok:
                    raise RuntimeError(f"MonolithFaceSwap video processing failed: {err}")

                out_video = InputImpl.VideoFromFile(out_video_path)
                ui_videos.append({"filename": out_video_fn, "subfolder": subf, "type": "output"})

                if out_image is None:
                    thumb_path = os.path.splitext(out_video_path)[0] + ".png"
                    if os.path.exists(thumb_path):
                        thumb_bgr = cv2.imread(thumb_path)
                        thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        out_image = torch.from_numpy(thumb_rgb).unsqueeze(0)
                    else:
                        out_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            else:
                rgb_frames, fps, err = engine.process_video_frames(
                    source=source_embedding,
                    target_path=video_input_path,
                    enhance=do_enhance,
                    enhancer_blend=actual_enhancer_blend,
                    max_frames=max_video_frames,
                    progress_callback=pbar_update,
                )
                if rgb_frames is None or len(rgb_frames) == 0:
                    raise RuntimeError(f"MonolithFaceSwap video frame processing failed: {err or 'No frames decoded'}")

                frames_tensor = torch.from_numpy(np.stack(rgb_frames, axis=0)).to(dtype=torch.float32).div_(255.0)
                if out_image is None:
                    out_image = frames_tensor

                mid_rgb = rgb_frames[len(rgb_frames) // 2]
                full_out_dir, base_fn, counter, subf, _ = folder_paths.get_save_image_path("monolith_faceswap_preview", folder_paths.get_temp_directory())
                os.makedirs(full_out_dir, exist_ok=True)
                preview_fn = f"{base_fn}_{counter:05}_.png"
                Image.fromarray(mid_rgb).save(os.path.join(full_out_dir, preview_fn))
                ui_images.append({"filename": preview_fn, "subfolder": subf, "type": "temp"})

                if Types is not None and hasattr(InputImpl, "VideoFromComponents"):
                    audio_dict = _extract_audio_from_file(video_input_path)
                    fps_frac = Fraction(max(1, round(fps * 1000)), 1000)
                    out_video = InputImpl.VideoFromComponents(
                        Types.VideoComponents(images=frames_tensor, audio=audio_dict, frame_rate=fps_frac)
                    )
        elif has_component_video:
            comp = target_video.get_components()
            vid_images = comp.images
            limit = vid_images.shape[0] if (max_video_frames == 0 or max_video_frames > vid_images.shape[0]) else max_video_frames
            pbar = comfy.utils.ProgressBar(limit)
            swapped_rgb = []
            ref_face_vid = None
            for i in range(limit):
                frame_bgr = (vid_images[i].cpu().numpy() * 255.0).astype(np.uint8)[:, :, ::-1]
                if ref_face_vid is None and engine.face_selector_mode == "reference":
                    f0_faces = engine.sort_faces_by_order(engine.get_many_faces([frame_bgr]), engine.face_selector_order)
                    ref_face_vid = engine.get_one_face(f0_faces, 0)
                result_bgr = engine.process_frame(
                    source_embedding=source_embedding,
                    frame=frame_bgr,
                    enhance=do_enhance,
                    enhancer_blend=actual_enhancer_blend,
                    reference_face=ref_face_vid,
                )
                swapped_rgb.append(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
                pbar.update(1)

            frames_tensor = torch.from_numpy(np.stack(swapped_rgb, axis=0)).to(dtype=torch.float32).div_(255.0)
            if out_image is None:
                out_image = frames_tensor

            if Types is not None and hasattr(InputImpl, "VideoFromComponents"):
                comp_video = InputImpl.VideoFromComponents(
                    Types.VideoComponents(images=frames_tensor, audio=comp.audio, frame_rate=comp.frame_rate)
                )
                if save_video:
                    prefix = (filename_prefix or "video/ComfyUI").strip() or "video/ComfyUI"
                    full_out_dir, base_fn, counter, subf, _ = folder_paths.get_save_image_path(prefix, folder_paths.get_output_directory())
                    os.makedirs(full_out_dir, exist_ok=True)
                    out_video_fn = f"{base_fn}_{counter:05}_.mp4"
                    out_video_path = os.path.join(full_out_dir, out_video_fn)
                    comp_video.save_to(out_video_path)
                    out_video = InputImpl.VideoFromFile(out_video_path)
                    ui_videos.append({"filename": out_video_fn, "subfolder": subf, "type": "output"})
                else:
                    out_video = comp_video
                    mid_rgb = swapped_rgb[len(swapped_rgb) // 2]
                    full_out_dir, base_fn, counter, subf, _ = folder_paths.get_save_image_path("monolith_faceswap_preview", folder_paths.get_temp_directory())
                    os.makedirs(full_out_dir, exist_ok=True)
                    preview_fn = f"{base_fn}_{counter:05}_.png"
                    Image.fromarray(mid_rgb).save(os.path.join(full_out_dir, preview_fn))
                    ui_images.append({"filename": preview_fn, "subfolder": subf, "type": "temp"})

        if out_image is None:
            out_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        ui_output = {}
        if ui_images:
            ui_output["images"] = ui_images
        if ui_videos:
            ui_output["videos"] = ui_videos

        return {
            "ui": ui_output,
            "result": (out_image, out_video),
        }


class MonolithFaceEnhancer:
    """Standalone face enhancement node for static images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhance_model": (list_available_enhance_models(), {"default": "gfpgan_1.4"}),
                "blend": ("INT", {"default": 80, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = "Monolith/FaceSwap"

    def enhance(self, image: torch.Tensor, enhance_model: str, blend: int):
        import cv2
        FastFaceSwapEngine, create_box_mask = _get_engine_cls()
        engine = FastFaceSwapEngine(
            swap_model="inswapper_128_fp16",
            enhance_model=enhance_model,
            execution_providers=["cuda"],
        )
        if not engine.pre_check():
            raise RuntimeError(f"MonolithFaceEnhancer: Failed to load enhancer model '{enhance_model}'")

        out_frames = []
        for i in range(image.shape[0]):
            frame_bgr = (image[i].cpu().numpy() * 255.0).astype(np.uint8)[:, :, ::-1]
            raw_faces = engine.get_many_faces([frame_bgr])
            faces = engine.select_target_faces(raw_faces)
            if not faces:
                out_frames.append(image[i])
                continue

            temp_frame = frame_bgr.copy()
            for face in faces:
                unenhanced = temp_frame.copy()
                enhance_size = engine.enhancer_info.get("size", (512, 512))
                enhance_template = engine.enhancer_info.get("template", "ffhq_512")

                crop, affine = engine.warp_face(temp_frame, face.landmark_set.get("5/68"), enhance_template, enhance_size)
                occ_mask = engine.create_occlusion_mask(crop)
                enhanced = engine._forward_enhancer(crop)

                crop_size = (enhance_size[1], enhance_size[0])
                box_mask = create_box_mask(crop_size, engine.mask_blur, [0, 0, 0, 0])
                mask = np.minimum(box_mask, occ_mask).clip(0, 1)

                pasted = engine.paste_back(temp_frame, enhanced, mask, affine)
                face_enhancer_blend = 1.0 - (blend / 100.0)
                temp_frame = cv2.addWeighted(unenhanced, face_enhancer_blend, pasted, 1.0 - face_enhancer_blend, 0)

            result_rgb = temp_frame[:, :, ::-1].astype(np.float32) / 255.0
            out_frames.append(torch.from_numpy(result_rgb))

        return (torch.stack(out_frames),)


FACESWAP_CLASS_MAPPINGS = {
    "MonolithFaceSwapLoader": MonolithFaceSwapLoader,
    "MonolithFaceBoost": MonolithFaceBoost,
    "MonolithFaceSwap": MonolithFaceSwap,
    "MonolithFaceEnhancer": MonolithFaceEnhancer,
}

FACESWAP_NAME_MAPPINGS = {
    "MonolithFaceSwapLoader": "Monolith FaceSwap Loader",
    "MonolithFaceBoost": "Monolith Face Boost",
    "MonolithFaceSwap": "Monolith FaceSwap",
    "MonolithFaceEnhancer": "Monolith Face Enhancer",
}
