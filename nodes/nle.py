from fractions import Fraction
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import wave

import numpy as np
from PIL import Image
import torch

import folder_paths
from .prompt import MonolithPromptSwitchByImage

try:
    from comfy_api.latest import InputImpl, Types
except ImportError:
    class _DummyVideo:
        def __init__(self, path: str):
            self.path = path

        def get_stream_source(self) -> str:
            return self.path

    InputImpl = type("InputImpl", (), {"VideoFromFile": _DummyVideo})
    Types = None

logger = logging.getLogger(__name__)

JELLYFIN_FFMPEG = Path("/usr/lib/jellyfin-ffmpeg/ffmpeg")
JELLYFIN_FFPROBE = Path("/usr/lib/jellyfin-ffmpeg/ffprobe")

_CACHED_HW_ENCODER: Optional[str] = None

ROTATE_MAP = {
    "0": "0",
    "90CW": "1",
    "180": "2",
    "270CW": "3",
    "1": "1",
    "2": "2",
    "3": "3",
}


def get_ffmpeg_bin() -> str:
    if JELLYFIN_FFMPEG.is_file() and os.access(JELLYFIN_FFMPEG, os.X_OK):
        return str(JELLYFIN_FFMPEG)
    return shutil.which("ffmpeg") or "ffmpeg"


def get_ffprobe_bin() -> str:
    if JELLYFIN_FFPROBE.is_file() and os.access(JELLYFIN_FFPROBE, os.X_OK):
        return str(JELLYFIN_FFPROBE)
    return shutil.which("ffprobe") or "ffprobe"


def detect_working_hw_encoder() -> str:
    global _CACHED_HW_ENCODER
    if _CACHED_HW_ENCODER is not None:
        return _CACHED_HW_ENCODER

    ffmpeg_bin = get_ffmpeg_bin()
    try:
        r = subprocess.run([ffmpeg_bin, "-encoders"], capture_output=True, text=True, timeout=5)
        candidates = []
        if "h264_rkmpp" in r.stdout:
            candidates.append("rkmpp")
        if "h264_nvenc" in r.stdout:
            candidates.append("nvenc")
        if "h264_videotoolbox" in r.stdout:
            candidates.append("videotoolbox")

        for enc in candidates:
            enc_name = f"h264_{enc}"
            probe = subprocess.run(
                [
                    ffmpeg_bin,
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=duration=1:size=1280x720:rate=30",
                    "-c:v",
                    enc_name,
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if probe.returncode == 0:
                _CACHED_HW_ENCODER = enc
                return enc
    except Exception as e:
        logger.warning("Monolith NLE: error detecting HW encoder: %s", e)

    _CACHED_HW_ENCODER = "cpu"
    return "cpu"


def _list_input_files(content_types: List[str]) -> List[str]:
    try:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return sorted(folder_paths.filter_files_content_types(files, content_types))
    except Exception:
        return []


def get_input_media_files() -> List[str]:
    return _list_input_files(["video", "image"])


def get_input_audio_files() -> List[str]:
    return _list_input_files(["audio", "video"])


def _resolve_input_file(filename: str) -> Optional[str]:
    if not filename or filename == "none":
        return None
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    try:
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        exact, canonical = MonolithPromptSwitchByImage._build_lookup([(f, f) for f in all_files])
        resolved_name, _ = MonolithPromptSwitchByImage._resolve_lookup(
            filename, exact, canonical, default=filename
        )
    except Exception:
        resolved_name = filename

    candidate = os.path.realpath(os.path.join(input_dir, resolved_name))
    if candidate.startswith(input_dir + os.sep) and os.path.isfile(candidate):
        return candidate
    return None


def _parse_fps(fps_str: str) -> float:
    try:
        if "/" in str(fps_str):
            num, den = str(fps_str).split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 25.0
        return float(fps_str)
    except Exception:
        return 25.0


def probe_media_specs(
    path: str, ffprobe_bin: str = "ffprobe", still_duration: float = 4.0
) -> Tuple[int, int, str, float, bool]:
    ext = os.path.splitext(path)[1].lower()
    is_image = ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(r.stdout) if r.stdout else {}
    stream = data.get("streams", [{}])[0] if data.get("streams") else {}
    fps = stream.get("r_frame_rate", "25/1")
    width = int(stream.get("width", 1080))
    height = int(stream.get("height", 1920))

    if is_image:
        return width, height, "25/1", float(still_duration), True

    if "duration" in stream and float(stream["duration"]) > 0.1:
        duration = float(stream["duration"])
    else:
        cmd2 = [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        fmt = (json.loads(r2.stdout) if r2.stdout else {}).get("format", {})
        duration = float(fmt["duration"]) if "duration" in fmt and float(fmt["duration"]) > 0.1 else float(still_duration)

    return width, height, fps, duration, False


def probe_has_audio(path: str, ffprobe_bin: str = "ffprobe") -> bool:
    cmd = [ffprobe_bin, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
        return len(data.get("streams", [])) > 0
    except Exception:
        return False


def extract_middle_frame(video_path: str, thumb_path: str, media_dur: Optional[float] = None) -> bool:
    ffprobe_bin = get_ffprobe_bin()
    ffmpeg_bin = get_ffmpeg_bin()
    try:
        if media_dur is not None and media_dur > 0:
            duration = media_dur
        else:
            cmd_probe = [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                video_path,
            ]
            r = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
            data = json.loads(r.stdout)
            duration = float(data.get("format", {}).get("duration", 0.0))

        timestamp = min(2.0, duration / 2.0) if duration > 0 else 0.0
        cmd_extract = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            thumb_path,
        ]
        subprocess.run(cmd_extract, capture_output=True, text=True, check=True)
        return os.path.isfile(thumb_path)
    except Exception as e:
        logger.warning("Monolith NLE: failed to extract middle frame thumbnail: %s", e)
        return False


def _materialize_video_or_image(
    video_obj: Optional[Any],
    image_tensor: Optional[torch.Tensor],
    file_name: str,
    temp_dir: str,
    prefix: str,
) -> Optional[str]:
    if video_obj is not None:
        if hasattr(video_obj, "get_stream_source"):
            trim_win = video_obj.get_active_trim_window() if hasattr(video_obj, "get_active_trim_window") else (0.0, 0.0)
            src = video_obj.get_stream_source()
            if isinstance(src, str) and os.path.isfile(src) and trim_win == (0.0, 0.0):
                return src
        if hasattr(video_obj, "save_to"):
            out_mp4 = os.path.join(temp_dir, f"{prefix}.mp4")
            video_obj.save_to(out_mp4)
            return out_mp4
        if isinstance(video_obj, str) and os.path.isfile(video_obj):
            return video_obj

    if image_tensor is not None and isinstance(image_tensor, torch.Tensor) and image_tensor.numel() > 0:
        if image_tensor.shape[0] == 1 or Types is None:
            arr = np.clip(image_tensor[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            out_png = os.path.join(temp_dir, f"{prefix}.png")
            Image.fromarray(arr).save(out_png)
            return out_png
        else:
            out_mp4 = os.path.join(temp_dir, f"{prefix}.mp4")
            vid = InputImpl.VideoFromComponents(
                Types.VideoComponents(images=image_tensor, frame_rate=Fraction(25))
            )
            vid.save_to(out_mp4)
            return out_mp4

    return _resolve_input_file(file_name)


def _materialize_mask(
    mask_tensor: Optional[torch.Tensor],
    alpha_file_name: str,
    invert_mask: bool,
    temp_dir: str,
    prefix: str = "overlay_alpha",
) -> Optional[str]:
    if mask_tensor is not None and isinstance(mask_tensor, torch.Tensor) and mask_tensor.numel() > 0:
        m = mask_tensor.detach().cpu()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim == 4:
            m = m[..., 0]
        # Skip dummy 64x64 all-zero mask emitted by LoadImage when source image has no alpha channel
        if m.shape[-2:] == (64, 64) and float(m.max().item()) == 0.0 and not invert_mask:
            return _resolve_input_file(alpha_file_name)

        alpha = m[0].numpy() if invert_mask else (1.0 - m[0].numpy())
        arr = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        out_png = os.path.join(temp_dir, f"{prefix}.png")
        Image.fromarray(arr, mode="L").save(out_png)
        return out_png

    return _resolve_input_file(alpha_file_name)


def _materialize_audio(
    audio_obj: Optional[Dict[str, Any]],
    audio_file_name: str,
    temp_dir: str,
) -> Optional[str]:
    if audio_obj is not None and isinstance(audio_obj, dict) and "waveform" in audio_obj:
        waveform = audio_obj["waveform"].detach().cpu()
        sample_rate = int(audio_obj.get("sample_rate", 44100))
        if waveform.ndim == 3:
            waveform = waveform[0]
        # waveform shape: [channels, samples]
        pcm = np.clip(waveform.t().numpy(), -1.0, 1.0)
        pcm_int16 = (pcm * 32767.0).astype(np.int16)
        channels = pcm_int16.shape[1] if pcm_int16.ndim == 2 else 1
        out_wav = os.path.join(temp_dir, "audio_track.wav")
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16.tobytes())
        return out_wav

    return _resolve_input_file(audio_file_name)


def execute_nle_render(
    media_path: str,
    output_path: str,
    overlay_path: Optional[str] = None,
    overlay_alpha_path: Optional[str] = None,
    endcard_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    rotate_input: str = "0",
    reverse: bool = False,
    still_duration: float = 4.0,
    endcard_duration: float = 4.0,
    audio_backtime: bool = True,
    audio_fade_frames: int = 6,
    hw_encoder: str = "auto",
) -> Dict[str, Any]:
    ffmpeg_bin = get_ffmpeg_bin()
    ffprobe_bin = get_ffprobe_bin()
    actual_hw = detect_working_hw_encoder() if hw_encoder == "auto" else hw_encoder
    rot_code = ROTATE_MAP.get(str(rotate_input), "0")

    width, height, fps, media_dur, is_image = probe_media_specs(
        media_path, ffprobe_bin, still_duration=still_duration
    )
    media_has_audio = probe_has_audio(media_path, ffprobe_bin)

    if rot_code in ("1", "3"):
        width, height = height, width

    # Ensure even dimensions for yuv420p H.264 encoders
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))

    endcard_dur = 0.0
    is_endcard_image = False
    if endcard_path:
        try:
            _, _, _, endcard_dur, is_endcard_image = probe_media_specs(
                endcard_path, ffprobe_bin, still_duration=endcard_duration
            )
        except Exception:
            endcard_dur = float(endcard_duration)
            is_endcard_image = True

    total_dur = media_dur + endcard_dur
    fps_val = _parse_fps(fps)

    inputs = [media_path]
    filter_lines = []

    if is_image:
        media_filter = "[0:v]"
        if rot_code == "1":
            media_filter += "transpose=1,"
        elif rot_code == "2":
            media_filter += "transpose=1,transpose=1,"
        elif rot_code == "3":
            media_filter += "transpose=2,"
        media_filter += f"scale={width}:{height},setsar=1[media_prep];"
        filter_lines.append(media_filter)

        input_idx = 1
        if overlay_path:
            inputs.append(overlay_path)
            overlay_idx = input_idx
            input_idx += 1
            if overlay_alpha_path:
                inputs.append(overlay_alpha_path)
                alpha_idx = input_idx
                input_idx += 1
                filter_lines.append(f"[{alpha_idx}:v]scale={width}:{height},setsar=1[alpha_v];")
                filter_lines.append(f"[{overlay_idx}:v]scale={width}:{height},setsar=1[overlay_v];")
                filter_lines.append("[overlay_v][alpha_v]alphamerge[overlay_merged];")
                filter_lines.append("[media_prep][overlay_merged]overlay=0:0[slide_single];")
            else:
                filter_lines.append(f"[{overlay_idx}:v]scale={width}:{height},setsar=1[overlay_v];")
                filter_lines.append("[media_prep][overlay_v]overlay=0:0[slide_single];")
            slide_src = "slide_single"
        else:
            slide_src = "media_prep"

        media_frames = max(1, int(round(media_dur * fps_val)))
        filter_lines.append(f"[{slide_src}]loop=loop={media_frames - 1}:size=1:start=0,fps={fps}[slide_v];")

        if endcard_path:
            inputs.append(endcard_path)
            endcard_idx = input_idx
            input_idx += 1
            if is_endcard_image:
                endcard_frames = max(1, int(round(endcard_dur * fps_val)))
                filter_lines.append(
                    f"[{endcard_idx}:v]scale={width}:{height},setsar=1,loop=loop={endcard_frames - 1}:size=1:start=0,fps={fps}[endcard_v];"
                )
            else:
                filter_lines.append(f"[{endcard_idx}:v]scale={width}:{height},setsar=1,fps={fps}[endcard_v];")
            filter_lines.append("[slide_v][endcard_v]concat=n=2:v=1:a=0[last_v];")
        else:
            filter_lines.append("[slide_v]null[last_v];")
        last_v = "last_v"
    else:
        media_filter = "[0:v]"
        if rot_code == "1":
            media_filter += "transpose=1,"
        elif rot_code == "2":
            media_filter += "transpose=1,transpose=1,"
        elif rot_code == "3":
            media_filter += "transpose=2,"
        if reverse:
            media_filter += f"reverse,scale={width}:{height},setsar=1,fps={fps}[media_prep];"
        else:
            media_filter += f"scale={width}:{height},setsar=1,fps={fps}[media_prep];"
        filter_lines.append(media_filter)

        last_v = "media_prep"
        input_idx = 1
        if overlay_path:
            inputs.append(overlay_path)
            overlay_idx = input_idx
            input_idx += 1
            if overlay_alpha_path:
                inputs.append(overlay_alpha_path)
                alpha_idx = input_idx
                input_idx += 1
                filter_lines.append(f"[{alpha_idx}:v]scale={width}:{height},setsar=1[alpha_v];")
                filter_lines.append(f"[{overlay_idx}:v]scale={width}:{height},setsar=1[overlay_v];")
                filter_lines.append("[overlay_v][alpha_v]alphamerge[overlay_merged];")
                filter_lines.append(f"[{last_v}][overlay_merged]overlay=0:0:eof_action=repeat[with_overlay];")
            else:
                filter_lines.append(f"[{overlay_idx}:v]scale={width}:{height},setsar=1[overlay_v];")
                filter_lines.append(f"[{last_v}][overlay_v]overlay=0:0:eof_action=repeat[with_overlay];")
            last_v = "with_overlay"

        if endcard_path:
            inputs.append(endcard_path)
            endcard_idx = input_idx
            input_idx += 1
            if is_endcard_image:
                endcard_frames = max(1, int(round(endcard_dur * fps_val)))
                filter_lines.append(
                    f"[{endcard_idx}:v]scale={width}:{height},setsar=1,loop=loop={endcard_frames - 1}:size=1:start=0,fps={fps}[endcard_v];"
                )
            else:
                filter_lines.append(f"[{endcard_idx}:v]scale={width}:{height},setsar=1,fps={fps}[endcard_v];")
            filter_lines.append(f"[{last_v}][endcard_v]concat=n=2:v=1:a=0[concat_v];")
            last_v = "concat_v"

    has_audio = False
    audio_mode = "none"
    audio_raw_dur = 0.0
    if audio_path:
        inputs.append(audio_path)
        audio_idx = input_idx
        input_idx += 1
        cmd_a = [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "json", audio_path]
        r_a = subprocess.run(cmd_a, capture_output=True, text=True)
        try:
            audio_raw_dur = float(json.loads(r_a.stdout)["format"]["duration"])
        except Exception:
            audio_raw_dur = total_dur

        if audio_backtime and audio_raw_dur > total_dur:
            trim_start = audio_raw_dur - total_dur
            fade_dur = max(0.0, float(audio_fade_frames) / fps_val)
            if fade_dur > 0:
                filter_lines.append(
                    f"[{audio_idx}:a]atrim=start={trim_start:.3f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={fade_dur:.3f}[audio_out]"
                )
            else:
                filter_lines.append(f"[{audio_idx}:a]atrim=start={trim_start:.3f},asetpts=PTS-STARTPTS[audio_out]")
            audio_mode = "backtimed"
        else:
            filter_lines.append(f"[{audio_idx}:a]anull[audio_out]")
            audio_mode = "external"
        has_audio = True
    elif media_has_audio:
        if reverse:
            filter_lines.append("[0:a]areverse[audio_out]")
            audio_mode = "embedded_reversed"
        else:
            filter_lines.append("[0:a]anull[audio_out]")
            audio_mode = "embedded"
        audio_raw_dur = media_dur
        has_audio = True

    if filter_lines[-1].endswith(";"):
        filter_lines[-1] = filter_lines[-1][:-1]
    filter_complex = " ".join(filter_lines)

    cmd = [ffmpeg_bin, "-y"]
    for p in inputs:
        cmd += ["-i", p]

    cmd += ["-filter_complex", filter_complex, "-map", f"[{last_v}]"]
    if has_audio:
        cmd += ["-map", "[audio_out]", "-c:a", "aac", "-b:a", "192k"]

    cmd += ["-t", f"{total_dur:.6f}"]

    if actual_hw == "nvenc":
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-b:v",
            "10M",
            "-maxrate",
            "12M",
            "-bufsize",
            "20M",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
    elif actual_hw in ("videotoolbox", "rkmpp"):
        cmd += [
            "-c:v",
            f"h264_{actual_hw}",
            "-b:v",
            "10M",
            "-maxrate",
            "12M",
            "-bufsize",
            "20M",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "10M",
            "-maxrate",
            "12M",
            "-bufsize",
            "20M",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Monolith NLE FFmpeg render failed ({actual_hw}): {res.stderr[-600:]}")

    thumb_ts = min(2.0, media_dur / 2.0) if media_dur > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": round(fps_val, 2),
        "media_dur": round(media_dur, 2),
        "endcard_dur": round(endcard_dur, 2),
        "total_dur": round(total_dur, 2),
        "is_image": is_image,
        "has_overlay": bool(overlay_path),
        "has_alpha": bool(overlay_alpha_path),
        "has_endcard": bool(endcard_path),
        "audio_mode": audio_mode,
        "audio_dur": round(audio_raw_dur, 2),
        "hw_encoder": actual_hw,
        "rotate_input": rotate_input,
        "reverse": reverse,
        "thumb_ts": round(thumb_ts, 2),
    }


def _shared_all_in_one_input_types() -> Dict[str, Any]:
    media_files = ["none"] + get_input_media_files()
    audio_files = ["none"] + get_input_audio_files()
    return {
        "required": {
            "rotate_input": (["0", "90CW", "180", "270CW"], {"default": "0"}),
            "reverse": ("BOOLEAN", {"default": False}),
            "still_duration": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 60.0, "step": 0.5}),
            "endcard_duration": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 60.0, "step": 0.5}),
            "invert_overlay_mask": ("BOOLEAN", {"default": False}),
            "hw_encoder": (["auto", "nvenc", "rkmpp", "videotoolbox", "cpu"], {"default": "auto"}),
        },
        "optional": {
            "media_video": ("VIDEO",),
            "media_image": ("IMAGE",),
            "overlay_video": ("VIDEO",),
            "overlay_image": ("IMAGE",),
            "overlay_mask": ("MASK",),
            "endcard_video": ("VIDEO",),
            "endcard_image": ("IMAGE",),
            "audio": ("AUDIO",),
            "media_file": (media_files, {"default": "none"}),
            "overlay_file": (media_files, {"default": "none"}),
            "overlay_alpha_file": (media_files, {"default": "none"}),
            "endcard_file": (media_files, {"default": "none"}),
            "audio_file": (audio_files, {"default": "none"}),
        },
    }


def _run_assembler_common(
    rotate_input: str = "0",
    reverse: bool = False,
    still_duration: float = 4.0,
    endcard_duration: float = 4.0,
    invert_overlay_mask: bool = False,
    hw_encoder: str = "auto",
    media_video: Optional[Any] = None,
    media_image: Optional[torch.Tensor] = None,
    overlay_video: Optional[Any] = None,
    overlay_image: Optional[torch.Tensor] = None,
    overlay_mask: Optional[torch.Tensor] = None,
    endcard_video: Optional[Any] = None,
    endcard_image: Optional[torch.Tensor] = None,
    audio: Optional[Dict[str, Any]] = None,
    media_file: str = "none",
    overlay_file: str = "none",
    overlay_alpha_file: str = "none",
    endcard_file: str = "none",
    audio_file: str = "none",
    audio_backtime: bool = True,
    audio_fade_frames: int = 6,
):
    temp_root = folder_paths.get_temp_directory()
    os.makedirs(temp_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="monolith_nle_", dir=temp_root)

    try:
        media_path = _materialize_video_or_image(media_video, media_image, media_file, temp_dir, "media")
        if not media_path:
            raise ValueError(
                "Monolith NLE Assembler: provide a base media input (media_video, media_image, or media_file)."
            )

        overlay_path = _materialize_video_or_image(overlay_video, overlay_image, overlay_file, temp_dir, "overlay")
        overlay_alpha_path = (
            _materialize_mask(overlay_mask, overlay_alpha_file, invert_overlay_mask, temp_dir)
            if overlay_path
            else None
        )
        endcard_path = _materialize_video_or_image(endcard_video, endcard_image, endcard_file, temp_dir, "endcard")
        audio_path = _materialize_audio(audio, audio_file, temp_dir)

        full_out_dir, base_fn, counter, subf, _ = folder_paths.get_save_image_path(
            "monolith_nle", folder_paths.get_output_directory()
        )
        os.makedirs(full_out_dir, exist_ok=True)
        out_video_fn = f"{base_fn}_{counter:05}_.mp4"
        out_video_path = os.path.join(full_out_dir, out_video_fn)

        timeline_meta = execute_nle_render(
            media_path=media_path,
            output_path=out_video_path,
            overlay_path=overlay_path,
            overlay_alpha_path=overlay_alpha_path,
            endcard_path=endcard_path,
            audio_path=audio_path,
            rotate_input=rotate_input,
            reverse=reverse,
            still_duration=still_duration,
            endcard_duration=endcard_duration,
            audio_backtime=audio_backtime,
            audio_fade_frames=audio_fade_frames,
            hw_encoder=hw_encoder,
        )

        thumb_path = os.path.join(temp_dir, "thumbnail.jpg")
        if extract_middle_frame(out_video_path, thumb_path, media_dur=timeline_meta["media_dur"]):
            thumb_pil = Image.open(thumb_path).convert("RGB")
            thumb_tensor = torch.from_numpy(np.array(thumb_pil).astype(np.float32) / 255.0).unsqueeze(0)
        else:
            thumb_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        out_video = InputImpl.VideoFromFile(out_video_path)
        video_entry = {"filename": out_video_fn, "subfolder": subf, "type": "output"}

        return {
            "ui": {
                "images": [video_entry],
                "animated": (True,),
                "videos": [video_entry],
                "nle_timeline": [timeline_meta],
            },
            "result": (out_video, thumb_tensor),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class MonolithNLEAssemblerV1:
    """
    v1 (Option A): Single All-in-One NLE Assembler node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _shared_all_in_one_input_types()

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "thumbnail")
    OUTPUT_NODE = True
    FUNCTION = "assemble"
    CATEGORY = "Monolith/NLE"

    def assemble(self, **kwargs):
        return _run_assembler_common(**kwargs)


class MonolithNLEAssemblerV2:
    """
    v2 (Option B): Single All-in-One NLE Assembler node with interactive multi-track visual timeline canvas.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _shared_all_in_one_input_types()

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "thumbnail")
    OUTPUT_NODE = True
    FUNCTION = "assemble"
    CATEGORY = "Monolith/NLE"

    def assemble(self, **kwargs):
        return _run_assembler_common(**kwargs)


class MonolithNLEOverlayTrack:
    """
    v3 (Option C): Modular V2 Overlay Track builder node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        media_files = ["none"] + get_input_media_files()
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "overlay_video": ("VIDEO",),
                "overlay_image": ("IMAGE",),
                "overlay_mask": ("MASK",),
                "overlay_file": (media_files, {"default": "none"}),
                "overlay_alpha_file": (media_files, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("MONOLITH_NLE_OVERLAY",)
    RETURN_NAMES = ("overlay_track",)
    FUNCTION = "build_track"
    CATEGORY = "Monolith/NLE"

    def build_track(
        self,
        enabled: bool = True,
        invert_mask: bool = False,
        overlay_video: Optional[Any] = None,
        overlay_image: Optional[torch.Tensor] = None,
        overlay_mask: Optional[torch.Tensor] = None,
        overlay_file: str = "none",
        overlay_alpha_file: str = "none",
    ):
        return (
            {
                "enabled": enabled,
                "invert_mask": invert_mask,
                "overlay_video": overlay_video,
                "overlay_image": overlay_image,
                "overlay_mask": overlay_mask,
                "overlay_file": overlay_file,
                "overlay_alpha_file": overlay_alpha_file,
            },
        )


class MonolithNLEEndcardTrack:
    """
    v3 (Option C): Modular V1 Endcard Track builder node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        media_files = ["none"] + get_input_media_files()
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "endcard_duration": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 60.0, "step": 0.5}),
            },
            "optional": {
                "endcard_video": ("VIDEO",),
                "endcard_image": ("IMAGE",),
                "endcard_file": (media_files, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("MONOLITH_NLE_ENDCARD",)
    RETURN_NAMES = ("endcard_track",)
    FUNCTION = "build_track"
    CATEGORY = "Monolith/NLE"

    def build_track(
        self,
        enabled: bool = True,
        endcard_duration: float = 4.0,
        endcard_video: Optional[Any] = None,
        endcard_image: Optional[torch.Tensor] = None,
        endcard_file: str = "none",
    ):
        return (
            {
                "enabled": enabled,
                "endcard_duration": endcard_duration,
                "endcard_video": endcard_video,
                "endcard_image": endcard_image,
                "endcard_file": endcard_file,
            },
        )


class MonolithNLEAudioTrack:
    """
    v3 (Option C): Modular A1 Audio Track builder node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        audio_files = ["none"] + get_input_audio_files()
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "backtime": ("BOOLEAN", {"default": True}),
                "fade_in_frames": ("INT", {"default": 6, "min": 0, "max": 120, "step": 1}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_file": (audio_files, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("MONOLITH_NLE_AUDIO",)
    RETURN_NAMES = ("audio_track",)
    FUNCTION = "build_track"
    CATEGORY = "Monolith/NLE"

    def build_track(
        self,
        enabled: bool = True,
        backtime: bool = True,
        fade_in_frames: int = 6,
        audio: Optional[Dict[str, Any]] = None,
        audio_file: str = "none",
    ):
        return (
            {
                "enabled": enabled,
                "backtime": backtime,
                "fade_in_frames": fade_in_frames,
                "audio": audio,
                "audio_file": audio_file,
            },
        )


class MonolithNLEAssemblerV3:
    """
    v3 (Option C): Modular NLE Master Assembler node accepting track wires.
    """

    @classmethod
    def INPUT_TYPES(cls):
        media_files = ["none"] + get_input_media_files()
        return {
            "required": {
                "rotate_input": (["0", "90CW", "180", "270CW"], {"default": "0"}),
                "reverse": ("BOOLEAN", {"default": False}),
                "still_duration": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "hw_encoder": (["auto", "nvenc", "rkmpp", "videotoolbox", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "media_video": ("VIDEO",),
                "media_image": ("IMAGE",),
                "overlay_track": ("MONOLITH_NLE_OVERLAY",),
                "endcard_track": ("MONOLITH_NLE_ENDCARD",),
                "audio_track": ("MONOLITH_NLE_AUDIO",),
                "media_file": (media_files, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "thumbnail")
    OUTPUT_NODE = True
    FUNCTION = "assemble"
    CATEGORY = "Monolith/NLE"

    def assemble(
        self,
        rotate_input: str = "0",
        reverse: bool = False,
        still_duration: float = 4.0,
        hw_encoder: str = "auto",
        media_video: Optional[Any] = None,
        media_image: Optional[torch.Tensor] = None,
        overlay_track: Optional[Dict[str, Any]] = None,
        endcard_track: Optional[Dict[str, Any]] = None,
        audio_track: Optional[Dict[str, Any]] = None,
        media_file: str = "none",
    ):
        ov = overlay_track if (overlay_track and overlay_track.get("enabled", True)) else {}
        ec = endcard_track if (endcard_track and endcard_track.get("enabled", True)) else {}
        au = audio_track if (audio_track and audio_track.get("enabled", True)) else {}

        return _run_assembler_common(
            rotate_input=rotate_input,
            reverse=reverse,
            still_duration=still_duration,
            endcard_duration=ec.get("endcard_duration", 4.0),
            invert_overlay_mask=ov.get("invert_mask", False),
            hw_encoder=hw_encoder,
            media_video=media_video,
            media_image=media_image,
            overlay_video=ov.get("overlay_video"),
            overlay_image=ov.get("overlay_image"),
            overlay_mask=ov.get("overlay_mask"),
            endcard_video=ec.get("endcard_video"),
            endcard_image=ec.get("endcard_image"),
            audio=au.get("audio"),
            media_file=media_file,
            overlay_file=ov.get("overlay_file", "none"),
            overlay_alpha_file=ov.get("overlay_alpha_file", "none"),
            endcard_file=ec.get("endcard_file", "none"),
            audio_file=au.get("audio_file", "none"),
            audio_backtime=au.get("backtime", True),
            audio_fade_frames=au.get("fade_in_frames", 6),
        )


NLE_CLASS_MAPPINGS = {
    "MonolithNLEAssemblerV1": MonolithNLEAssemblerV1,
    "MonolithNLEAssemblerV2": MonolithNLEAssemblerV2,
    "MonolithNLEAssemblerV3": MonolithNLEAssemblerV3,
    "MonolithNLEOverlayTrack": MonolithNLEOverlayTrack,
    "MonolithNLEEndcardTrack": MonolithNLEEndcardTrack,
    "MonolithNLEAudioTrack": MonolithNLEAudioTrack,
}

NLE_NAME_MAPPINGS = {
    "MonolithNLEAssemblerV1": "Monolith NLE Assembler v1 (All-in-One)",
    "MonolithNLEAssemblerV2": "Monolith NLE Assembler v2 (Visual Timeline)",
    "MonolithNLEAssemblerV3": "Monolith NLE Assembler v3 (Modular)",
    "MonolithNLEOverlayTrack": "Monolith NLE Overlay Track (v3)",
    "MonolithNLEEndcardTrack": "Monolith NLE Endcard Track (v3)",
    "MonolithNLEAudioTrack": "Monolith NLE Audio Track (v3)",
}
