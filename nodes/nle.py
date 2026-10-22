from fractions import Fraction
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import wave
import xml.etree.ElementTree as ET

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

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif")
AUDIO_EXTS = (".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".opus")

ROTATE_MAP = {
    "0": "0",
    "90CW": "1",
    "180": "2",
    "270CW": "3",
    "1": "1",
    "2": "2",
    "3": "3",
}


class _AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = _AnyType("*")


class _FlexibleOptionalDict(dict):
    """Allows dynamic input sockets populated from OTIO/XML timelines to pass ComfyUI input lookup."""

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        return (ANY_TYPE,)


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


def get_timeline_files() -> List[str]:
    found = set()
    for base_dir in (folder_paths.get_input_directory(), folder_paths.get_temp_directory()):
        try:
            if not base_dir or not os.path.isdir(base_dir):
                continue
            for fn in os.listdir(base_dir):
                low = fn.lower()
                if low.endswith((".otio", ".xml")) and os.path.isfile(os.path.join(base_dir, fn)):
                    found.add(fn)
        except Exception:
            pass
    otio_files = sorted(f for f in found if f.lower().endswith(".otio"))
    xml_files = sorted(f for f in found if f.lower().endswith(".xml"))
    return otio_files + xml_files + ["none"]


def _resolve_file_in_dirs(filename: str, search_dirs: List[str]) -> Optional[str]:
    if not filename or filename == "none":
        return None
    clean_name = os.path.basename(filename.replace("\\", "/"))
    for raw_dir in search_dirs:
        if not raw_dir or not os.path.isdir(raw_dir):
            continue
        real_dir = os.path.realpath(raw_dir)
        direct = os.path.realpath(os.path.join(real_dir, clean_name))
        if direct.startswith(real_dir + os.sep) and os.path.isfile(direct):
            return direct
        try:
            all_files = [f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
            exact, canonical = MonolithPromptSwitchByImage._build_lookup([(f, f) for f in all_files])
            resolved_name, matched = MonolithPromptSwitchByImage._resolve_lookup(
                clean_name, exact, canonical, default=""
            )
            if matched and resolved_name:
                cand = os.path.realpath(os.path.join(real_dir, resolved_name))
                if cand.startswith(real_dir + os.sep) and os.path.isfile(cand):
                    return cand
        except Exception:
            pass
    return None


def _resolve_input_file(filename: str, extra_dirs: Optional[List[str]] = None) -> Optional[str]:
    dirs = list(extra_dirs or []) + [
        folder_paths.get_input_directory(),
        folder_paths.get_temp_directory(),
    ]
    return _resolve_file_in_dirs(filename, dirs)


def _resolve_timeline_file(filename: str) -> Optional[str]:
    return _resolve_file_in_dirs(
        filename,
        [folder_paths.get_input_directory(), folder_paths.get_temp_directory()],
    )


def _parse_fps(fps_str: Any) -> float:
    try:
        if "/" in str(fps_str):
            num, den = str(fps_str).split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 24.0
        return float(fps_str)
    except Exception:
        return 24.0


def probe_media_specs(
    path: str, ffprobe_bin: str = "ffprobe", still_duration: float = 4.0
) -> Tuple[int, int, str, float, bool]:
    ext = os.path.splitext(path)[1].lower()
    is_image = ext in IMAGE_EXTS

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
    fps = stream.get("r_frame_rate", "24/1")
    width = int(stream.get("width", 720))
    height = int(stream.get("height", 1280))

    if is_image:
        return width, height, "24/1", float(still_duration), True

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

        timestamp = (duration / 2.0) if duration > 0 else 0.0
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


def _extract_filename_from_url(target_url: str, fallback_name: str = "") -> str:
    if target_url:
        unquoted = urllib.parse.unquote(target_url)
        if unquoted.startswith("file://"):
            unquoted = unquoted[7:]
        base = os.path.basename(unquoted.replace("\\", "/"))
        if base:
            return base
    return os.path.basename(fallback_name.replace("\\", "/")) if fallback_name else "clip"


def _pair_matte_tracks_and_build_inputs(
    video_tracks: List[Dict[str, Any]],
    audio_tracks: List[Dict[str, Any]],
    rate: float,
    total_frames: int,
    width: int,
    height: int,
    timeline_dir: str,
) -> Dict[str, Any]:
    # Pair alpha/matte clips across adjacent video tracks:
    # Rule A: Premiere effect "Alpha Adjust" (AE.ADBE Alpha Adjust) on upper clip or "Track Matte Key" on lower clip
    # Rule B: Upper clip filename stem ends with _alpha, _matte, or _mask over the same timeline span
    for i in range(len(video_tracks) - 1):
        lower_track = video_tracks[i]
        upper_track = video_tracks[i + 1]
        for lclip in lower_track["clips"]:
            if lclip.get("matte_clip") is not None:
                continue
            has_track_matte_fx = any(
                "Track Matte" in en or "Track Matte" in mn
                for en, mn in lclip.get("effects", [])
            )
            for uclip in list(upper_track["clips"]):
                has_alpha_fx = any(
                    "Alpha Adjust" in en or "Alpha Adjust" in mn
                    for en, mn in uclip.get("effects", [])
                )
                stem = os.path.splitext(uclip["filename"].lower())[0]
                is_alpha_named = stem.endswith(("_alpha", "_matte", "_mask"))
                same_span = (
                    abs(lclip["timeline_start_f"] - uclip["timeline_start_f"]) <= 1.0
                    and abs(lclip["timeline_end_f"] - uclip["timeline_end_f"]) <= 1.0
                )
                if same_span and (has_alpha_fx or has_track_matte_fx or is_alpha_named):
                    uclip["role"] = "matte"
                    lclip["matte_clip"] = uclip
                    upper_track["clips"].remove(uclip)
                    break

    active_video_tracks = [t for t in video_tracks if t["clips"]]

    # If resolution wasn't stored in sequence header (e.g. DaVinci Resolve OTIO), probe first resolved video clip
    if width <= 0 or height <= 0:
        width, height = 720, 1280
        for tr in active_video_tracks:
            for cl in tr["clips"]:
                resolved = _resolve_input_file(cl["filename"], [timeline_dir])
                if resolved:
                    try:
                        pw, ph, _, _, _ = probe_media_specs(resolved, get_ffprobe_bin())
                        if pw > 0 and ph > 0:
                            width, height = pw, ph
                            break
                    except Exception:
                        pass
            else:
                continue
            break

    # Build ordered unique input socket declarations and visual track summary
    inputs_spec: List[Dict[str, Any]] = []
    seen_names = set()
    ui_tracks: List[Dict[str, Any]] = []

    def _add_input_spec(clip_dict: Dict[str, Any], role: str, track_label: str):
        fn = clip_dict["filename"]
        if fn in seen_names:
            return
        seen_names.add(fn)
        ext = os.path.splitext(fn)[1].lower()
        if role == "matte":
            sock_type = "MASK,VIDEO,IMAGE"
        elif role == "audio" or ext in AUDIO_EXTS:
            sock_type = "AUDIO"
        elif ext in IMAGE_EXTS:
            sock_type = "IMAGE,VIDEO"
            role = "image"
        else:
            sock_type = "VIDEO,IMAGE"
        resolved_path = _resolve_input_file(fn, [timeline_dir])
        inputs_spec.append({
            "name": fn,
            "type": sock_type,
            "role": role,
            "track": track_label,
            "start": round(clip_dict["timeline_start_f"] / rate, 3),
            "end": round(clip_dict["timeline_end_f"] / rate, 3),
            "duration": round(clip_dict["timeline_dur_f"] / rate, 3),
            "src_in": round(clip_dict["src_in_f"] / rate, 3),
            "resolved": bool(resolved_path),
            "matte": clip_dict["matte_clip"]["filename"] if clip_dict.get("matte_clip") else None,
        })

    for v_idx, tr in enumerate(active_video_tracks):
        t_label = f"V{v_idx + 1}"
        ui_clips = []
        for cl in tr["clips"]:
            _add_input_spec(cl, "video", t_label)
            matte_cl = cl.get("matte_clip")
            if matte_cl:
                _add_input_spec(matte_cl, "matte", f"{t_label}M")
            ui_clips.append({
                "name": cl["filename"],
                "start": round(cl["timeline_start_f"] / rate, 3),
                "end": round(cl["timeline_end_f"] / rate, 3),
                "duration": round(cl["timeline_dur_f"] / rate, 3),
                "matte": matte_cl["filename"] if matte_cl else None,
                "is_image": os.path.splitext(cl["filename"])[1].lower() in IMAGE_EXTS,
            })
        ui_tracks.append({
            "id": t_label,
            "name": tr.get("name") or t_label,
            "kind": "video",
            "clips": ui_clips,
        })

    for a_idx, tr in enumerate(audio_tracks):
        t_label = f"A{a_idx + 1}"
        ui_clips = []
        for cl in tr["clips"]:
            _add_input_spec(cl, "audio", t_label)
            ui_clips.append({
                "name": cl["filename"],
                "start": round(cl["timeline_start_f"] / rate, 3),
                "end": round(cl["timeline_end_f"] / rate, 3),
                "duration": round(cl["timeline_dur_f"] / rate, 3),
                "src_in": round(cl["src_in_f"] / rate, 3),
            })
        ui_tracks.append({
            "id": t_label,
            "name": tr.get("name") or t_label,
            "kind": "audio",
            "clips": ui_clips,
        })

    total_dur = total_frames / rate if rate > 0 else 0.0
    return {
        "valid": True,
        "width": width,
        "height": height,
        "rate": rate,
        "fps": round(rate, 2),
        "total_frames": total_frames,
        "total_dur": round(total_dur, 4),
        "duration_s": total_dur,
        "video_tracks": active_video_tracks,
        "audio_tracks": audio_tracks,
        "inputs": inputs_spec,
        "ui_tracks": ui_tracks,
    }


def parse_nle_timeline(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Monolith NLE: timeline file not found: {path}")

    timeline_dir = os.path.dirname(os.path.realpath(path))
    ext = os.path.splitext(path)[1].lower()

    if ext == ".xml":
        return _parse_fcp7_xml_timeline(path, timeline_dir)
    return _parse_otio_json_timeline(path, timeline_dir)


def _parse_otio_json_timeline(path: str, timeline_dir: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks_data = data.get("tracks", {})
    meta = tracks_data.get("metadata", {}).get("PremierePro_OTIO", {})
    vr = meta.get("VideoResolution", {})
    width = int(vr.get("Width", 0)) if vr else 0
    height = int(vr.get("Height", 0)) if vr else 0

    video_tracks: List[Dict[str, Any]] = []
    audio_tracks: List[Dict[str, Any]] = []
    rate = 24.0
    total_frames = 0

    for track in tracks_data.get("children", []):
        if not str(track.get("OTIO_SCHEMA", "")).startswith("Track.") or not track.get("enabled", True):
            continue
        kind = track.get("kind", "Video")
        cursor_frames = 0.0
        clips = []
        for item in track.get("children", []):
            schema = str(item.get("OTIO_SCHEMA", ""))
            sr = item.get("source_range")
            if not sr:
                continue
            dur = float(sr["duration"]["value"])
            item_rate = float(sr["duration"].get("rate", 24.0))
            if item_rate > 0:
                rate = item_rate
            src_in = float(sr["start_time"]["value"])

            if schema.startswith("Gap."):
                cursor_frames += dur
            elif schema.startswith("Clip."):
                if item.get("enabled", True):
                    mrs = item.get("media_references", {})
                    mr = mrs.get("DEFAULT_MEDIA") or (next(iter(mrs.values())) if mrs else {})
                    target_url = mr.get("target_url", "")
                    filename = _extract_filename_from_url(target_url, item.get("name", ""))
                    effects = []
                    for eff in item.get("effects", []):
                        eff_name = eff.get("effect_name", "")
                        match_name = (
                            eff.get("metadata", {})
                            .get("PremierePro_OTIO", {})
                            .get("Effect", {})
                            .get("MatchName", "")
                        )
                        effects.append((eff_name, match_name))
                    clips.append({
                        "name": item.get("name") or filename,
                        "filename": filename,
                        "target_url": target_url,
                        "timeline_start_f": cursor_frames,
                        "timeline_end_f": cursor_frames + dur,
                        "timeline_dur_f": dur,
                        "src_in_f": src_in,
                        "rate": rate,
                        "effects": effects,
                    })
                cursor_frames += dur
        total_frames = max(total_frames, int(round(cursor_frames)))
        if kind == "Video" and clips:
            video_tracks.append({"name": track.get("name", ""), "clips": clips})
        elif kind == "Audio" and clips:
            audio_tracks.append({"name": track.get("name", ""), "clips": clips})

    return _pair_matte_tracks_and_build_inputs(
        video_tracks, audio_tracks, rate, total_frames, width, height, timeline_dir
    )


def _parse_fcp7_xml_timeline(path: str, timeline_dir: str) -> Dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    seq = root.find(".//sequence")
    if seq is None:
        raise ValueError(f"Monolith NLE: no <sequence> found in XML timeline {path}")

    rate = float(seq.findtext("./rate/timebase", "24.0"))
    total_frames = int(float(seq.findtext("./duration", "0")))
    width = int(float(seq.findtext("./media/video/format/samplecharacteristics/width", "0")))
    height = int(float(seq.findtext("./media/video/format/samplecharacteristics/height", "0")))

    file_url_map: Dict[str, str] = {}
    for file_el in seq.findall(".//file"):
        fid = file_el.get("id")
        purl = file_el.findtext("pathurl", "")
        if fid and purl:
            file_url_map[fid] = purl

    video_tracks: List[Dict[str, Any]] = []
    for t_idx, vtrack in enumerate(seq.findall("./media/video/track")):
        if vtrack.findtext("enabled", "TRUE").upper() == "FALSE":
            continue
        clips = []
        for ci in vtrack.findall("clipitem"):
            if ci.findtext("enabled", "TRUE").upper() == "FALSE":
                continue
            start_f = float(ci.findtext("start", "0"))
            end_f = float(ci.findtext("end", "0"))
            in_f = float(ci.findtext("in", "0"))
            dur_f = max(0.0, end_f - start_f)
            file_el = ci.find("file")
            purl = ""
            if file_el is not None:
                purl = file_el.findtext("pathurl", "") or file_url_map.get(file_el.get("id", ""), "")
            cname = ci.findtext("name", "")
            filename = _extract_filename_from_url(purl, cname)
            total_frames = max(total_frames, int(round(end_f)))
            clips.append({
                "name": cname or filename,
                "filename": filename,
                "target_url": purl,
                "timeline_start_f": start_f,
                "timeline_end_f": end_f,
                "timeline_dur_f": dur_f,
                "src_in_f": in_f,
                "rate": rate,
                "effects": [],
            })
        if clips:
            video_tracks.append({"name": f"Video {t_idx + 1}", "clips": clips})

    audio_tracks: List[Dict[str, Any]] = []
    seen_audio_spans = set()
    for t_idx, atrack in enumerate(seq.findall("./media/audio/track")):
        if atrack.findtext("enabled", "TRUE").upper() == "FALSE":
            continue
        clips = []
        for ci in atrack.findall("clipitem"):
            if ci.findtext("enabled", "TRUE").upper() == "FALSE":
                continue
            start_f = float(ci.findtext("start", "0"))
            end_f = float(ci.findtext("end", "0"))
            in_f = float(ci.findtext("in", "0"))
            dur_f = max(0.0, end_f - start_f)
            file_el = ci.find("file")
            purl = ""
            if file_el is not None:
                purl = file_el.findtext("pathurl", "") or file_url_map.get(file_el.get("id", ""), "")
            cname = ci.findtext("name", "")
            filename = _extract_filename_from_url(purl, cname)
            dedup_key = (filename, round(start_f, 2), round(end_f, 2), round(in_f, 2))
            if dedup_key in seen_audio_spans:
                continue
            seen_audio_spans.add(dedup_key)
            total_frames = max(total_frames, int(round(end_f)))
            clips.append({
                "name": cname or filename,
                "filename": filename,
                "target_url": purl,
                "timeline_start_f": start_f,
                "timeline_end_f": end_f,
                "timeline_dur_f": dur_f,
                "src_in_f": in_f,
                "rate": rate,
                "effects": [],
            })
        if clips:
            audio_tracks.append({"name": f"Audio {len(audio_tracks) + 1}", "clips": clips})

    return _pair_matte_tracks_and_build_inputs(
        video_tracks, audio_tracks, rate, total_frames, width, height, timeline_dir
    )


def _materialize_video_or_image(
    video_obj: Optional[Any],
    image_tensor: Optional[torch.Tensor],
    file_name: str,
    temp_dir: str,
    prefix: str,
    extra_dirs: Optional[List[str]] = None,
) -> Optional[str]:
    if video_obj is not None:
        if isinstance(video_obj, torch.Tensor):
            image_tensor = video_obj
            video_obj = None
        else:
            if hasattr(video_obj, "get_stream_source"):
                trim_win = (
                    video_obj.get_active_trim_window()
                    if hasattr(video_obj, "get_active_trim_window")
                    else (0.0, 0.0)
                )
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
                Types.VideoComponents(images=image_tensor, frame_rate=Fraction(24))
            )
            vid.save_to(out_mp4)
            return out_mp4

    return _resolve_input_file(file_name, extra_dirs=extra_dirs)


def _materialize_mask(
    mask_input: Optional[Any],
    alpha_file_name: str,
    invert_mask: bool,
    temp_dir: str,
    prefix: str = "overlay_alpha",
    extra_dirs: Optional[List[str]] = None,
) -> Tuple[Optional[str], bool]:
    if mask_input is not None:
        if not isinstance(mask_input, torch.Tensor):
            vid_path = _materialize_video_or_image(mask_input, None, "none", temp_dir, prefix, extra_dirs=extra_dirs)
            if vid_path:
                return vid_path, bool(invert_mask)
        elif mask_input.numel() > 0:
            m = mask_input.detach().cpu()
            if m.ndim == 4 and m.shape[-1] in (3, 4):
                img_path = _materialize_video_or_image(None, m, "none", temp_dir, prefix, extra_dirs=extra_dirs)
                if img_path:
                    return img_path, bool(invert_mask)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            elif m.ndim == 4:
                m = m[..., 0]
            if m.shape[-2:] == (64, 64) and float(m.max().item()) == 0.0 and not invert_mask:
                resolved = _resolve_input_file(alpha_file_name, extra_dirs=extra_dirs)
                return resolved, bool(invert_mask) if resolved else False

            alpha = m if invert_mask else (1.0 - m)
            if alpha.shape[0] > 1 and Types is not None:
                rgb_batch = alpha.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
                vid_path = _materialize_video_or_image(None, rgb_batch, "none", temp_dir, prefix, extra_dirs=extra_dirs)
                return vid_path, False

            arr = np.clip(alpha[0].numpy() * 255.0, 0, 255).astype(np.uint8)
            out_png = os.path.join(temp_dir, f"{prefix}.png")
            Image.fromarray(arr, mode="L").save(out_png)
            return out_png, False

    resolved = _resolve_input_file(alpha_file_name, extra_dirs=extra_dirs)
    return resolved, bool(invert_mask) if resolved else False


def _materialize_audio(
    audio_obj: Optional[Any],
    audio_file_name: str,
    temp_dir: str,
    prefix: str = "audio_track",
    extra_dirs: Optional[List[str]] = None,
) -> Optional[str]:
    if audio_obj is not None:
        if isinstance(audio_obj, dict) and "waveform" in audio_obj:
            waveform = audio_obj["waveform"].detach().cpu()
            sample_rate = int(audio_obj.get("sample_rate", 44100))
            if waveform.ndim == 3:
                waveform = waveform[0]
            pcm = np.clip(waveform.t().numpy(), -1.0, 1.0)
            pcm_int16 = (pcm * 32767.0).astype(np.int16)
            channels = pcm_int16.shape[1] if pcm_int16.ndim == 2 else 1
            out_wav = os.path.join(temp_dir, f"{prefix}.wav")
            with wave.open(out_wav, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_int16.tobytes())
            return out_wav
        vid_path = _materialize_video_or_image(audio_obj, None, "none", temp_dir, prefix, extra_dirs=extra_dirs)
        if vid_path:
            return vid_path

    return _resolve_input_file(audio_file_name, extra_dirs=extra_dirs)


def _resolve_upstream_load_image_path(
    input_name: str,
    prompt: Optional[Dict[str, Any]],
    unique_id: Optional[str],
) -> Optional[str]:
    if not prompt or not unique_id or str(unique_id) not in prompt:
        return None
    try:
        node_inputs = prompt[str(unique_id)].get("inputs", {})
        link = node_inputs.get(input_name)
        if isinstance(link, list) and len(link) == 2:
            src_node_id = str(link[0])
            src_node = prompt.get(src_node_id, {})
            if src_node.get("class_type") == "LoadImage":
                img_name = src_node.get("inputs", {}).get("image")
                if img_name and folder_paths.exists_annotated_filepath(img_name):
                    return folder_paths.get_annotated_filepath(img_name)
    except Exception:
        pass
    return None


def execute_otio_timeline_render(
    timeline_path: str,
    output_path: str,
    connected_inputs: Dict[str, Any],
    temp_dir: str,
    hw_encoder: str = "auto",
    invert_alpha_matte: bool = False,
    prompt: Optional[Dict[str, Any]] = None,
    unique_id: Optional[str] = None,
) -> Dict[str, Any]:
    ffmpeg_bin = get_ffmpeg_bin()
    if not isinstance(hw_encoder, str) or hw_encoder not in ("auto", "nvenc", "rkmpp", "videotoolbox", "cpu"):
        hw_encoder = "auto"
    actual_hw = detect_working_hw_encoder() if hw_encoder == "auto" else hw_encoder
    timeline_dir = os.path.dirname(os.path.realpath(timeline_path))
    timeline = parse_nle_timeline(timeline_path)

    w = max(2, int(timeline["width"]) - (int(timeline["width"]) % 2))
    h = max(2, int(timeline["height"]) - (int(timeline["height"]) % 2))
    fps = float(timeline["rate"])
    total_frames = int(timeline["total_frames"])
    dur_s = float(timeline["duration_s"])

    # Support legacy media_video / media_image fallback onto the first V1 clip if present
    if timeline["video_tracks"] and timeline["video_tracks"][0]["clips"]:
        first_v1_name = timeline["video_tracks"][0]["clips"][0]["filename"]
        if first_v1_name not in connected_inputs or connected_inputs[first_v1_name] is None:
            for legacy_key in ("media_video", "media_image"):
                if connected_inputs.get(legacy_key) is not None:
                    connected_inputs[first_v1_name] = connected_inputs[legacy_key]
                    break

    raw_stdin_bytes: Optional[bytes] = None
    raw_stdin_spec: Optional[Tuple[int, int, float]] = None
    embedded_audio_fallback: Optional[str] = None

    cmd_inputs: List[str] = [
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={w}x{h}:r={fps}:d={dur_s:.6f}",
    ]
    input_idx = 1
    filter_parts = ["[0:v]format=yuv420p[base0]"]
    current_base = "base0"
    overlay_idx = 0

    def _add_video_or_image_input(clip_dict: Dict[str, Any], prefix: str) -> Tuple[int, bool]:
        nonlocal input_idx, raw_stdin_bytes, raw_stdin_spec, embedded_audio_fallback
        fn = clip_dict["filename"]
        val = connected_inputs.get(fn)

        # Direct in-memory pipe for the first unencoded VideoFromComponents input (e.g. MonolithFaceSwap)
        if (
            raw_stdin_bytes is None
            and val is not None
            and hasattr(InputImpl, "VideoFromComponents")
            and isinstance(val, InputImpl.VideoFromComponents)
            and hasattr(val, "get_components")
        ):
            comp = val.get_components()
            frames_np = np.clip(comp.images.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            in_fps = float(comp.frame_rate) if comp.frame_rate else fps
            raw_stdin_bytes = frames_np.tobytes()
            raw_stdin_spec = (int(frames_np.shape[2]), int(frames_np.shape[1]), in_fps)
            if comp.audio is not None and embedded_audio_fallback is None:
                embedded_audio_fallback = _materialize_audio(comp.audio, "none", temp_dir, prefix="emb_audio")
            cmd_inputs.extend([
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{raw_stdin_spec[0]}x{raw_stdin_spec[1]}",
                "-r", str(raw_stdin_spec[2]),
                "-i", "-",
            ])
            idx = input_idx
            input_idx += 1
            return idx, False

        # Preserve RGBA transparency when an upstream LoadImage node is wired into an image clip socket
        fpath = _resolve_upstream_load_image_path(fn, prompt, unique_id)
        if not fpath:
            fpath = _materialize_video_or_image(
                val, None, fn, temp_dir, prefix, extra_dirs=[timeline_dir]
            )
        if not fpath:
            raise ValueError(
                f"Monolith NLE: missing media input for timeline clip '{fn}'. "
                f"Connect an input to '{fn}' or place '{fn}' in ComfyUI/input/."
            )
        ext = os.path.splitext(fpath)[1].lower()
        is_img = ext in IMAGE_EXTS
        c_dur = clip_dict["timeline_dur_f"] / fps
        if is_img:
            cmd_inputs.extend(["-loop", "1", "-framerate", str(fps), "-t", f"{c_dur:.6f}", "-i", fpath])
        else:
            cmd_inputs.extend(["-i", fpath])
        idx = input_idx
        input_idx += 1
        return idx, is_img

    for t_idx, track in enumerate(timeline["video_tracks"]):
        for c_idx, clip in enumerate(track["clips"]):
            t_start = clip["timeline_start_f"] / fps
            t_end = clip["timeline_end_f"] / fps
            c_dur = clip["timeline_dur_f"] / fps
            src_in = clip["src_in_f"] / fps

            clip_in_idx, is_image = _add_video_or_image_input(clip, f"v_{t_idx}_{c_idx}")
            eff_src_in = 0.0 if is_image else src_in

            matte = clip.get("matte_clip")
            if matte:
                m_fn = matte["filename"]
                m_val = connected_inputs.get(m_fn)
                mpath, inv_matte = _materialize_mask(
                    m_val,
                    m_fn,
                    invert_alpha_matte,
                    temp_dir,
                    prefix=f"m_{t_idx}_{c_idx}",
                    extra_dirs=[timeline_dir],
                )
                if not mpath:
                    raise ValueError(
                        f"Monolith NLE: missing alpha matte input '{m_fn}' for clip '{clip['filename']}'."
                    )
                m_ext = os.path.splitext(mpath)[1].lower()
                m_is_img = m_ext in IMAGE_EXTS
                m_dur = matte["timeline_dur_f"] / fps
                m_src_in = 0.0 if m_is_img else (matte["src_in_f"] / fps)
                if m_is_img:
                    cmd_inputs.extend(["-loop", "1", "-framerate", str(fps), "-t", f"{m_dur:.6f}", "-i", mpath])
                else:
                    cmd_inputs.extend(["-i", mpath])
                matte_in_idx = input_idx
                input_idx += 1
                neg_str = ",negate" if inv_matte else ""

                filter_parts.append(
                    f"[{clip_in_idx}:v]trim=start={eff_src_in:.6f}:duration={c_dur:.6f},"
                    f"setpts=PTS-STARTPTS,fps={fps},scale={w}:{h},setsar=1,format=rgba[vfill_{overlay_idx}]"
                )
                filter_parts.append(
                    f"[{matte_in_idx}:v]trim=start={m_src_in:.6f}:duration={m_dur:.6f},"
                    f"setpts=PTS-STARTPTS,fps={fps},scale={w}:{h},setsar=1,format=gray{neg_str}[vmatte_{overlay_idx}]"
                )
                filter_parts.append(
                    f"[vfill_{overlay_idx}][vmatte_{overlay_idx}]alphamerge,"
                    f"setpts=PTS+{t_start:.6f}/TB[vlayer_{overlay_idx}]"
                )
            else:
                pix_fmt = "rgba" if is_image else "yuv420p"
                filter_parts.append(
                    f"[{clip_in_idx}:v]trim=start={eff_src_in:.6f}:duration={c_dur:.6f},"
                    f"setpts=PTS-STARTPTS,fps={fps},scale={w}:{h},setsar=1,format={pix_fmt},"
                    f"setpts=PTS+{t_start:.6f}/TB[vlayer_{overlay_idx}]"
                )

            next_base = f"base{overlay_idx + 1}"
            filter_parts.append(
                f"[{current_base}][vlayer_{overlay_idx}]overlay=0:0:"
                f"enable='between(t,{t_start:.6f},{t_end:.6f})':eof_action=pass[{next_base}]"
            )
            current_base = next_base
            overlay_idx += 1

    audio_labels: List[str] = []
    for a_t_idx, atrack in enumerate(timeline["audio_tracks"]):
        for a_c_idx, aclip in enumerate(atrack["clips"]):
            a_fn = aclip["filename"]
            a_val = connected_inputs.get(a_fn)
            if a_val is None and connected_inputs.get("audio") is not None:
                a_val = connected_inputs["audio"]
            apath = _materialize_audio(
                a_val,
                a_fn,
                temp_dir,
                prefix=f"a_{a_t_idx}_{a_c_idx}",
                extra_dirs=[timeline_dir],
            )
            if not apath and embedded_audio_fallback:
                apath = embedded_audio_fallback
            if not apath:
                continue

            cmd_inputs.extend(["-i", apath])
            a_idx = input_idx
            input_idx += 1
            a_src_in = aclip["src_in_f"] / fps
            a_dur = aclip["timeline_dur_f"] / fps
            a_delay_ms = int(round((aclip["timeline_start_f"] / fps) * 1000.0))
            alabel = f"aud_{len(audio_labels)}"
            filter_parts.append(
                f"[{a_idx}:a]atrim=start={a_src_in:.6f}:duration={a_dur:.6f},"
                f"asetpts=PTS-STARTPTS,adelay={a_delay_ms}|{a_delay_ms},"
                f"apad=whole_dur={dur_s:.6f},asetpts=N/SR/TB[{alabel}]"
            )
            audio_labels.append(alabel)

    has_audio = len(audio_labels) > 0
    if len(audio_labels) == 1:
        audio_map_label = f"[{audio_labels[0]}]"
    elif len(audio_labels) > 1:
        mix_in = "".join(f"[{lbl}]" for lbl in audio_labels)
        filter_parts.append(
            f"{mix_in}amix=inputs={len(audio_labels)}:duration=first:dropout_transition=0,asetpts=N/SR/TB[aout_mix]"
        )
        audio_map_label = "[aout_mix]"
    else:
        audio_map_label = ""

    cmd = [ffmpeg_bin, "-y"] + cmd_inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", f"[{current_base}]",
    ]
    if has_audio:
        cmd.extend(["-map", audio_map_label, "-c:a", "aac", "-b:a", "192k"])

    if actual_hw == "nvenc":
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-b:v", "10M",
            "-maxrate", "12M",
            "-bufsize", "20M",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ])
    elif actual_hw in ("videotoolbox", "rkmpp"):
        cmd.extend([
            "-c:v", f"h264_{actual_hw}",
            "-b:v", "10M",
            "-maxrate", "12M",
            "-bufsize", "20M",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-b:v", "10M",
            "-maxrate", "12M",
            "-bufsize", "20M",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ])

    cmd.extend([
        "-r", str(fps),
        "-frames:v", str(total_frames),
        "-movflags", "+faststart",
        output_path,
    ])

    if raw_stdin_bytes is not None:
        res = subprocess.run(cmd, input=raw_stdin_bytes, capture_output=True)
        if res.returncode != 0:
            err_txt = res.stderr.decode("utf-8", errors="replace")[-600:]
            raise RuntimeError(f"Monolith NLE OTIO FFmpeg render failed ({actual_hw}): {err_txt}")
    else:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Monolith NLE OTIO FFmpeg render failed ({actual_hw}): {res.stderr[-600:]}")

    thumb_ts = min(2.0, dur_s / 2.0) if dur_s > 0 else 0.0
    return {
        "mode": "otio",
        "timeline_file": os.path.basename(timeline_path),
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "media_dur": round(dur_s, 2),
        "endcard_dur": 0.0,
        "total_dur": round(dur_s, 2),
        "hw_encoder": actual_hw,
        "thumb_ts": round(thumb_ts, 2),
        "ui_tracks": timeline["ui_tracks"],
        "inputs": timeline["inputs"],
    }


def _shared_otio_input_types() -> Dict[str, Any]:
    timeline_files = get_timeline_files()
    default_tl = (
        "dutton_ranch_premiere.otio"
        if "dutton_ranch_premiere.otio" in timeline_files
        else (timeline_files[0] if timeline_files else "none")
    )
    return {
        "required": {
            "timeline_file": (timeline_files, {"default": default_tl}),
            "hw_encoder": (["auto", "nvenc", "rkmpp", "videotoolbox", "cpu"], {"default": "auto"}),
            "invert_alpha_matte": ("BOOLEAN", {"default": False}),
        },
        "optional": _FlexibleOptionalDict(),
        "hidden": {
            "prompt": "PROMPT",
            "unique_id": "UNIQUE_ID",
        },
    }


def _run_otio_assembler(
    timeline_file: str = "none",
    hw_encoder: str = "auto",
    invert_alpha_matte: bool = False,
    prompt: Optional[Dict[str, Any]] = None,
    unique_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    timeline_path = _resolve_timeline_file(timeline_file)
    if not timeline_path:
        raise ValueError(
            f"Monolith NLE Assembler: select a valid .otio or .xml timeline_file (got '{timeline_file}')."
        )

    temp_root = folder_paths.get_temp_directory()
    os.makedirs(temp_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="monolith_nle_otio_", dir=temp_root)

    try:
        full_out_dir, base_fn, counter, subf, _ = folder_paths.get_save_image_path(
            "monolith_nle", folder_paths.get_output_directory()
        )
        os.makedirs(full_out_dir, exist_ok=True)
        out_video_fn = f"{base_fn}_{counter:05}_.mp4"
        out_video_path = os.path.join(full_out_dir, out_video_fn)

        timeline_meta = execute_otio_timeline_render(
            timeline_path=timeline_path,
            output_path=out_video_path,
            connected_inputs=kwargs,
            temp_dir=temp_dir,
            hw_encoder=hw_encoder,
            invert_alpha_matte=invert_alpha_matte,
            prompt=prompt,
            unique_id=unique_id,
        )

        thumb_path = os.path.join(temp_dir, "thumbnail.jpg")
        if extract_middle_frame(out_video_path, thumb_path, media_dur=timeline_meta["total_dur"]):
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
    OTIO / XML Timeline-driven NLE Assembler node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _shared_otio_input_types()

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "thumbnail")
    OUTPUT_NODE = True
    FUNCTION = "assemble"
    CATEGORY = "Monolith/NLE"

    def assemble(self, timeline_file: str = "none", hw_encoder: str = "auto", invert_alpha_matte: bool = False, **kwargs):
        return _run_otio_assembler(
            timeline_file=timeline_file,
            hw_encoder=hw_encoder,
            invert_alpha_matte=invert_alpha_matte,
            **kwargs,
        )


class MonolithNLEAssemblerV2(MonolithNLEAssemblerV1):
    """
    OTIO / XML Timeline-driven NLE Assembler node with interactive multi-track visual timeline canvas.
    """
    pass


class MonolithNLEOverlayTrack:
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
                "overlay_mask": ("MASK,VIDEO,IMAGE",),
                "overlay_file": (media_files, {"default": "none"}),
                "overlay_alpha_file": (media_files, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("MONOLITH_NLE_OVERLAY",)
    RETURN_NAMES = ("overlay_track",)
    FUNCTION = "build_track"
    CATEGORY = "Monolith/NLE"

    def build_track(self, **kwargs):
        return (kwargs,)


class MonolithNLEEndcardTrack:
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

    def build_track(self, **kwargs):
        return (kwargs,)


class MonolithNLEAudioTrack:
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

    def build_track(self, **kwargs):
        return (kwargs,)


class MonolithNLEAssemblerV3(MonolithNLEAssemblerV1):
    """
    OTIO / XML Timeline-driven NLE Assembler (v3 alias).
    """
    pass


# Register HTTP route for frontend dynamic socket & timeline discovery
try:
    from aiohttp import web
    from server import PromptServer

    if PromptServer.instance is not None:
        @PromptServer.instance.routes.get("/monolith/nle/timeline_info")
        async def get_nle_timeline_info(request):
            filename = request.rel_url.query.get("file", "none")
            tl_path = _resolve_timeline_file(filename)
            if not tl_path:
                return web.json_response({"valid": False, "error": f"Timeline not found: {filename}"})
            try:
                parsed = parse_nle_timeline(tl_path)
                return web.json_response({
                    "valid": True,
                    "file": os.path.basename(tl_path),
                    "width": parsed["width"],
                    "height": parsed["height"],
                    "fps": parsed["fps"],
                    "total_frames": parsed["total_frames"],
                    "total_dur": parsed["total_dur"],
                    "inputs": parsed["inputs"],
                    "ui_tracks": parsed["ui_tracks"],
                })
            except Exception as exc:
                logger.exception("Monolith NLE: failed to parse timeline %s", filename)
                return web.json_response({"valid": False, "error": str(exc)}, status=400)
except Exception:
    pass


NLE_CLASS_MAPPINGS = {
    "MonolithNLEAssemblerV1": MonolithNLEAssemblerV1,
    "MonolithNLEAssemblerV2": MonolithNLEAssemblerV2,
    "MonolithNLEAssemblerV3": MonolithNLEAssemblerV3,
    "MonolithNLEOverlayTrack": MonolithNLEOverlayTrack,
    "MonolithNLEEndcardTrack": MonolithNLEEndcardTrack,
    "MonolithNLEAudioTrack": MonolithNLEAudioTrack,
}

NLE_NAME_MAPPINGS = {
    "MonolithNLEAssemblerV1": "Monolith NLE Assembler v1 (OTIO Timeline)",
    "MonolithNLEAssemblerV2": "Monolith NLE Assembler v2 (OTIO Visual Timeline)",
    "MonolithNLEAssemblerV3": "Monolith NLE Assembler v3 (OTIO Modular)",
    "MonolithNLEOverlayTrack": "Monolith NLE Overlay Track (v3)",
    "MonolithNLEEndcardTrack": "Monolith NLE Endcard Track (v3)",
    "MonolithNLEAudioTrack": "Monolith NLE Audio Track (v3)",
}
