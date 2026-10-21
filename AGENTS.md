# `comfyui-monolith-nodes` — Agent Rules & Architecture

## 1. Mission
This custom node suite (`git@github.com:nAtic86/comfyui-monolith-nodes.git`, branch `main`) powers the Enson photobooth `worker_core` (`~/CORES/worker_core`).
Its purpose is to progressively migrate execution modules from `worker_core/backend/apps/` (`pp_fastfaceswap`, `pp_ffmpeg` NLE `renderAssembler`, prompt/image routing, depth sorting, NSFW safety) into modular ComfyUI nodes so complex media/AI pipelines run natively inside ComfyUI workflows (`pp_comfyui`) rather than hardcoded Python wrappers.

## 2. Node Domains & Conventions
- All node classes live under `nodes/<domain>.py` with `CATEGORY = "Monolith/<Domain>"` and are merged in `nodes/__init__.py`.
- Frontend extensions live in `web/js/` (`WEB_DIRECTORY = "./web"` in `__init__.py`).
- Do **not** arbitrarily modify vendored `fastfacefusion/facefusion/` files (stripped FaceFusion 3.6.1 core); make adjustments in `fastfacefusion/pp_fastfaceswap_engine.py`, `fastfacefusion/pp_fastfaceswap_models.py`, or `nodes/faceswap.py`.
- Output nodes consumed by `worker_core` (`pp_comfyui`) must emit `{"ui": {"images": [...], "videos": [...]}, "result": (...)}` with `"type": "output"` (never `"temp"`) so `pp_comfyui._execute_once()` harvests deliverables when nodes are titled `output_image*` or `output_video*`.
- Maintain `README.md` whenever node classes, inputs/outputs, or domains are added or changed.

## 3. Git & Cross-Host Sync (`zurich` <-> `johnmatrix`)
Use `~/comfyui/.agents/workflows/git.md` (or skill `monolith-git`) to commit/push `comfyui-monolith-nodes` to `origin main`, restart local ComfyUI, and automatically pull + restart ComfyUI on the peer GPU host (`zurich` or `johnmatrix`) over SSH.
