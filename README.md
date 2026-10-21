# Monolith Nodes (`comfyui-monolith-nodes`)

Modular custom node suite for ComfyUI.

## Architecture

Nodes are grouped by domain under `nodes/` and aggregated in `nodes/__init__.py`:

```text
comfyui-monolith-nodes/
├── __init__.py                 # Package entry point for ComfyUI loader
├── pyproject.toml              # ComfyUI registry & package metadata
├── requirements.txt            # Runtime dependencies
├── fastfacefusion/             # Self-contained FaceSwap ONNX engine & model manager
│   ├── pp_fastfaceswap_engine.py
│   ├── pp_fastfaceswap_models.py
│   └── facefusion/             # Bundled FaceFusion core helpers
├── web/
│   └── js/
│       └── monolith_nle.js     # Interactive NLE timeline canvas & video player widget
└── nodes/
    ├── __init__.py             # Merges domain NODE_CLASS_MAPPINGS
    ├── depth.py                # Monolith/Depth nodes
    ├── safety.py               # Monolith/Safety nodes
    ├── faceswap.py             # Monolith/FaceSwap nodes
    ├── prompt.py               # Monolith/Prompt nodes
    └── nle.py                  # Monolith/NLE timeline assembly nodes
```

### Adding a New Node Domain
1. Create `nodes/<domain>.py` with `<DOMAIN>_CLASS_MAPPINGS` and `<DOMAIN>_NAME_MAPPINGS`.
2. Set `CATEGORY = "Monolith/<Domain>"` on each node class.
3. Import and merge the mappings in `nodes/__init__.py`.

## Included Nodes

### `Monolith/Depth`
- **SEGS Depth Sorter (`SEGSDepthSorter`)**: Processes Impact Pack `SEGS` alongside a depth map, computing median/mean depth inside each segmentation mask and sorting `SEGS` by camera distance.
- **Image List Depth Sorter (`ImageListDepthSorter`)**: Sorts a list of cropped cutout images (`ImpactImageBatchToImageList`) by median brightness of non-black pixels.

### `Monolith/Safety`
- **Monolith Safety Fallback (`MonolithSafetyFallback`)**: Evaluates an image with `AdamCodd/vit-base-nsfw-detector` and returns `fallback_image` (or a black tensor) along with `(IMAGE, is_nsfw, score)` when the NSFW score exceeds `threshold`.

### `Monolith/FaceSwap`
- **Monolith FaceSwap Loader (`MonolithFaceSwapLoader`)**: Preloads swap (`inswapper_128`, `hyperswap_*`, `simswap_*`, etc.) and enhancer (`gfpgan_1.4`, `codeformer`, `gpen_bfr_*`) ONNX sessions into VRAM with configurable pixel boost.
- **Monolith Face Boost (`MonolithFaceBoost`)**: Pre-cursor enhancer/restoration config node that feeds into `MonolithFaceSwap`.
- **Monolith FaceSwap (`MonolithFaceSwap`)**: Swaps a source face onto a target image, target video, or both in a single pass with mask padding presets and NVENC support.
- **Monolith Face Enhancer (`MonolithFaceEnhancer`)**: Standalone face restoration node for static images.

### `Monolith/Prompt`
- **Monolith Prompt Switch By Image (`MonolithPromptSwitchByImage`)**: Walks upstream from an `IMAGE` input wire to resolve the source filename (e.g. from `LoadImage`) and outputs the prompt matching `[filename.png]`, `[stem]`, or `[default]`.
- **Monolith Image Switch By Image (`MonolithImageSwitchByImage`)**: Patch-bay image router using native ComfyUI V3 `Autogrow` slots (`image1..N` and `image1_output..N_output`). Matches the input image against connected candidate images by upstream filename (with pixel-tensor fallback) and outputs `(IMAGE, MASK, detected_filename)` for the winning `imageN_output` (or `default_image`), automatically extracting the upstream `LoadImage` alpha mask without extra mask inputs.

### `Monolith/NLE`
- **Monolith NLE Assembler v1 (`MonolithNLEAssemblerV1`)**: All-in-one FFmpeg NLE timeline composer (base media + alpha overlay + endcard concat + backtimed/faded audio + hardware encoding via `nvenc`/`rkmpp`/`videotoolbox` + midpoint thumbnail extraction).
- **Monolith NLE Assembler v2 (`MonolithNLEAssemblerV2`)**: All-in-one NLE Assembler with an interactive HTML5 multi-track timeline canvas (`V2 OVR`, `V1 VID`, `A1 AUD`) and inline video preview widget.
- **Monolith NLE Assembler v3 (`MonolithNLEAssemblerV3`) & Track Builders (`MonolithNLEOverlayTrack`, `MonolithNLEEndcardTrack`, `MonolithNLEAudioTrack`)**: Modular multi-node NLE timeline architecture accepting dedicated track wires (`MONOLITH_NLE_OVERLAY`, `MONOLITH_NLE_ENDCARD`, `MONOLITH_NLE_AUDIO`).
