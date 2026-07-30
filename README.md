# Monolith Nodes

Custom nodes for ComfyUI.

## Features

- **SEGS Depth Sorter**: Directly processes Impact Pack SEGS alongside a depth map, calculating depth within the exact segmentation mask, and sorting the SEGS based on distance from the camera.
  - **Inputs:**
    - `segs`: Connect the `SEGS` output from `SegmDetectorSEGS` (Impact Pack).
    - `depth_map`: Connect a depth map image (e.g., from `DepthAnythingV2`).
- **Image List Depth Sorter**: Takes a list of cutout images, finds the median brightness of the non-black pixels, and sorts the list.
  - **Inputs:**
    - `images`: A list of cropped images, typically straight from `ImpactImageBatchToImageList`.

## Requirements

No extra dependencies beyond what ComfyUI and standard extensions already provide.
