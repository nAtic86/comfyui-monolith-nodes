# Monolith Nodes

Custom nodes for ComfyUI.

## Features

- **SEGS Depth Sorter**: Directly processes Impact Pack SEGS alongside a depth map, calculating depth within the exact segmentation mask, and sorting the SEGS based on distance from the camera.
- **Image List Depth Sorter**: Takes a list of cutout images, finds the median brightness of the non-black pixels, and sorts the list.

## Requirements

No extra dependencies beyond what ComfyUI and standard extensions already provide.
