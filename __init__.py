from .nodes import SEGSDepthSorter, ImageListDepthSorter

NODE_CLASS_MAPPINGS = {
    "SEGSDepthSorter": SEGSDepthSorter,
    "ImageListDepthSorter": ImageListDepthSorter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SEGSDepthSorter": "SEGS Depth Sorter",
    "ImageListDepthSorter": "Image List Depth Sorter",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
