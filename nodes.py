import torch
import torch.nn.functional as F
import numpy as np

class ImageListDepthSorter:
    DESCRIPTION = "Takes a list of cutout images (as output by ImpactImageBatchToImageList), finds the median brightness of the non-black pixels (the person), and sorts the list."
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "A list of cropped images, typically from ImpactImageBatchToImageList."}),
                "order": (["descending (nearest first)", "ascending (farthest first)"], {"tooltip": "Sort order. Descending puts the brightest (closest) objects first."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("The sorted list of images.",)
    FUNCTION = "sort_images"
    CATEGORY = "Monolith"

    def sort_images(self, images, order):
        reverse_sort = order[0] == "descending (nearest first)"
        
        def get_median(img_tensor):
            if len(img_tensor.shape) == 4 and img_tensor.shape[3] >= 3:
                gray = img_tensor.mean(dim=3)
            else:
                gray = img_tensor
                
            valid_pixels = gray[gray > 0.01]
            if valid_pixels.numel() == 0:
                return 0.0
                
            return valid_pixels.median().item()
        
        images_with_depth = [(img, get_median(img)) for img in images]
        images_with_depth.sort(key=lambda x: x[1], reverse=reverse_sort)
        
        sorted_images = [item[0] for item in images_with_depth]
        return (sorted_images, )


class SEGSDepthSorter:
    DESCRIPTION = "Directly processes Impact Pack SEGS alongside a depth map, calculating depth within the exact segmentation mask, sorting, and returning SEGS."

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", {"tooltip": "The SEGS output from SegmDetectorSEGS (Impact Pack)."}),
                        "depth_map": ("IMAGE", {"tooltip": "A depth map image (e.g. from DepthAnythingV2)."}),
                        "sort_method": (["median", "mean"], {"tooltip": "How to calculate the depth value of the mask."}),
                        "order": (["descending (nearest first)", "ascending (farthest first)"], {"tooltip": "Sort order. Descending puts the brightest (closest) objects first."}),
                     },
                }

    RETURN_TYPES = ("SEGS", )
    RETURN_NAMES = ("sorted_SEGS", )
    OUTPUT_TOOLTIPS = ("The sorted SEGS tuple, ready for further Impact Pack processing.",)
    FUNCTION = "doit"
    CATEGORY = "Monolith"

    def doit(self, segs, depth_map, sort_method, order):
        shape, seg_list = segs
        reverse_sort = order == "descending (nearest first)"
        
        orig_h, orig_w = shape[0], shape[1]
        if depth_map.shape[1] != orig_h or depth_map.shape[2] != orig_w:
            d_map_resized = depth_map.permute(0, 3, 1, 2)
            d_map_resized = F.interpolate(d_map_resized, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            depth_map = d_map_resized.permute(0, 2, 3, 1)

        if len(depth_map.shape) == 4 and depth_map.shape[3] >= 3:
            d_map = depth_map.mean(dim=3, keepdim=True)[0]
        else:
            d_map = depth_map[0]
            
        def compute_depth(seg):
            x1, y1, x2, y2 = seg.crop_region
            cropped_depth = d_map[y1:y2, x1:x2, 0]
            
            mask = seg.cropped_mask
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(cropped_depth.device)
                
            valid_pixels = cropped_depth[mask > 0.5]
            
            if valid_pixels.numel() == 0:
                return 0.0
                
            if sort_method == "median":
                return valid_pixels.median().item()
            else:
                return valid_pixels.mean().item()
        
        sorted_list = list(seg_list)
        sorted_list.sort(key=compute_depth, reverse=reverse_sort)
        
        return ((shape, sorted_list), )
