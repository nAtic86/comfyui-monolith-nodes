import torch
from torchvision import transforms
from transformers import pipeline


class MonolithSafetyFallback:
    DESCRIPTION = "Runs vit-base-nsfw-detector. If the NSFW score exceeds threshold, outputs fallback_image (e.g. C2A / retry prompt) instead of the input image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The generated image to evaluate."}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "NSFW probability threshold (0.0 to 1.0). If exceeded, fallback_image is returned.",
                }),
                "cuda": ("BOOLEAN", {"default": True, "tooltip": "Run inference on CUDA if available."}),
            },
            "optional": {
                "fallback_image": ("IMAGE", {"tooltip": "Image to output when NSFW is detected (e.g. C2A / retry notice). If unconnected, outputs black tensor."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("IMAGE", "is_nsfw", "score")
    OUTPUT_TOOLTIPS = (
        "Output image: input image if safe, fallback_image if NSFW.",
        "True if NSFW threshold was exceeded.",
        "Raw NSFW confidence score (0.0 to 1.0).",
    )
    FUNCTION = "process_image"
    CATEGORY = "Monolith/Safety"

    def __init__(self):
        self._classifiers = {}

    def _get_classifier(self, device: str):
        if device not in self._classifiers:
            try:
                self._classifiers[device] = pipeline(
                    "image-classification",
                    model="AdamCodd/vit-base-nsfw-detector",
                    device=device,
                    model_kwargs={"local_files_only": True},
                )
            except Exception:
                self._classifiers[device] = pipeline(
                    "image-classification",
                    model="AdamCodd/vit-base-nsfw-detector",
                    device=device,
                )
        return self._classifiers[device]

    def process_image(self, image, threshold, cuda, fallback_image=None):
        device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
        classifier = self._get_classifier(device)

        pil_img = transforms.ToPILImage()(image[0].cpu().permute(2, 0, 1))
        results = classifier(pil_img)
        score = next(item["score"] for item in results if item["label"] == "nsfw")
        is_nsfw = float(score) > threshold

        if is_nsfw:
            output = fallback_image if fallback_image is not None else torch.zeros_like(image)
        else:
            output = image

        return (output, is_nsfw, float(score))


SAFETY_CLASS_MAPPINGS = {
    "MonolithSafetyFallback": MonolithSafetyFallback,
}

SAFETY_NAME_MAPPINGS = {
    "MonolithSafetyFallback": "Monolith Safety Fallback",
}
