import os
import re
import torch
import folder_paths
import nodes
from comfy_api.latest import io

DEFAULT_RULES = """[image.png]
image.png prompt

[image2.png]
image2.png prompt

[default]
fallback prompt if filename is not listed"""


class MonolithPromptSwitchByImage:
    DESCRIPTION = "Outputs a prompt matched to the upstream image's filename using [filename.png] sections."
    _DUP_RE = re.compile(r"^(.*?)\s*\((\d+)\)$")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Connect an image (e.g. from LoadImage). The upstream filename is detected automatically."}),
                "rules": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_RULES,
                    "tooltip": "INI-style mapping of [filename.png] or [stem] headers to prompts. Automatically handles ComfyUI ' (1)' duplicate suffixes and [default] fallback.",
                }),
            },
            "optional": {
                "filename_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional explicit filename string. Leave empty to auto-detect from the connected image wire.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "detected_filename")
    OUTPUT_TOOLTIPS = (
        "The prompt corresponding to the matched [filename] section (or [default]).",
        "The resolved basename of the input image (useful for debugging).",
    )
    FUNCTION = "select_prompt"
    CATEGORY = "Monolith/Prompt"

    @staticmethod
    def _clean_filename(raw: str) -> str:
        cleaned = re.sub(r"\s+\[(input|output|temp)\]$", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = cleaned.replace("\\", "/")
        return os.path.basename(cleaned)

    @classmethod
    def _split_canonical(cls, name: str) -> tuple[str, str, int]:
        lower = cls._clean_filename(name).lower()
        stem, ext = os.path.splitext(lower)
        match = cls._DUP_RE.match(stem)
        if match and match.group(1):
            return match.group(1).strip(), ext, int(match.group(2))
        return stem.strip(), ext, 0

    @classmethod
    def _find_upstream_raw_file(cls, prompt, start_node_id, input_key: str = "image") -> str:
        if not prompt or start_node_id is None:
            return ""
        start_node = prompt.get(str(start_node_id), {})
        image_link = start_node.get("inputs", {}).get(input_key)
        if not (isinstance(image_link, list) and len(image_link) == 2):
            return ""

        visited = set()
        queue = [str(image_link[0])]
        while queue:
            node_id = queue.pop(0)
            if node_id in visited or node_id not in prompt:
                continue
            visited.add(node_id)
            inputs = prompt[node_id].get("inputs", {})
            for key in ("image", "filename", "file", "image_path"):
                val = inputs.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            for val in inputs.values():
                if isinstance(val, list) and len(val) == 2:
                    queue.append(str(val[0]))
        return ""

    @classmethod
    def _find_upstream_filename(cls, prompt, start_node_id, input_key: str = "image") -> str:
        raw = cls._find_upstream_raw_file(prompt, start_node_id, input_key)
        return cls._clean_filename(raw) if raw else ""

    @classmethod
    def _build_lookup(cls, entries: list[tuple[str, object]]):
        exact = {}
        canonical = {}
        for header, value in entries:
            lower = header.lower()
            stem, _ = os.path.splitext(lower)
            exact[lower] = value
            exact[stem] = value

            canon_stem, ext, ver = cls._split_canonical(header)
            for key in (canon_stem + ext, canon_stem):
                if not key:
                    continue
                prev = canonical.get(key)
                if prev is None or ver >= prev[0]:
                    canonical[key] = (ver, value)

        return exact, canonical

    @classmethod
    def _resolve_lookup(cls, raw_name: str, exact: dict, canonical: dict, default=None):
        lower = raw_name.lower()
        stem, _ = os.path.splitext(lower)
        canon_stem, ext, ver = cls._split_canonical(raw_name)
        canon_full = canon_stem + ext

        if ver > 0 and lower in exact:
            return exact[lower], True
        if ver > 0 and stem in exact:
            return exact[stem], True
        if canon_full and canon_full in canonical:
            return canonical[canon_full][1], True
        if canon_stem and canon_stem in canonical:
            return canonical[canon_stem][1], True
        return default, False

    @classmethod
    def _parse_rules(cls, rules_text: str):
        raw_sections = []
        current_key = None
        buffer = []
        for line in rules_text.splitlines():
            match = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
            if match:
                if current_key is not None:
                    raw_sections.append((current_key, "\n".join(buffer).strip()))
                current_key = cls._clean_filename(match.group(1))
                buffer = []
            elif current_key is not None:
                buffer.append(line)
        if current_key is not None:
            raw_sections.append((current_key, "\n".join(buffer).strip()))

        return cls._build_lookup(raw_sections)

    def select_prompt(self, image, rules, filename_override="", prompt=None, unique_id=None):
        raw_name = (
            self._clean_filename(filename_override)
            if filename_override and filename_override.strip()
            else self._find_upstream_filename(prompt, unique_id)
        )
        exact, canonical = self._parse_rules(rules)
        selected, _ = self._resolve_lookup(raw_name, exact, canonical, default=exact.get("default", ""))
        return (selected, raw_name)


class MonolithImageSwitchByImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        slot_indices = range(1, 51)
        match_template = io.Autogrow.TemplateNames(
            input=io.Image.Input(
                "image",
                tooltip="Candidate image to match against the input image (e.g. from LoadImage).",
            ),
            names=[f"image{i}" for i in slot_indices],
            min=0,
        )
        output_template = io.Autogrow.TemplateNames(
            input=io.Image.Input(
                "image_output",
                tooltip="Output image returned when the corresponding imageN matches (e.g. overlay LoadImage).",
            ),
            names=[f"image{i}_output" for i in slot_indices],
            min=0,
        )
        return io.Schema(
            node_id="MonolithImageSwitchByImage",
            display_name="Monolith Image Switch By Image",
            category="Monolith/Prompt",
            description="Patch-bay image router: matches the input image against connected image1..N slots and outputs the corresponding image1_output..N_output.",
            search_aliases=["Image Switch By Image", "patch bay", "overlay switch"],
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Connect the active image (e.g. costume from LoadImage). Its upstream filename is detected automatically.",
                ),
                io.Autogrow.Input(
                    "match_images",
                    template=match_template,
                    tooltip="Connect candidate images (image1, image2, ...) to compare against the input image.",
                ),
                io.Autogrow.Input(
                    "output_images",
                    template=output_template,
                    tooltip="Connect the output image (image1_output, image2_output, ...) for each corresponding candidate image.",
                ),
                io.Image.Input(
                    "default_image",
                    optional=True,
                    tooltip="Fallback image returned when no imageN slot matches. If unconnected, falls back to the input image.",
                ),
                io.String.Input(
                    "filename_override",
                    default="",
                    optional=True,
                    tooltip="Optional explicit filename string for the input image. Leave empty to auto-detect from the connected image wire.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGE",
                    tooltip="The matched imageN_output (or default_image / input image if no match).",
                ),
                io.Mask.Output(
                    display_name="MASK",
                    tooltip="Alpha mask extracted from the selected output image's upstream LoadImage (matching LoadImage's MASK output).",
                ),
                io.String.Output(
                    display_name="detected_filename",
                    tooltip="The resolved basename of the input image (useful for debugging).",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def _resolve_mask(cls, prompt, unique_id, input_keys: tuple[str, ...], out_img: torch.Tensor) -> torch.Tensor:
        for key in input_keys:
            raw_file = MonolithPromptSwitchByImage._find_upstream_raw_file(prompt, unique_id, key)
            if raw_file and folder_paths.exists_annotated_filepath(raw_file):
                _, mask = nodes.LoadImage().load_image(raw_file)
                if (
                    isinstance(out_img, torch.Tensor)
                    and out_img.ndim == 4
                    and mask.shape[-2:] != (64, 64)
                    and mask.shape[-2:] != (out_img.shape[1], out_img.shape[2])
                ):
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(1),
                        size=(out_img.shape[1], out_img.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                return mask

        if isinstance(out_img, torch.Tensor) and out_img.ndim == 4:
            if out_img.shape[-1] == 4:
                return 1.0 - out_img[..., 3]
            return torch.zeros((out_img.shape[0], 64, 64), dtype=out_img.dtype, device=out_img.device)
        return torch.zeros((1, 64, 64), dtype=torch.float32)

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        match_images: io.Autogrow.Type = None,
        output_images: io.Autogrow.Type = None,
        default_image: torch.Tensor | None = None,
        filename_override: str = "",
    ) -> io.NodeOutput:
        hidden = getattr(cls, "hidden", None)
        prompt = hidden.prompt if hidden is not None else None
        unique_id = hidden.unique_id if hidden is not None else None

        raw_name = (
            MonolithPromptSwitchByImage._clean_filename(filename_override)
            if filename_override and filename_override.strip()
            else MonolithPromptSwitchByImage._find_upstream_filename(prompt, unique_id, "image")
        )

        match_images = match_images or {}
        output_images = output_images or {}

        named_entries = []
        tensor_entries = []
        for slot_name, cand_img in match_images.items():
            out_key = f"{slot_name}_output"
            out_img = output_images.get(out_key)
            if out_img is None:
                continue

            out_keys = (f"output_images.{out_key}", out_key)
            cand_name = (
                MonolithPromptSwitchByImage._find_upstream_filename(prompt, unique_id, f"match_images.{slot_name}")
                or MonolithPromptSwitchByImage._find_upstream_filename(prompt, unique_id, slot_name)
            )
            if cand_name:
                named_entries.append((cand_name, (out_img, out_keys)))
            tensor_entries.append((cand_img, (out_img, out_keys)))

        if raw_name and named_entries:
            exact, canonical = MonolithPromptSwitchByImage._build_lookup(named_entries)
            selected, matched = MonolithPromptSwitchByImage._resolve_lookup(raw_name, exact, canonical)
            if matched:
                sel_img, sel_keys = selected
                return io.NodeOutput(sel_img, cls._resolve_mask(prompt, unique_id, sel_keys, sel_img), raw_name)
        elif not raw_name:
            for cand_img, (out_img, out_keys) in tensor_entries:
                if (
                    isinstance(image, torch.Tensor)
                    and isinstance(cand_img, torch.Tensor)
                    and image.shape == cand_img.shape
                    and torch.equal(image, cand_img)
                ):
                    return io.NodeOutput(out_img, cls._resolve_mask(prompt, unique_id, out_keys, out_img), raw_name)

        if default_image is not None:
            return io.NodeOutput(
                default_image,
                cls._resolve_mask(prompt, unique_id, ("default_image",), default_image),
                raw_name,
            )
        return io.NodeOutput(image, cls._resolve_mask(prompt, unique_id, ("image",), image), raw_name)


PROMPT_CLASS_MAPPINGS = {
    "MonolithPromptSwitchByImage": MonolithPromptSwitchByImage,
    "MonolithImageSwitchByImage": MonolithImageSwitchByImage,
}

PROMPT_NAME_MAPPINGS = {
    "MonolithPromptSwitchByImage": "Monolith Prompt Switch By Image",
    "MonolithImageSwitchByImage": "Monolith Image Switch By Image",
}
