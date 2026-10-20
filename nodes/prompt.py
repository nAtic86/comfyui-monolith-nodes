import os
import re

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
    def _find_upstream_filename(cls, prompt, start_node_id) -> str:
        if not prompt or start_node_id is None:
            return ""
        start_node = prompt.get(str(start_node_id), {})
        image_link = start_node.get("inputs", {}).get("image")
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
                    return cls._clean_filename(val)
            for val in inputs.values():
                if isinstance(val, list) and len(val) == 2:
                    queue.append(str(val[0]))
        return ""

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

        exact = {}
        canonical = {}
        for header, body in raw_sections:
            lower = header.lower()
            stem, _ = os.path.splitext(lower)
            exact[lower] = body
            exact[stem] = body

            canon_stem, ext, ver = cls._split_canonical(header)
            for key in (canon_stem + ext, canon_stem):
                if not key:
                    continue
                prev = canonical.get(key)
                if prev is None or ver >= prev[0]:
                    canonical[key] = (ver, body)

        return exact, canonical

    def select_prompt(self, image, rules, filename_override="", prompt=None, unique_id=None):
        raw_name = (
            self._clean_filename(filename_override)
            if filename_override and filename_override.strip()
            else self._find_upstream_filename(prompt, unique_id)
        )
        lower = raw_name.lower()
        stem, _ = os.path.splitext(lower)
        canon_stem, ext, ver = self._split_canonical(raw_name)
        canon_full = canon_stem + ext

        exact, canonical = self._parse_rules(rules)

        if ver > 0 and lower in exact:
            selected = exact[lower]
        elif ver > 0 and stem in exact:
            selected = exact[stem]
        elif canon_full and canon_full in canonical:
            selected = canonical[canon_full][1]
        elif canon_stem and canon_stem in canonical:
            selected = canonical[canon_stem][1]
        else:
            selected = exact.get("default", "")

        return (selected, raw_name)


PROMPT_CLASS_MAPPINGS = {
    "MonolithPromptSwitchByImage": MonolithPromptSwitchByImage,
}

PROMPT_NAME_MAPPINGS = {
    "MonolithPromptSwitchByImage": "Monolith Prompt Switch By Image",
}
