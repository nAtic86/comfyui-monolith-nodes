"""
pp_fastfaceswap Model Manager

Downloads required ONNX models from GitHub releases (facefusion-assets).
Supports multiple swap and enhancer models with licensing info.
"""

import os
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional, Dict, List

# Base URL for GitHub releases (primary) and HuggingFace (fallback)
GITHUB_BASE = "https://github.com/facefusion/facefusion-assets/releases/download"
HUGGINGFACE_BASE = "https://huggingface.co/facefusion"

def github_url(version: str, filename: str) -> str:
    """Generate GitHub release URL."""
    return f"{GITHUB_BASE}/{version}/{filename}"

def huggingface_url(version: str, filename: str) -> str:
    """Generate HuggingFace URL (fallback)."""
    return f"{HUGGINGFACE_BASE}/{version}/resolve/main/{filename}"

# Model registry with download URLs and metadata
# License info: 
#   - Ghost: Apache-2.0 (COMMERCIAL SAFE!)
#   - SimSwap: Non-commercial
#   - InsightFace/Inswapper: Non-commercial by default
#   - Hyperswap: ResearchRAIL (FaceFusion 2025)
#   - GFPGAN/GPEN: Apache-2.0 (commercial safe)

MODEL_REGISTRY: Dict[str, Dict] = {
    # === SWAP MODELS: INSWAPPER (inswapper type) ===
    "inswapper_128": {
        "filename": "inswapper_128.onnx",
        "url": github_url("models-3.0.0", "inswapper_128.onnx"),
        "type": "inswapper",
        "size": (128, 128),
        "template": "arcface_128",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "license": "InsightFace (non-commercial)",
        "commercial_safe": False,
    },
    "inswapper_128_fp16": {
        "filename": "inswapper_128_fp16.onnx",
        "url": github_url("models-3.0.0", "inswapper_128_fp16.onnx"),
        "type": "inswapper",
        "size": (128, 128),
        "template": "arcface_128",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "license": "InsightFace (non-commercial)",
        "commercial_safe": False,
    },
    
    # === SWAP MODELS: HYPERSWAP (hyperswap type) - models-3.3.0 ===
    "hyperswap_1a_256": {
        "filename": "hyperswap_1a_256.onnx",
        "url": github_url("models-3.3.0", "hyperswap_1a_256.onnx"),
        "type": "hyperswap",
        "size": (256, 256),
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "license": "ResearchRAIL (FaceFusion 2025)",
        "commercial_safe": False,
    },
    "hyperswap_1b_256": {
        "filename": "hyperswap_1b_256.onnx",
        "url": github_url("models-3.3.0", "hyperswap_1b_256.onnx"),
        "type": "hyperswap",
        "size": (256, 256),
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "license": "ResearchRAIL (FaceFusion 2025)",
        "commercial_safe": False,
    },
    "hyperswap_1c_256": {
        "filename": "hyperswap_1c_256.onnx",
        "url": github_url("models-3.3.0", "hyperswap_1c_256.onnx"),
        "type": "hyperswap",
        "size": (256, 256),
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "license": "ResearchRAIL (FaceFusion 2025)",
        "commercial_safe": False,
    },
    
    # === SWAP MODELS: GHOST (ghost type) - COMMERCIAL SAFE! ===
    "ghost_1_256": {
        "filename": "ghost_1_256.onnx",
        "url": github_url("models-3.0.0", "ghost_1_256.onnx"),
        "type": "ghost",
        "size": (256, 256),
        "template": "arcface_112_v1",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "embedding_converter": "crossface_ghost",
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    "ghost_2_256": {
        "filename": "ghost_2_256.onnx",
        "url": github_url("models-3.0.0", "ghost_2_256.onnx"),
        "type": "ghost",
        "size": (256, 256),
        "template": "arcface_112_v1",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "embedding_converter": "crossface_ghost",
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    "ghost_3_256": {
        "filename": "ghost_3_256.onnx",
        "url": github_url("models-3.0.0", "ghost_3_256.onnx"),
        "type": "ghost",
        "size": (256, 256),
        "template": "arcface_112_v1",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "embedding_converter": "crossface_ghost",
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    
    # === SWAP MODELS: SIMSWAP (simswap type) ===
    "simswap_256": {
        "filename": "simswap_256.onnx",
        "url": github_url("models-3.0.0", "simswap_256.onnx"),
        "type": "simswap",
        "size": (256, 256),
        "template": "arcface_112_v1",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "embedding_converter": "crossface_simswap",
        "license": "Non-commercial",
        "commercial_safe": False,
    },
    "simswap_unofficial_512": {
        "filename": "simswap_unofficial_512.onnx",
        "url": github_url("models-3.0.0", "simswap_unofficial_512.onnx"),
        "type": "simswap",
        "size": (512, 512),
        "template": "arcface_112_v1",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "embedding_converter": "crossface_simswap",
        "license": "Non-commercial",
        "commercial_safe": False,
    },
    
    # === SWAP MODELS: BLENDSWAP ===
    "blendswap_256": {
        "filename": "blendswap_256.onnx",
        "url": github_url("models-3.0.0", "blendswap_256.onnx"),
        "type": "blendswap",
        "size": (256, 256),
        "template": "ffhq_512",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "license": "Non-commercial (mapooon 2023)",
        "commercial_safe": False,
    },
    
    # === SWAP MODELS: UNIFACE ===
    "uniface_256": {
        "filename": "uniface_256.onnx",
        "url": github_url("models-3.0.0", "uniface_256.onnx"),
        "type": "uniface",
        "size": (256, 256),
        "template": "ffhq_512",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "license": "Unknown (xc-csc101 2022)",
        "commercial_safe": False,
    },
    
    # === SWAP MODELS: HIFIFACE - models-3.1.0 ===
    "hififace_unofficial_256": {
        "filename": "hififace_unofficial_256.onnx",
        "url": github_url("models-3.1.0", "hififace_unofficial_256.onnx"),
        "type": "hififace",
        "size": (256, 256),
        "template": "mtcnn_512",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "embedding_converter": "crossface_hififace",
        "license": "Unknown (GuijiAI 2021)",
        "commercial_safe": False,
    },
    
    # === EMBEDDING CONVERTERS - models-3.4.0 ===
    "crossface_ghost": {
        "filename": "crossface_ghost.onnx",
        "url": github_url("models-3.4.0", "crossface_ghost.onnx"),
        "type": "embedding_converter",
        "license": "MIT",
        "commercial_safe": True,
    },
    "crossface_simswap": {
        "filename": "crossface_simswap.onnx",
        "url": github_url("models-3.4.0", "crossface_simswap.onnx"),
        "type": "embedding_converter",
        "license": "MIT",
        "commercial_safe": True,
    },
    "crossface_hififace": {
        "filename": "crossface_hififace.onnx",
        "url": github_url("models-3.4.0", "crossface_hififace.onnx"),
        "type": "embedding_converter",
        "license": "MIT",
        "commercial_safe": True,
    },
    
    # === ENHANCER MODELS ===
    "gfpgan_1.4": {
        "filename": "gfpgan_1.4.onnx",
        "url": github_url("models-3.0.0", "gfpgan_1.4.onnx"),
        "type": "enhancer",
        "template": "ffhq_512",
        "size": (512, 512),
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    "codeformer": {
        "filename": "codeformer.onnx",
        "url": github_url("models-3.0.0", "codeformer.onnx"),
        "type": "enhancer",
        "template": "ffhq_512",
        "size": (512, 512),
        "license": "MIT",
        "commercial_safe": True,
    },
    "restoreformer_plus_plus": {
        "filename": "restoreformer_plus_plus.onnx",
        "url": github_url("models-3.0.0", "restoreformer_plus_plus.onnx"),
        "type": "enhancer",
        "size": (512, 512),
        "license": "MIT",
        "commercial_safe": True,
    },
    "gpen_bfr_256": {
        "filename": "gpen_bfr_256.onnx",
        "url": github_url("models-3.0.0", "gpen_bfr_256.onnx"),
        "type": "enhancer",
        "size": (256, 256),
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    "gpen_bfr_512": {
        "filename": "gpen_bfr_512.onnx",
        "url": github_url("models-3.0.0", "gpen_bfr_512.onnx"),
        "type": "enhancer",
        "size": (512, 512),
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    "gpen_bfr_1024": {
        "filename": "gpen_bfr_1024.onnx",
        "url": github_url("models-3.0.0", "gpen_bfr_1024.onnx"),
        "type": "enhancer",
        "size": (1024, 1024),
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    "gpen_bfr_2048": {
        "filename": "gpen_bfr_2048.onnx",
        "url": github_url("models-3.0.0", "gpen_bfr_2048.onnx"),
        "type": "enhancer",
        "size": (2048, 2048),
        "license": "Apache-2.0",
        "commercial_safe": True,
    },
    
    # === DETECTION/RECOGNITION MODELS (Required) ===
    "yoloface_8n": {
        "filename": "yoloface_8n.onnx",
        "url": github_url("models-3.0.0", "yoloface_8n.onnx"),
        "type": "detector",
        "license": "AGPL-3.0",
        "required": True,
    },
    "2dfan4": {
        "filename": "2dfan4.onnx",
        "url": github_url("models-3.0.0", "2dfan4.onnx"),
        "type": "landmarker",
        "license": "MIT",
        "required": True,
    },
    "arcface_w600k_r50": {
        "filename": "arcface_w600k_r50.onnx",
        "url": github_url("models-3.0.0", "arcface_w600k_r50.onnx"),
        "type": "recognizer",
        "license": "MIT",
        "required": True,
    },
    "bisenet_resnet_34": {
        "filename": "bisenet_resnet_34.onnx",
        "url": github_url("models-3.0.0", "bisenet_resnet_34.onnx"),
        "type": "masker",
        "license": "MIT",
        "required": True,
    },
    "fairface": {
        "filename": "fairface.onnx",
        "url": github_url("models-3.0.0", "fairface.onnx"),
        "type": "classifier",
        "license": "MIT",
        "required": True,
    },
    "fan_68_5": {
        "filename": "fan_68_5.onnx",
        "url": github_url("models-3.0.0", "fan_68_5.onnx"),
        "type": "landmarker",
        "license": "MIT",
        "required": True,
    },
}

# Default models to use (Dune 3.0.1 baseline)
DEFAULT_SWAP_MODEL = "inswapper_128_fp16"
DEFAULT_ENHANCE_MODEL = "gfpgan_1.4"

# Required models that must always be present
REQUIRED_MODELS = [k for k, v in MODEL_REGISTRY.items() if v.get("required", False)]

# Swap model types for validation
SWAP_MODELS = [k for k, v in MODEL_REGISTRY.items() 
               if v.get("type") in ["inswapper", "hyperswap", "ghost", "simswap", "blendswap", "uniface", "hififace"]]

ENHANCE_MODELS = [k for k, v in MODEL_REGISTRY.items() if v.get("type") == "enhancer"]


import folder_paths

def get_models_dir() -> Path:
    """Returns the primary models directory path."""
    for dirname in ["facefusion", "fastfacefusion"]:
        d = Path(folder_paths.models_dir) / dirname
        if d.exists():
            return d
    d = Path(folder_paths.models_dir) / "facefusion"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_exists(model_name: str) -> bool:
    """Check if a model file exists."""
    p = get_model_path(model_name)
    return p is not None and p.exists()


def download_model(model_name: str, force: bool = False) -> bool:
    """
    Download a model from GitHub releases if not present.
    
    Args:
        model_name: Key from MODEL_REGISTRY
        force: Re-download even if exists
        
    Returns:
        True if model is available (downloaded or already existed)
    """
    if model_name not in MODEL_REGISTRY:
        print(f"[Models] Unknown model: {model_name}", flush=True)
        return False
    
    model_info = MODEL_REGISTRY[model_name]
    model_path = get_models_dir() / model_info["filename"]
    
    if model_path.exists() and not force:
        return True
    
    # Ensure directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    url = model_info["url"]
    print(f"[Models] Downloading {model_name} from {url}...", flush=True)
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                mb = downloaded // 1024 // 1024
                print(f"\r[Models] {model_name}: {percent}% ({mb}MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, model_path, reporthook=report_progress)
        print(flush=True)  # Newline after progress
        print(f"[Models] Downloaded: {model_path}", flush=True)
        return True
        
    except Exception as e:
        print(f"\n[Models] Failed to download {model_name}: {e}", flush=True)
        if model_path.exists():
            model_path.unlink()  # Remove partial download
        return False


def ensure_required_models() -> bool:
    """
    Ensure all required models are present.
    Downloads any missing models.
    
    Returns:
        True if all required models are available
    """
    all_ok = True
    for model_name in REQUIRED_MODELS:
        if not model_exists(model_name):
            print(f"[Models] Missing required model: {model_name}", flush=True)
            if not download_model(model_name):
                all_ok = False
    return all_ok


def ensure_swap_model(model_name: str = DEFAULT_SWAP_MODEL) -> bool:
    """Ensure a swap model and its dependencies are available."""
    if model_name not in MODEL_REGISTRY:
        return False
    
    # Download the main model
    if not model_exists(model_name):
        if not download_model(model_name):
            return False
    
    # Check for embedding converter dependency
    model_info = MODEL_REGISTRY[model_name]
    converter_name = model_info.get("embedding_converter")
    if converter_name:
        if not model_exists(converter_name):
            if not download_model(converter_name):
                return False
    
    return True


def ensure_enhance_model(model_name: str = DEFAULT_ENHANCE_MODEL) -> bool:
    """Ensure an enhancer model is available."""
    if not model_exists(model_name):
        return download_model(model_name)
    return True


def get_model_path(model_name: str) -> Optional[Path]:
    """Get the path to a model file, searching both facefusion and fastfacefusion directories."""
    if model_name not in MODEL_REGISTRY:
        return None
    fname = MODEL_REGISTRY[model_name]["filename"]
    for dirname in ["facefusion", "fastfacefusion"]:
        p = Path(folder_paths.models_dir) / dirname / fname
        if p.exists():
            return p
    return get_models_dir() / fname


def get_model_info(model_name: str) -> Optional[Dict]:
    """Get model metadata."""
    return MODEL_REGISTRY.get(model_name)


def list_available_swap_models() -> List[str]:
    """List available swap models."""
    return SWAP_MODELS


def list_available_enhance_models() -> List[str]:
    """List available enhancer models."""
    return ENHANCE_MODELS


def list_commercial_safe_models() -> List[str]:
    """List models safe for commercial use."""
    return [k for k, v in MODEL_REGISTRY.items() if v.get("commercial_safe", False)]


def print_model_status():
    """Print status of all models."""
    print("\n=== Model Status ===", flush=True)
    print(f"Models directory: {get_models_dir()}", flush=True)
    print(flush=True)
    
    # Group by type
    types_order = ["inswapper", "hyperswap", "ghost", "simswap", "blendswap", "uniface", "hififace", 
                   "embedding_converter", "enhancer", "detector", "landmarker", "recognizer", "masker", "classifier"]
    
    for model_type in types_order:
        models = [(k, v) for k, v in MODEL_REGISTRY.items() if v.get("type") == model_type]
        if models:
            print(f"\n  --- {model_type.upper()} ---", flush=True)
            for model_name, info in models:
                exists = "OK" if model_exists(model_name) else "MISSING"
                commercial = "Commercial OK" if info.get("commercial_safe") else "Non-commercial"
                required = " (REQUIRED)" if info.get("required") else ""
                print(f"  [{exists}] {model_name}: {info.get('license', 'Unknown')} - {commercial}{required}", flush=True)
    
    print(flush=True)


if __name__ == "__main__":
    # Test: Print status and ensure required models
    print_model_status()
    print("\nEnsuring required models...", flush=True)
    if ensure_required_models():
        print("\nAll required models available", flush=True)
    else:
        print("\nSome models failed to download", flush=True)
