"""
pp_fastfaceswap Engine

High-performance face swap engine with multi-model support and configurable parameters.

CRITICAL: Library path setup must happen BEFORE importing onnxruntime.
- Linux: Set LD_LIBRARY_PATH to include NVIDIA pip package libraries
- Windows: Use bundled CUDA DLLs in ./cuda_libs folder
"""
import os
import sys
import signal
import atexit

# =============================================================================
# CUDA/NVIDIA LIBRARY PATH FIX - MUST BE AT THE ABSOLUTE TOP BEFORE ANY ONNX IMPORTS
# =============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _setup_nvidia_library_paths():
    """Auto-detect and add NVIDIA pip package libraries to LD_LIBRARY_PATH and preload via ctypes (Linux)."""
    if sys.platform != 'linux':
        return
    
    import ctypes
    import site
    search_paths = site.getsitepackages() + [site.getusersitepackages()]
    
    nvidia_lib_dirs = []
    nvidia_packages = ['cu13', 'cudnn', 'cuda_runtime', 'cublas', 'cufft', 'curand', 'cusparse', 'cusolver', 'nvjitlink']
    
    for sp in search_paths:
        nvidia_base = os.path.join(sp, 'nvidia')
        if os.path.isdir(nvidia_base):
            for pkg in nvidia_packages:
                lib_dir = os.path.join(nvidia_base, pkg, 'lib')
                if os.path.isdir(lib_dir):
                    nvidia_lib_dirs.append(lib_dir)
    
    wsl_cuda_path = '/usr/lib/wsl/lib'
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower() and os.path.isdir(wsl_cuda_path):
                nvidia_lib_dirs.append(wsl_cuda_path)
    
    if nvidia_lib_dirs:
        existing = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = ':'.join(nvidia_lib_dirs)
        os.environ['LD_LIBRARY_PATH'] = new_paths + ':' + existing if existing else new_paths
        for lib_dir in nvidia_lib_dirs:
            for so_name in ('libcudart.so.13', 'libcublasLt.so.13', 'libcublas.so.13', 'libcufft.so.12', 'libcurand.so.10', 'libcudnn.so.9'):
                so_path = os.path.join(lib_dir, so_name)
                if os.path.exists(so_path):
                    try:
                        ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

_setup_nvidia_library_paths()

if sys.platform == 'win32':
    _cuda_libs = os.path.join(_SCRIPT_DIR, 'cuda_libs')
    if os.path.exists(_cuda_libs):
        os.add_dll_directory(_cuda_libs)
        os.environ['PATH'] = _cuda_libs + os.pathsep + os.environ.get('PATH', '')
        print(f"[CUDA-DLL] Using bundled libs: {_cuda_libs}", flush=True)
    
    _cuda_path = os.environ.get("CUDA_PATH", "")
    if _cuda_path:
        _cuda_bin = os.path.join(_cuda_path, "bin")
        if os.path.exists(_cuda_bin):
            os.add_dll_directory(_cuda_bin)

# =============================================================================
# NOW safe to import onnxruntime
# =============================================================================

import cv2
import numpy
import onnx
import onnxruntime
from onnx import numpy_helper
import subprocess
import time
import pickle
from typing import Optional, Tuple, List, Dict, Any

# =============================================================================
# LOGGING HELPER
# =============================================================================
def log(msg: str, level: str = "INFO"):
    """Simple timestamped logging."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

# =============================================================================
# PROCESS CLEANUP REGISTRY
# =============================================================================
_active_processes: List[subprocess.Popen] = []
_active_captures: List[cv2.VideoCapture] = []

def _cleanup_all():
    """Cleanup all active processes and resources."""
    for proc in _active_processes:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
        except:
            try:
                proc.kill()
            except:
                pass
    _active_processes.clear()
    
    for cap in _active_captures:
        try:
            cap.release()
        except:
            pass
    _active_captures.clear()

atexit.register(_cleanup_all)

# Import model manager
try:
    from .pp_fastfaceswap_models import (
        MODEL_REGISTRY, 
        get_model_path, 
        ensure_required_models,
        ensure_swap_model,
        ensure_enhance_model,
        DEFAULT_SWAP_MODEL,
        DEFAULT_ENHANCE_MODEL,
        get_model_info,
    )
except (ImportError, ValueError):
    from pp_fastfaceswap_models import (
        MODEL_REGISTRY, 
        get_model_path, 
        ensure_required_models,
        ensure_swap_model,
        ensure_enhance_model,
        DEFAULT_SWAP_MODEL,
        DEFAULT_ENHANCE_MODEL,
        get_model_info,
    )

# ---------------------------------------------------------------------
# CONFIGURATION DEFAULTS
# ---------------------------------------------------------------------
FACE_DETECTOR_SIZE = '640x640'
FACE_DETECTOR_SCORE = 0.5
FACE_LANDMARKER_SCORE = 0.5
# For NVIDIA GPU: Use short names matching facefusion's choices.py
# NOTE: TensorRT requires native libs (libnvinfer.so) which pip tensorrt doesn't include
# Using CUDA only, which is fully supported by nvidia-cudnn-cu12 + nvidia-cublas-cu12
EXECUTION_PROVIDERS = ['cuda']

# Mask defaults (matching facefusion)
DEFAULT_MASK_BLUR = 0.3        # 0.0-1.0, edge softness
DEFAULT_MASK_PADDING = [0, 0, 0, 0]  # [top, right, bottom, left] as percentage


# ---------------------------------------------------------------------
# MASK FUNCTIONS (matching facefusion/face_masker.py)
# ---------------------------------------------------------------------
def create_box_mask(crop_size: Tuple[int, int], blur: float = 0.3, 
                    padding: List[int] = None) -> numpy.ndarray:
    """
    Create a box mask with blur and padding (facefusion-compatible).
    
    Args:
        crop_size: (width, height) of the crop
        blur: 0.0-1.0, amount of edge blur
        padding: [top, right, bottom, left] as percentage (0-100)
        
    Returns:
        Mask array of shape (height, width) with float32 values 0-1
    """
    if padding is None:
        padding = DEFAULT_MASK_PADDING
    
    width, height = crop_size
    blur_amount = int(width * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)
    
    # Start with full mask
    box_mask = numpy.ones((height, width), dtype=numpy.float32)
    
    # Apply padding (set edges to 0)
    # Top
    top_pad = max(blur_area, int(height * padding[0] / 100))
    if top_pad > 0:
        box_mask[:top_pad, :] = 0
    
    # Bottom  
    bottom_pad = max(blur_area, int(height * padding[2] / 100))
    if bottom_pad > 0:
        box_mask[-bottom_pad:, :] = 0
    
    # Left
    left_pad = max(blur_area, int(width * padding[3] / 100))
    if left_pad > 0:
        box_mask[:, :left_pad] = 0
    
    # Right
    right_pad = max(blur_area, int(width * padding[1] / 100))
    if right_pad > 0:
        box_mask[:, -right_pad:] = 0
    
    # Apply Gaussian blur to soften edges
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    
    return box_mask


def check_nvenc_available() -> bool:
    """Check if NVENC GPU encoding is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False


class FastFaceSwapEngine:
    """
    High-performance face swap engine with multi-model support.
    
    Features:
    - Multiple swap models (simswap, inswapper, hyperswap)
    - Configurable enhancement
    - Face detection caching
    - GPU-accelerated encoding
    - Thumbnail generation
    """
    
    def __init__(
        self,
        swap_model: str = DEFAULT_SWAP_MODEL,
        enhance_model: str = DEFAULT_ENHANCE_MODEL,
        execution_providers: List[str] = None,
        # Mask settings (facefusion-compatible)
        mask_blur: float = DEFAULT_MASK_BLUR,
        mask_padding: List[int] = None,
        pixel_boost: str = "128x128",
        face_swapper_weight: float = 0.5,
        face_selector_mode: str = "reference",
        face_selector_order: str = "large-small",
        reference_face_distance: float = 0.6,
    ):
        self.script_dir = _SCRIPT_DIR
        self.swap_model_name = swap_model
        self.enhance_model_name = enhance_model
        self.execution_providers = execution_providers or EXECUTION_PROVIDERS
        
        # Mask settings
        self.mask_blur = mask_blur
        self.mask_padding = mask_padding or DEFAULT_MASK_PADDING.copy()
        
        self.pixel_boost = pixel_boost
        self.face_swapper_weight = face_swapper_weight
        self.face_selector_mode = face_selector_mode
        self.face_selector_order = face_selector_order
        self.reference_face_distance = reference_face_distance
        
        self.initialized = False
        self.session = None
        self.enhancer_session = None
        self.embedding_converter_session = None  # For ghost/simswap models
        self.model_initializer = None  # For inswapper models
        self.model_info = None
        self.enhancer_info = None
        self.model_type = None  # inswapper, hyperswap, ghost, simswap
        
        # GPU encoding availability
        self._nvenc_available = None
        
        # Add local facefusion module to path
        sys.path.insert(0, self.script_dir)
        
        # Initialize facefusion state
        self._init_facefusion_state()
        
        # Import helpers
        from facefusion.face_analyser import get_one_face, get_many_faces, get_average_face
        from facefusion.face_selector import sort_faces_by_order, find_match_faces
        from facefusion.face_helper import warp_face_by_face_landmark_5, paste_back
        from facefusion.execution import create_inference_providers
        from facefusion.pixel_boost import implode_pixel_boost, explode_pixel_boost
        from facefusion.vision import unpack_resolution, read_static_image
        
        self.get_one_face = get_one_face
        self.get_many_faces = get_many_faces
        self.get_average_face = get_average_face
        self.sort_faces_by_order = sort_faces_by_order
        self.find_match_faces = find_match_faces
        self.warp_face = warp_face_by_face_landmark_5
        self.paste_back = paste_back
        self.read_image = read_static_image
        self.create_providers = create_inference_providers
        self.implode_pixel_boost = implode_pixel_boost
        self.explode_pixel_boost = explode_pixel_boost
        self.unpack_resolution = unpack_resolution
    
    def _init_facefusion_state(self):
        """Initialize FaceFusion state manager with required config (Dune 3.0.1 defaults)."""
        from facefusion import state_manager
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_size', FACE_DETECTOR_SIZE)
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_detector_score', FACE_DETECTOR_SCORE)
        state_manager.init_item('face_detector_margin', [0, 0, 0, 0])
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', FACE_LANDMARKER_SCORE)
        state_manager.init_item('face_selector_mode', self.face_selector_mode)
        state_manager.init_item('face_selector_order', self.face_selector_order)
        state_manager.init_item('reference_face_position', 0)
        state_manager.init_item('reference_face_distance', self.reference_face_distance)
        state_manager.init_item('reference_frame_number', 0)
        state_manager.init_item('download_providers', ['huggingface', 'github'])
        state_manager.init_item('execution_providers', self.execution_providers)
        state_manager.init_item('execution_thread_count', 6)
        state_manager.init_item('execution_queue_count', 1)
        state_manager.init_item('execution_device_ids', [0])
        state_manager.init_item('face_swapper_model', self.swap_model_name)
        self.state_manager = state_manager
        
        # Monkey patch filesystem to redirect .assets/models to comfyui/models
        from facefusion import filesystem
        original_resolve = filesystem.resolve_relative_path
        def patched_resolve(path):
            if '.assets/models' in path:
                basename = os.path.basename(path)
                import folder_paths
                for dirname in ['facefusion', 'fastfacefusion']:
                    candidate = os.path.join(folder_paths.models_dir, dirname, basename)
                    if os.path.exists(candidate):
                        return candidate
                return os.path.join(folder_paths.models_dir, 'facefusion', basename)
            return original_resolve(path)
        filesystem.resolve_relative_path = patched_resolve
    
    @property
    def nvenc_available(self) -> bool:
        """Check if NVENC is available (cached)."""
        if self._nvenc_available is None:
            self._nvenc_available = check_nvenc_available()
        return self._nvenc_available
    
    def pre_check(self) -> bool:
        """
        Load models into VRAM. Returns True if successful.
        Downloads missing models automatically.
        """
        try:
            log("Loading models...")
            log(f"  Swap model: {self.swap_model_name}")
            log(f"  Enhance model: {self.enhance_model_name}")
            
            # Ensure required models are downloaded
            if not ensure_required_models():
                log("Failed to download required models", "ERROR")
                return False
            
            # Ensure swap model (and embedding converter if needed)
            if not ensure_swap_model(self.swap_model_name):
                log(f"Failed to download swap model: {self.swap_model_name}", "ERROR")
                return False
            
            # Get model info from registry
            self.model_info = get_model_info(self.swap_model_name)
            if not self.model_info:
                log(f"Unknown swap model: {self.swap_model_name}", "ERROR")
                return False
            
            self.model_type = self.model_info.get("type", "inswapper")
            log(f"  Model type: {self.model_type}")
            
            from facefusion import inference_manager, logger, translator
            from onnxruntime import InferenceSession
            def patched_create(model_path, execution_device_id, execution_providers):
                from facefusion.execution import create_inference_providers
                try:
                    inference_providers = create_inference_providers(execution_device_id, execution_providers)
                    return InferenceSession(model_path, providers=inference_providers)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
            inference_manager.create_inference_session = patched_create

            from facefusion.face_detector import get_inference_pool as get_detector_pool
            from facefusion.face_landmarker import get_inference_pool as get_landmarker_pool
            
            pool = get_detector_pool()
            log(f"Detector pool keys: {list(pool.keys())}")
            log("Loading face landmarker...")
            pool = get_landmarker_pool()
            log(f"Landmarker pool keys: {list(pool.keys())}")
            
            # Load swap model
            swap_model_path = get_model_path(self.swap_model_name)
            if not swap_model_path or not swap_model_path.exists():
                log(f"Swap model not found: {self.swap_model_name}", "ERROR")
                return False
            
            log(f"Loading swap model: {swap_model_path.name}...")
            providers = self.create_providers(0, self.execution_providers)
            self.session = onnxruntime.InferenceSession(str(swap_model_path), providers=providers)
            log(f"Swap model loaded. Providers: {self.session.get_providers()}")
            
            # Load model initializer for inswapper type (embedding normalization matrix)
            if self.model_type == "inswapper":
                model_proto = onnx.load(str(swap_model_path))
                self.model_initializer = numpy_helper.to_array(model_proto.graph.initializer[-1])
                log("Inswapper initializer loaded")
            
            # Load embedding converter for ghost/simswap types
            converter_name = self.model_info.get("embedding_converter")
            if converter_name:
                converter_path = get_model_path(converter_name)
                if converter_path and converter_path.exists():
                    log(f"Loading embedding converter: {converter_path.name}...")
                    self.embedding_converter_session = onnxruntime.InferenceSession(
                        str(converter_path), providers=providers
                    )
                    log(f"Embedding converter loaded: {converter_name}")
                else:
                    log(f"Embedding converter not found: {converter_name}", "ERROR")
                    return False
            
            # Load enhancer (optional)
            if self.enhance_model_name:
                if ensure_enhance_model(self.enhance_model_name):
                    enhance_path = get_model_path(self.enhance_model_name)
                    if enhance_path and enhance_path.exists():
                        log(f"Loading enhancer: {enhance_path.name}...")
                        self.enhancer_session = onnxruntime.InferenceSession(
                            str(enhance_path), providers=providers
                        )
                        self.enhancer_info = get_model_info(self.enhance_model_name)
                        log(f"Enhancer loaded: {self.enhance_model_name}")
                    else:
                        log("Enhancer not found, disabled", "WARN")
                        self.enhancer_session = None
            
            log("All models loaded successfully")
            self.initialized = True
            return True
            
        except Exception as e:
            log(f"Initialization failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
    
    def set_enhancer(self, enhance_model_name: Optional[str]):
        """Dynamically set or change the enhancer model."""
        if not enhance_model_name or enhance_model_name == "none":
            self.enhance_model_name = None
            self.enhancer_session = None
            self.enhancer_info = None
            return
        
        if self.enhance_model_name == enhance_model_name and self.enhancer_session is not None:
            return
            
        self.enhance_model_name = enhance_model_name
        providers = self.create_providers(0, self.execution_providers)
        if ensure_enhance_model(self.enhance_model_name):
            enhance_path = get_model_path(self.enhance_model_name)
            if enhance_path and enhance_path.exists():
                log(f"Loading enhancer: {enhance_path.name}...")
                self.enhancer_session = onnxruntime.InferenceSession(
                    str(enhance_path), providers=providers
                )
                self.enhancer_info = get_model_info(self.enhance_model_name)
                log(f"Enhancer loaded: {self.enhance_model_name}")

    def warmup(self):
        """Ensure models are loaded."""
        if not self.initialized:
            self.pre_check()
    
    def _convert_embedding(self, embedding):
        """Convert embedding using crossface converter (for ghost/simswap)."""
        if self.embedding_converter_session is None:
            return embedding, embedding / numpy.linalg.norm(embedding)
        
        embedding = embedding.reshape(-1, 512)
        converted = self.embedding_converter_session.run(None, {'input': embedding})[0]
        converted = converted.ravel()
        converted_norm = converted / numpy.linalg.norm(converted)
        return converted, converted_norm
    
    def _prepare_source_embedding(self, source_face):
        """Prepare source face embedding for swap based on model type (FaceFusion 3.0.1 reference)."""
        if self.model_type == "inswapper":
            source_embedding = source_face.embedding.reshape((1, -1))
            source_embedding = numpy.dot(source_embedding, self.model_initializer) / numpy.linalg.norm(source_embedding)
            return source_embedding
        
        elif self.model_type == "hyperswap":
            if hasattr(source_face, 'embedding_norm'):
                return source_face.embedding_norm.reshape((1, -1))
            embedding = source_face.embedding.reshape((1, -1))
            return embedding / numpy.linalg.norm(embedding)
        
        elif self.model_type == "ghost":
            source_embedding = source_face.embedding.reshape(-1, 512)
            converted, _ = self._convert_embedding(source_embedding)
            return converted.reshape(1, -1)
        
        elif self.model_type == "simswap":
            source_embedding = source_face.embedding.reshape(-1, 512)
            _, converted_norm = self._convert_embedding(source_embedding)
            return converted_norm.reshape(1, -1)
        
        else:
            source_embedding = source_face.embedding.reshape((1, -1))
            return source_embedding / numpy.linalg.norm(source_embedding)

    def select_target_faces(self, faces: List[Any], reference_face: Optional[Any] = None) -> List[Any]:
        """Sort and filter target faces according to FaceFusion 3.0.1 face_selector rules."""
        if not faces:
            return []
        ordered = self.sort_faces_by_order(faces, self.face_selector_order) if self.face_selector_order else faces
        if self.face_selector_mode == "one":
            one = self.get_one_face(ordered, 0)
            return [one] if one else []
        if self.face_selector_mode == "reference":
            ref = reference_face if reference_face is not None else self.get_one_face(ordered, 0)
            if ref is not None:
                return self.find_match_faces([ref], ordered, self.reference_face_distance)
            return []
        return ordered
    
    def _prepare_crop_frame(self, crop_vision_frame):
        """Prepare cropped frame for model input."""
        mean = self.model_info.get("mean", [0.0, 0.0, 0.0])
        std = self.model_info.get("std", [1.0, 1.0, 1.0])
        
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - mean) / std
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(numpy.float32)
        return crop_vision_frame
    
    def _normalize_crop_frame(self, crop_vision_frame):
        """Normalize model output back to image based on model type."""
        mean = self.model_info.get("mean", [0.0, 0.0, 0.0])
        std = self.model_info.get("std", [1.0, 1.0, 1.0])
        
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        
        if self.model_type in ["ghost", "hyperswap", "hififace", "uniface"]:
            crop_vision_frame = crop_vision_frame * std + mean
        
        crop_vision_frame = crop_vision_frame.clip(0, 1)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
        return crop_vision_frame
    
    def _forward_enhancer(self, crop_frame):
        """Run enhancer inference on a 512x512 crop matching FaceFusion 3.0.1 face_enhancer.py."""
        enhance_size = self.enhancer_info.get("size", (512, 512))
        input_frame = cv2.resize(crop_frame, enhance_size) if crop_frame.shape[:2] != (enhance_size[1], enhance_size[0]) else crop_frame
        input_frame = input_frame[:, :, ::-1] / 255.0
        input_frame = (input_frame - 0.5) / 0.5
        input_frame = numpy.expand_dims(input_frame.transpose(2, 0, 1), axis=0).astype(numpy.float32)
        
        inputs = {self.enhancer_session.get_inputs()[0].name: input_frame}
        for inp in self.enhancer_session.get_inputs():
            if inp.name == 'weight':
                inputs['weight'] = numpy.array([1.0]).astype(numpy.double)
                
        output = self.enhancer_session.run(None, inputs)[0][0]
        
        output = numpy.clip(output, -1, 1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = (output * 255.0).round()
        output = output.astype(numpy.uint8)[:, :, ::-1]
        return output

    def enhance_face(self, crop_frame, blend: int = 80):
        """
        Apply face enhancement directly on a cropped frame.
        Maintains full enhancer resolution (no downsampling to model_size).
        """
        if not self.enhancer_session or blend <= 0:
            return crop_frame
        
        original = crop_frame.copy()
        output = self._forward_enhancer(crop_frame)
        
        if blend < 100:
            alpha = blend / 100.0
            original_resized = cv2.resize(original, (output.shape[1], output.shape[0])) if original.shape[:2] != output.shape[:2] else original
            output = cv2.addWeighted(output, alpha, original_resized, 1 - alpha, 0)
        
        return output
    
    def process_frame(
        self, 
        source_embedding, 
        frame, 
        target_faces=None,
        enhance: bool = True,
        enhancer_blend: int = 80,
        reference_face: Optional[Any] = None,
    ):
        """
        Process a single frame matching FaceFusion 3.0.1 execution steps.
        """
        if target_faces is None:
            raw_faces = self.get_many_faces([frame])
        else:
            raw_faces = target_faces
        
        faces = self.select_target_faces(raw_faces, reference_face=reference_face)
        if not faces:
            return frame
        
        temp_frame = frame.copy()
        model_size = self.model_info.get("size", (128, 128))
        model_template = self.model_info.get("template", "arcface_128")
        
        for target_face in faces:
            # FaceFusion 3.0.1 uses pure source_embedding for inswapper (weight == 0 when face_swapper_weight == 0.5)
            if abs(self.face_swapper_weight - 0.5) < 1e-4 or self.model_type == "inswapper":
                blended_embedding = source_embedding.reshape(1, -1).astype(numpy.float32)
            else:
                weight = numpy.interp(self.face_swapper_weight, [0, 1], [0.35, -0.35]).astype(numpy.float32)
                target_embedding = target_face.embedding
                if self.model_info.get("type") in ["hififace", "hyperswap", "simswap"]:
                    target_embedding = target_embedding / numpy.linalg.norm(target_embedding)
                blended_embedding = (source_embedding.reshape(1, -1) * (1 - weight) + target_embedding.reshape(1, -1) * weight).astype(numpy.float32)
            
            # Pixel Boost sizing
            if self.pixel_boost and self.pixel_boost != "none":
                pixel_boost_size = self.unpack_resolution(self.pixel_boost)
                if pixel_boost_size[0] < model_size[0] or pixel_boost_size[1] < model_size[1]:
                    pixel_boost_size = model_size
            else:
                pixel_boost_size = model_size
                
            pixel_boost_total = max(pixel_boost_size[0] // model_size[0], 1)
            
            # 1. Warp/Crop
            crop_frame, affine_matrix = self.warp_face(
                temp_frame,
                target_face.landmark_set.get('5/68'),
                model_template,
                pixel_boost_size
            )
            
            if pixel_boost_total > 1:
                pixel_boost_vision_frames = self.implode_pixel_boost(crop_frame, pixel_boost_total, model_size)
            else:
                pixel_boost_vision_frames = [crop_frame]
                
            temp_vision_frames = []
            
            for pixel_boost_vision_frame in pixel_boost_vision_frames:
                # 2. Prepare input
                target_input = self._prepare_crop_frame(pixel_boost_vision_frame)
                
                # 3. Swap inference
                inputs = {}
                for model_input in self.session.get_inputs():
                    if model_input.name == 'source':
                        inputs['source'] = blended_embedding
                    elif model_input.name == 'target':
                        inputs['target'] = target_input
                    else:
                        if 'source' in model_input.name.lower() or 'embed' in model_input.name.lower():
                            inputs[model_input.name] = blended_embedding
                        else:
                            inputs[model_input.name] = target_input
                
                output = self.session.run(None, inputs)[0][0]
                
                # 4. Normalize output
                normalized_output = self._normalize_crop_frame(output)
                temp_vision_frames.append(normalized_output)
                
            if pixel_boost_total > 1:
                result_crop_frame = self.explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)
            else:
                result_crop_frame = temp_vision_frames[0]
            
            # 6. Create mask (facefusion-compatible box mask with blur/padding)
            crop_size = (pixel_boost_size[1], pixel_boost_size[0])  # (width, height)
            box_mask = create_box_mask(crop_size, self.mask_blur, self.mask_padding)
            
            # Expand dims for paste_back compatibility
            mask = numpy.expand_dims(box_mask, axis=-1)
            
            # 7. Paste back swapped face
            temp_frame = self.paste_back(temp_frame, result_crop_frame, mask, affine_matrix)
        
        # 8. Face Enhancement pass (FaceFusion 3.0.1 face_enhancer.py: (0,0,0,0) box mask padding + post-paste blend)
        if enhance and self.enhancer_session and enhancer_blend > 0:
            for target_face in faces:
                unenhanced_frame = temp_frame.copy()
                
                enhance_size = self.enhancer_info.get("size", (512, 512))
                enhance_template = self.enhancer_info.get("template", "ffhq_512")
                
                enhance_crop, enhance_affine_matrix = self.warp_face(
                    temp_frame,
                    target_face.landmark_set.get('5/68'),
                    enhance_template,
                    enhance_size
                )
                
                enhanced_crop = self._forward_enhancer(enhance_crop)
                
                enhance_crop_size = (enhance_size[1], enhance_size[0])
                enhance_box_mask = create_box_mask(enhance_crop_size, self.mask_blur, [0, 0, 0, 0])
                enhance_mask = numpy.expand_dims(enhance_box_mask, axis=-1)
                
                paste_vision_frame = self.paste_back(temp_frame, enhanced_crop, enhance_mask, enhance_affine_matrix)
                
                face_enhancer_blend = 1.0 - (enhancer_blend / 100.0)
                temp_frame = cv2.addWeighted(unenhanced_frame, face_enhancer_blend, paste_vision_frame, 1.0 - face_enhancer_blend, 0)
        
        return temp_frame
    
    def analyze_video(self, target_path: str) -> List[Tuple[int, Any]]:
        """
        Analyze video and cache face detections.
        """
        cache_path = target_path + '.faces.pkl'
        
        if os.path.exists(cache_path):
            log(f"Loading cached face data: {os.path.basename(cache_path)}")
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    log(f"Cache loaded: {len(data)} frames with faces")
                    return data
            except Exception as e:
                log(f"Cache load failed: {e}. Re-analyzing.", "WARN")
        
        log(f"Analyzing target video: {os.path.basename(target_path)}")
        cap = cv2.VideoCapture(target_path)
        _active_captures.append(cap)
        
        face_data = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        faces_found = 0
        start_time = time.time()
        
        log(f"Video has {total_frames} frames to analyze")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.get_many_faces([frame])
            if faces:
                face_data.append((frame_idx, faces))
                faces_found += 1
            
            frame_idx += 1
            if frame_idx % 25 == 0:
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed if elapsed > 0 else 0
                pct = frame_idx * 100 // total_frames
                print(f"\r[Analyzing] {frame_idx}/{total_frames} ({pct}%) - {fps:.1f} fps - {faces_found} faces found", end='', flush=True)
        
        print()
        cap.release()
        if cap in _active_captures:
            _active_captures.remove(cap)
        
        elapsed = time.time() - start_time
        log(f"Analysis complete: {faces_found} frames with faces in {elapsed:.1f}s")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(face_data, f)
            log(f"Analysis cached to {os.path.basename(cache_path)}")
        except Exception as e:
            log(f"Could not save cache: {e}", "WARN")
        
        return face_data
    
    def _generate_thumbnail(self, video_path: str, timestamp: float = 0.5) -> Optional[str]:
        thumb_path = os.path.splitext(video_path)[0] + '.png'
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(timestamp),
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '2',
                thumb_path
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            log(f"Thumbnail: {os.path.basename(thumb_path)}")
            return thumb_path
        except subprocess.TimeoutExpired:
            log("Thumbnail generation timed out", "WARN")
            return None
        except Exception as e:
            log(f"Thumbnail generation failed: {e}", "WARN")
            return None
    
    def process_video(
        self,
        source: Any,
        target_path: str,
        output_path: str,
        # Quality parameters (Dune 3.0.1 defaults: output_video_quality=80 -> cq 10, veryfast)
        video_quality: int = 10,
        use_gpu_encode: bool = True,
        # Enhancement parameters
        enhance: bool = True,
        enhancer_blend: int = 80,
        # Output options
        generate_thumbnail: bool = True,
        # Testing options
        max_frames: int = 0,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[bool, Optional[str]]:
        if not self.initialized:
            if not self.pre_check():
                return False, "Engine initialization failed"
        
        if not os.path.exists(target_path):
            return False, f"Target not found: {target_path}"
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        process = None
        cap = None
        
        try:
            # 1. Analyze source
            if isinstance(source, numpy.ndarray) and source.ndim in (1, 2) and source.shape[-1] == 512:
                source_embedding = source
            elif isinstance(source, numpy.ndarray) and source.ndim == 3:
                source_faces = self.sort_faces_by_order(self.get_many_faces([source]), self.face_selector_order)
                source_face = self.get_average_face(source_faces)
                if not source_face:
                    return False, "No face found in source image"
                source_embedding = self._prepare_source_embedding(source_face)
            elif isinstance(source, str):
                if not os.path.exists(source):
                    return False, f"Source not found: {source}"
                log(f"Source: {os.path.basename(source)}")
                src_frame = self.read_image(source)
                source_faces = self.sort_faces_by_order(self.get_many_faces([src_frame]), self.face_selector_order)
                source_face = self.get_average_face(source_faces)
                if not source_face:
                    return False, "No face found in source image"
                log("Source face detected, preparing embedding...")
                source_embedding = self._prepare_source_embedding(source_face)
            else:
                return False, f"Unsupported source type: {type(source)}"
            
            # 2. Analyze target (cached)
            video_face_data = self.analyze_video(target_path)
            video_face_map = {k: v for k, v in video_face_data}
            
            # Extract reference face from reference_frame_number (frame 0, largest face per Dune 3.0.1)
            reference_face = None
            if self.face_selector_mode == "reference":
                ref_frame_faces = video_face_map.get(0)
                if not ref_frame_faces and video_face_data:
                    ref_frame_faces = video_face_data[0][1]
                if ref_frame_faces:
                    ordered_ref = self.sort_faces_by_order(ref_frame_faces, self.face_selector_order)
                    reference_face = self.get_one_face(ordered_ref, 0)
            
            # 3. Setup video capture
            log(f"Opening target video: {os.path.basename(target_path)}")
            cap = cv2.VideoCapture(target_path)
            _active_captures.append(cap)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            log(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
            
            # 4. Setup FFmpeg
            use_nvenc = use_gpu_encode and self.nvenc_available
            encoder = 'h264_nvenc' if use_nvenc else 'libx264'
            preset = 'fast' if use_nvenc else 'veryfast'
            quality_param = '-cq' if use_nvenc else '-crf'
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-loglevel', 'error',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-i', target_path,
                '-c:v', encoder,
                '-preset', preset,
                quality_param, str(video_quality),
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                '-movflags', '+faststart',
                output_path
            ]
            
            log(f"Encoder: {encoder} (GPU={use_nvenc}, quality={video_quality})")
            log(f"Output: {os.path.basename(output_path)}")
            
            try:
                process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                _active_processes.append(process)
            except FileNotFoundError:
                return False, "FFmpeg not found in PATH"
            
            # 5. Process frames
            log("Starting face swap processing...")
            start_time = time.time()
            frame_idx = 0
            swapped_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if max_frames > 0 and frame_idx >= max_frames:
                    break
                
                target_faces = video_face_map.get(frame_idx, [])
                if target_faces:
                    swapped_frames += 1
                
                result_frame = self.process_frame(
                    source_embedding, 
                    frame, 
                    target_faces,
                    enhance=enhance,
                    enhancer_blend=enhancer_blend,
                    reference_face=reference_face,
                )
                
                process.stdin.write(result_frame.tobytes())
                
                frame_idx += 1
                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames)
                if frame_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    pct = frame_idx * 100 // total_frames
                    eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                    print(f"\r[Swapping] {frame_idx}/{total_frames} ({pct}%) - {fps_actual:.1f} fps - ETA: {eta:.0f}s", end='', flush=True)
            
            print(flush=True)  # newline after progress
            
            # Cleanup video capture
            cap.release()
            if cap in _active_captures:
                _active_captures.remove(cap)
            cap = None
            
            # Close FFmpeg stdin and wait for encoding to complete
            log("Closing FFmpeg pipe, waiting for encoding to finish...")
            process.stdin.close()
            
            try:
                # Give FFmpeg up to 120s to finish encoding
                returncode = process.wait(timeout=120)
                stderr_text = process.stderr.read().decode('utf-8', errors='replace').strip() if process.stderr else ''
                if returncode != 0:
                    log(f"FFmpeg error (code {returncode}): {stderr_text}", "ERROR")
                    return False, f"FFmpeg error (code {returncode}): {stderr_text}"
            except subprocess.TimeoutExpired:
                process.kill()
                log("FFmpeg killed after 120s timeout", "ERROR")
                return False, "FFmpeg encoding timed out after 120s"
            
            if process in _active_processes:
                _active_processes.remove(process)
            process = None
            
            elapsed = time.time() - start_time
            log(f"Done! {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")
            log(f"Swapped faces in {swapped_frames} frames")
            
            # 6. Generate thumbnail
            if generate_thumbnail:
                self._generate_thumbnail(output_path)
            
            return True, None
            
        except BrokenPipeError:
            stderr_text = ""
            if process is not None and process.stderr is not None:
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
                try:
                    stderr_text = process.stderr.read().decode('utf-8', errors='replace').strip()
                except Exception:
                    pass
            detail = f": {stderr_text}" if stderr_text else ""
            log(f"FFmpeg pipe broken!{detail}", "ERROR")
            return False, f"FFmpeg pipe broken{detail}"
        except subprocess.TimeoutExpired:
            log("FFmpeg timed out!", "ERROR")
            return False, "FFmpeg encoding timed out"
        except Exception as e:
            log(f"Processing error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False, str(e)
        finally:
            # Cleanup on any exit
            if cap is not None:
                try:
                    cap.release()
                    if cap in _active_captures:
                        _active_captures.remove(cap)
                except:
                    pass
            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                    if process in _active_processes:
                        _active_processes.remove(process)
                except:
                    try:
                        process.kill()
                    except:
                        pass

    def process_video_frames(
        self,
        source: Any,
        target_path: str,
        enhance: bool = True,
        enhancer_blend: int = 80,
        max_frames: int = 0,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[Optional[List[numpy.ndarray]], float, Optional[str]]:
        if not self.initialized:
            if not self.pre_check():
                return None, 0.0, "Engine initialization failed"

        if not os.path.exists(target_path):
            return None, 0.0, f"Target not found: {target_path}"

        cap = None
        try:
            if isinstance(source, numpy.ndarray) and source.ndim in (1, 2) and source.shape[-1] == 512:
                source_embedding = source
            elif isinstance(source, numpy.ndarray) and source.ndim == 3:
                source_faces = self.sort_faces_by_order(self.get_many_faces([source]), self.face_selector_order)
                source_face = self.get_average_face(source_faces)
                if not source_face:
                    return None, 0.0, "No face found in source image"
                source_embedding = self._prepare_source_embedding(source_face)
            elif isinstance(source, str):
                if not os.path.exists(source):
                    return None, 0.0, f"Source not found: {source}"
                src_frame = self.read_image(source)
                source_faces = self.sort_faces_by_order(self.get_many_faces([src_frame]), self.face_selector_order)
                source_face = self.get_average_face(source_faces)
                if not source_face:
                    return None, 0.0, "No face found in source image"
                source_embedding = self._prepare_source_embedding(source_face)
            else:
                return None, 0.0, f"Unsupported source type: {type(source)}"

            video_face_data = self.analyze_video(target_path)
            video_face_map = {k: v for k, v in video_face_data}

            reference_face = None
            if self.face_selector_mode == "reference":
                ref_frame_faces = video_face_map.get(0)
                if not ref_frame_faces and video_face_data:
                    ref_frame_faces = video_face_data[0][1]
                if ref_frame_faces:
                    ordered_ref = self.sort_faces_by_order(ref_frame_faces, self.face_selector_order)
                    reference_face = self.get_one_face(ordered_ref, 0)

            cap = cv2.VideoCapture(target_path)
            _active_captures.append(cap)

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            rgb_frames: List[numpy.ndarray] = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames > 0 and frame_idx >= max_frames:
                    break

                target_faces = video_face_map.get(frame_idx, [])
                result_frame = self.process_frame(
                    source_embedding,
                    frame,
                    target_faces,
                    enhance=enhance,
                    enhancer_blend=enhancer_blend,
                    reference_face=reference_face,
                )
                rgb_frames.append(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                frame_idx += 1
                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames)

            return rgb_frames, fps, None
        except Exception as e:
            log(f"Processing frames error: {e}", "ERROR")
            return None, 0.0, str(e)
        finally:
            if cap is not None:
                try:
                    cap.release()
                    if cap in _active_captures:
                        _active_captures.remove(cap)
                except Exception:
                    pass


# Singleton engine for worker_core integration
_engine: Optional[FastFaceSwapEngine] = None


def healthcheck(config: dict) -> Tuple[bool, Optional[dict]]:
    """
    Healthcheck for worker_core integration.
    Called at worker boot to initialize and warm up the engine.
    
    Args:
        config: Configuration dict from worker config.json
        
    Returns:
        (is_healthy, config_updates_or_none)
    """
    global _engine
    try:
        faceswap_cfg = config.get('pp_fastfaceswap', {})
        swap_model = faceswap_cfg.get('swap_model', DEFAULT_SWAP_MODEL)
        enhance_model = faceswap_cfg.get('enhance_model', DEFAULT_ENHANCE_MODEL)
        
        log("Healthcheck starting...")
        _engine = FastFaceSwapEngine(
            swap_model=swap_model,
            enhance_model=enhance_model,
        )
        
        if _engine.pre_check():
            log("Healthcheck passed")
            return True, None
        else:
            log("Healthcheck failed: pre_check returned False", "ERROR")
            return False, None
            
    except Exception as e:
        log(f"Healthcheck failed: {e}", "ERROR")
        return False, None


def get_engine() -> FastFaceSwapEngine:
    """
    Get the pre-loaded engine instance.
    For use in worker_core's process_job().
    """
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized - healthcheck not run?")
    return _engine


if __name__ == "__main__":
    # Quick test
    engine = FastFaceSwapEngine()
    if engine.pre_check():
        print("\n✓ Engine initialized successfully")
        print(f"  Swap model: {engine.swap_model_name}")
        print(f"  Enhance model: {engine.enhance_model_name}")
        print(f"  NVENC available: {engine.nvenc_available}")
    else:
        print("\n✗ Engine initialization failed")
