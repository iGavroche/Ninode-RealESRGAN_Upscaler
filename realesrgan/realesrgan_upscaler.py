#!/usr/bin/python
'''RealESRGAN Upscaler node.'''
# pylint: disable=invalid-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=unused-variable
# pylint: disable=bare-except
# pylint: disable=too-many-locals
# pylint: disable=line-too-long
# pylint: disable=no-member

# Import the Python modules.
import os
import gc
import warnings
import pathlib

# Set some module strings.
__author__ = "zentrocdot"
__copyright__ = "© Copyright 2025, zentrocdot"
__version__ = "0.0.1.1"

# Import the third party Python modules.
from PIL import Image
import cv2
import numpy as np
import torch
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from concurrent.futures import ThreadPoolExecutor
import threading

# Import ComfyUI progress reporting
from comfy.utils import ProgressBar

# Disable future warning.
warnings.filterwarnings("ignore", category=FutureWarning)

# Create a context manager.
class ClearCache:
    '''Clear cache class.'''
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()

# Global model cache to avoid reloading models
_model_cache = {}
_thread_lock = threading.Lock()

# Get the number of GPUs.
NUM_GPUS = torch.cuda.device_count()

# Initialse the GPU string.
GPU_STR = ""

# Create the GPU string.
for i in range(NUM_GPUS):
    VRAM = torch.cuda.get_device_properties(i).total_memory
    VRAM = str(int(VRAM / 1000**3)) + " GB"
    info = torch.cuda.get_device_name(i)
    print(i, info)
    GPU_STR = GPU_STR + str(i) + " " + info + " " + VRAM + "\n"
GPU_STR = GPU_STR.lstrip().rstrip()

# Create list with the GPUs numbers.
GPU_LIST = list(range(NUM_GPUS))

# Set some paths.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
MODELS_PATH = ''.join([str(PARENT_PATH), "/models"])

# Set file paths.
MOD_DIR = {'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
           'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
           'RealESRNet_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
           'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'}

# Download models.
for i, (k, v) in enumerate(MOD_DIR.items()):
    model_path = '/'.join([str(MODELS_PATH), k])
    model_file = pathlib.Path(model_path)
    print(model_file)
    if not model_file.is_file():
        response = requests.get(v, timeout=30)
        if response.status_code == 200:
            with open(model_path, 'wb') as file:
                file.write(response.content)
            print('File download succeeded!')
        else:
            print('File download failed!')

# Read models in dir into list.
MODS = []
for f in os.listdir(MODELS_PATH):
    if f.endswith('.pth'):
        MODS.append(f)

# -------------------------------
# Convert Tensor to PIL function.
# -------------------------------
def tensor2pil(image):
    '''Tensor to PIL image.'''
    # Convert tensor to numpy and handle batch dimensions
    np_image = image.cpu().numpy()
    
    # Remove batch dimensions if present
    while len(np_image.shape) > 3:
        np_image = np_image.squeeze(0)
    
    # Ensure we have the right shape (H, W, C)
    if len(np_image.shape) == 3 and np_image.shape[0] == 3:
        # If shape is (C, H, W), transpose to (H, W, C)
        np_image = np_image.transpose(1, 2, 0)
    elif len(np_image.shape) == 3 and np_image.shape[2] == 3:
        # If shape is (H, W, C), it's already correct
        pass
    else:
        # Handle other cases - try to reshape to (H, W, C)
        if np_image.size == np_image.shape[-1] * np_image.shape[-2] * 3:
            np_image = np_image.reshape(-1, np_image.shape[-2], 3)
    
    # Ensure the image is in the correct format for PIL
    if np_image.dtype != np.uint8:
        np_image = np.clip(255. * np_image, 0, 255).astype(np.uint8)
    
    # Return a PIL image.
    return Image.fromarray(np_image)

# ---------------------------------
# Convert Numpy to Tensor function.
# ---------------------------------
def numpy2tensor(image):
    '''Numpy image to Tensor.'''
    # Return a tensor.
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def numpy2tensor_batch(image):
    '''Numpy image to Tensor for batch processing (no extra dimensions).'''
    # Return a tensor without adding extra batch dimension
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)

# ----------------------------
# Model caching functions
# ----------------------------
def get_cached_model(models, gpu_id, tile, tile_pad, pre_pad, fp_fmt, netscale, denoise):
    '''Get or create a cached model to avoid reloading.'''
    MODEL_PATH = '/'.join([str(MODELS_PATH), models])
    # Cache key for the base model (without tile size)
    model_cache_key = f"{MODEL_PATH}_{gpu_id}_{fp_fmt}_{netscale}_{denoise}"
    
    # Thread-safe model caching
    with _thread_lock:
        # Check if we have a cached model
        if model_cache_key in _model_cache:
            print(f"RealESRGAN: Using cached model for {models}")
            cached_model = _model_cache[model_cache_key]
        else:
            print(f"RealESRGAN: Creating new model for {models}")
            
            # Check if the model file exists.
            if not os.path.isfile(MODEL_PATH):
                return None, f"Model file not found: {MODEL_PATH}"
            
            # Define the upscaler upsampler - create model and pass to RealESRGANer
            with ClearCache():
                try:
                    print(f"RealESRGAN: Creating RRDBNet model with scale={netscale}")
                    # Create the RRDBNet model with standard parameters for RealESRGAN
                    # Use netscale (4) for the model architecture, not the output scale
                    # For RealESRGAN_x4plus, the model should always be scale=4
                    model_scale = 4 if models == 'RealESRGAN_x4plus.pth' else netscale
                    print(f"RealESRGAN: Using model_scale={model_scale} for {models}")
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=model_scale, num_feat=64, num_block=23, num_grow_ch=32)
                    print(f"RealESRGAN: RRDBNet model created successfully, conv_first weight shape: {model.conv_first.weight.shape}")
                    
                    # Cache the model
                    _model_cache[model_cache_key] = model
                    cached_model = model
                    
                except Exception as e:
                    ERROR = f"Failed to create model: {str(e)}"
                    print(f"RealESRGAN: {ERROR}")
                    return None, ERROR
    
    # Set dni_weight to control the denoise strength.
    if denoise > 0:
        dni_weight = [denoise, 1 - denoise]
    else:
        dni_weight = None
    
    # Create RealESRGANer with cached model and current tile size
    try:
        print(f"RealESRGAN: Creating RealESRGANer with model_path: {MODEL_PATH}")
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=MODEL_PATH,
            model=cached_model,
            dni_weight=dni_weight,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp_fmt,
            gpu_id=gpu_id
        )
        print("RealESRGAN: RealESRGANer created successfully")
        return upsampler, None
        
    except Exception as e:
        ERROR = f"Failed to create upsampler: {str(e)}"
        print(f"RealESRGAN: {ERROR}")
        return None, ERROR

# ----------------------------
# RealESRGAN Upscaler function
# ----------------------------
def upscaler(input_img, outscale, gpu_id, tile, fp_fmt, denoise, netscale, tile_pad, pre_pad, models, optimize_memory=True, fast_mode=False):
    """Inference upscaler using Real-ESRGAN with optimizations.
    """
    ERROR = None
    print(f"RealESRGAN: upscaler called with model: {models}, scale: {outscale}, tile: {tile}")
    
    # Convert PIL image to Numpy array (optimized)
    image = np.array(input_img, dtype=np.uint8)  # Ensure uint8 for memory efficiency
    print(f"RealESRGAN: Input image shape: {image.shape}")
    
    # Optimized tile size calculation
    h, w = image.shape[:2]
    max_dimension = max(h, w)
    original_tile = tile
    
    # Always apply basic memory safety - prevent tiles larger than image dimensions
    if tile > max_dimension:
        tile = max_dimension
        print(f"RealESRGAN: Reduced tile size from {original_tile} to {tile} (image dimension limit)")
    
    # Additional safety for very large tiles that could cause memory issues
    # Calculate estimated memory usage: tile_size^2 * 4 * scale^2 * 3 channels
    estimated_memory_gb = (tile * tile * 4 * (outscale ** 2) * 3) / (1024**3)
    if estimated_memory_gb > 10:  # If estimated memory > 10GB
        # Reduce tile size to keep memory under 10GB
        max_safe_tile = int((10 * (1024**3) / (4 * (outscale ** 2) * 3)) ** 0.5)
        safe_tile = min(tile, max_safe_tile, 512)  # Cap at 512 for safety
        if safe_tile != tile:
            tile = safe_tile
            print(f"RealESRGAN: Reduced tile size from {original_tile} to {tile} for memory safety (estimated {estimated_memory_gb:.1f}GB -> {estimated_memory_gb * (tile/original_tile)**2:.1f}GB)")
    elif tile > 512 and max_dimension > 512:
        # For images larger than 512px, use more conservative tile sizes
        safe_tile = min(tile, 512)
        if safe_tile != tile:
            tile = safe_tile
            print(f"RealESRGAN: Reduced tile size from {original_tile} to {tile} for memory safety")
    
    if optimize_memory:
        # More conservative tile sizing to prevent distortions
        if fast_mode:
            # Fast mode: use smaller tiles but maintain quality
            if max_dimension <= 512:
                optimal_tile = min(tile, max_dimension, 256)  # Don't go below 256
            elif max_dimension <= 1024:
                optimal_tile = min(tile, max_dimension // 2, 256)  # Minimum 256
            else:
                optimal_tile = min(tile, max_dimension // 2, 256)  # Minimum 256
        else:
            # Normal mode: balanced performance and quality
            if max_dimension <= 512:
                # Small images: use larger tiles for efficiency
                optimal_tile = min(tile, max_dimension)
            elif max_dimension <= 1024:
                # Medium images: moderate tiles but don't go too small
                optimal_tile = min(tile, max_dimension // 2, 512)
            else:
                # Large images: conservative tiles but maintain quality
                optimal_tile = min(tile, max_dimension // 2, 512)  # Increased from 256 to 512
        
        # Only apply optimization if it's a meaningful improvement and doesn't hurt quality
        if optimal_tile != original_tile and optimal_tile >= 256:  # Don't go below 256 for quality
            tile = optimal_tile
            mode_str = "fast" if fast_mode else "optimized"
            print(f"RealESRGAN: {mode_str} tile size from {original_tile} to {tile} for image size {max_dimension}")
        elif optimal_tile < 256:
            print(f"RealESRGAN: Skipping tile optimization to maintain quality (would reduce to {optimal_tile})")
    
    # Ensure tile size is reasonable for quality (minimum 128 for very small images)
    if tile < 128:
        print(f"RealESRGAN: Warning - tile size {tile} is very small, may cause quality issues")
        tile = max(128, tile)  # Ensure minimum tile size
    
    # Get cached model
    upsampler, error = get_cached_model(models, gpu_id, tile, tile_pad, pre_pad, fp_fmt, netscale, denoise)
    if error is not None:
        return None, error
    
    # Optimized enhancement with better error handling
    try:
        print(f"RealESRGAN: Starting enhancement with outscale: {outscale}, tile: {tile}, tile_pad: {tile_pad}, pre_pad: {pre_pad}")
        
        # Pre-allocate memory and optimize for the specific scale
        if outscale == int(outscale):  # Integer scale factor
            outscale_int = int(outscale)
            # Use more efficient processing for integer scales
            outimg, _ = upsampler.enhance(image, outscale=outscale_int)
        else:
            # For non-integer scales, use the original method
            outimg, _ = upsampler.enhance(image, outscale=outscale)
            
        print(f"RealESRGAN: Enhancement completed, output shape: {outimg.shape if outimg is not None else 'None'}")
        
        # Quality check - warn if output seems suspicious
        if outimg is not None:
            h_out, w_out = outimg.shape[:2]
            expected_h = int(h * outscale)
            expected_w = int(w * outscale)
            if abs(h_out - expected_h) > 10 or abs(w_out - expected_w) > 10:
                print(f"RealESRGAN: Warning - output dimensions may be incorrect. Expected: {expected_h}x{expected_w}, Got: {h_out}x{w_out}")
        
    except RuntimeError as error:
        # More specific error handling
        error_str = str(error).lower()
        torch.cuda.empty_cache()
        gc.collect()
        outimg = None
        
        if "out of memory" in error_str or "hip out of memory" in error_str:
            ERROR = f"CUDA/HIP out of memory. Try reducing tile size from {tile} to {tile//2} or using a smaller model."
        elif "size mismatch" in error_str:
            ERROR = f"Model size mismatch. Ensure netscale={netscale} matches your model architecture."
        else:
            ERROR = f"Runtime error: {str(error)}"
        print(f"RealESRGAN: Runtime error: {ERROR}")
        
    except Exception as error:
        torch.cuda.empty_cache()
        gc.collect()
        outimg = None
        error_str = str(error)
        
        # Handle specific RealESRGAN library errors
        if "cannot access local variable 'output_tile'" in error_str:
            ERROR = f"RealESRGAN library error - likely due to memory issues. Try reducing tile size from {tile} to {tile//2} or enable memory optimization."
        elif "out of memory" in error_str.lower() or "hip out of memory" in error_str.lower():
            ERROR = f"CUDA/HIP out of memory. Try reducing tile size from {tile} to {tile//2} or using a smaller model."
        else:
            ERROR = f"Error during upscaling: {error_str}"
        
        print(f"RealESRGAN: General error: {ERROR}")
    
    # Return the upscaled image.
    return outimg, ERROR

def upscaler_no_cache(input_img, outscale, gpu_id, tile, fp_fmt, denoise, netscale, tile_pad, pre_pad, models, optimize_memory=True, fast_mode=False):
    """Inference upscaler using Real-ESRGAN without caching (for parallel processing)."""
    ERROR = None
    print(f"RealESRGAN: upscaler_no_cache called with model: {models}, scale: {outscale}, tile: {tile}")
    
    # Convert PIL image to Numpy array (optimized)
    image = np.array(input_img, dtype=np.uint8)  # Ensure uint8 for memory efficiency
    print(f"RealESRGAN: Input image shape: {image.shape}")
    
    # Optimized tile size calculation (same as main upscaler function)
    h, w = image.shape[:2]
    max_dimension = max(h, w)
    original_tile = tile
    
    # Always apply basic memory safety - prevent tiles larger than image dimensions
    if tile > max_dimension:
        tile = max_dimension
        print(f"RealESRGAN: Reduced tile size from {original_tile} to {tile} (image dimension limit)")
    
    # Additional safety for very large tiles that could cause memory issues
    # Calculate estimated memory usage: tile_size^2 * 4 * scale^2 * 3 channels
    estimated_memory_gb = (tile * tile * 4 * (outscale ** 2) * 3) / (1024**3)
    if estimated_memory_gb > 10:  # If estimated memory > 10GB
        # Reduce tile size to keep memory under 10GB
        max_safe_tile = int((10 * (1024**3) / (4 * (outscale ** 2) * 3)) ** 0.5)
        safe_tile = min(tile, max_safe_tile, 512)  # Cap at 512 for safety
        if safe_tile != tile:
            tile = safe_tile
            print(f"RealESRGAN: Reduced tile size from {original_tile} to {tile} for memory safety (estimated {estimated_memory_gb:.1f}GB -> {estimated_memory_gb * (tile/original_tile)**2:.1f}GB)")
    elif tile > 512 and max_dimension > 512:
        # For images larger than 512px, use more conservative tile sizes
        safe_tile = min(tile, 512)
        if safe_tile != tile:
            tile = safe_tile
            print(f"RealESRGAN: Reduced tile size from {original_tile} to {tile} for memory safety")
    
    if optimize_memory:
        # More conservative tile sizing to prevent distortions
        if fast_mode:
            # Fast mode: use smaller tiles but maintain quality
            if max_dimension <= 512:
                optimal_tile = min(tile, max_dimension, 256)  # Don't go below 256
            elif max_dimension <= 1024:
                optimal_tile = min(tile, max_dimension // 2, 256)  # Minimum 256
            else:
                optimal_tile = min(tile, max_dimension // 2, 256)  # Minimum 256
        else:
            # Normal mode: balanced performance and quality
            if max_dimension <= 512:
                # Small images: use larger tiles for efficiency
                optimal_tile = min(tile, max_dimension)
            elif max_dimension <= 1024:
                # Medium images: moderate tiles but don't go too small
                optimal_tile = min(tile, max_dimension // 2, 512)
            else:
                # Large images: conservative tiles but maintain quality
                optimal_tile = min(tile, max_dimension // 2, 512)  # Increased from 256 to 512
        
        # Only apply optimization if it's a meaningful improvement and doesn't hurt quality
        if optimal_tile != original_tile and optimal_tile >= 256:  # Don't go below 256 for quality
            tile = optimal_tile
            mode_str = "fast" if fast_mode else "optimized"
            print(f"RealESRGAN: {mode_str} tile size from {original_tile} to {tile} for image size {max_dimension}")
        elif optimal_tile < 256:
            print(f"RealESRGAN: Skipping tile optimization to maintain quality (would reduce to {optimal_tile})")
    
    # Ensure tile size is reasonable for quality (minimum 128 for very small images)
    if tile < 128:
        print(f"RealESRGAN: Warning - tile size {tile} is very small, may cause quality issues")
        tile = max(128, tile)  # Ensure minimum tile size
    
    MODEL_PATH = '/'.join([str(MODELS_PATH), models])
    
    # Check if the model file exists.
    if not os.path.isfile(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"
    
    # Set dni_weight to control the denoise strength.
    if denoise > 0:
        dni_weight = [denoise, 1 - denoise]
    else:
        dni_weight = None
    
    # Create model and RealESRGANer without caching
    with ClearCache():
        try:
            print(f"RealESRGAN: Creating RRDBNet model with scale={netscale}")
            # Create the RRDBNet model with standard parameters for RealESRGAN
            model_scale = 4 if models == 'RealESRGAN_x4plus.pth' else netscale
            print(f"RealESRGAN: Using model_scale={model_scale} for {models}")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=model_scale, num_feat=64, num_block=23, num_grow_ch=32)
            print(f"RealESRGAN: RRDBNet model created successfully, conv_first weight shape: {model.conv_first.weight.shape}")
            
            print(f"RealESRGAN: Creating RealESRGANer with model_path: {MODEL_PATH}")
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=MODEL_PATH,
                model=model,
                dni_weight=dni_weight,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=not fp_fmt,
                gpu_id=gpu_id
            )
            print("RealESRGAN: RealESRGANer created successfully")
            
        except Exception as e:
            ERROR = f"Failed to create upsampler: {str(e)}"
            print(f"RealESRGAN: {ERROR}")
            return None, ERROR
    
    # Determine the output image.
    try:
        print(f"RealESRGAN: Starting enhancement with outscale: {outscale}")
        outimg, _ = upsampler.enhance(image, outscale=outscale)
        print(f"RealESRGAN: Enhancement completed, output shape: {outimg.shape if outimg is not None else 'None'}")
    except RuntimeError as error:
        torch.cuda.empty_cache()
        gc.collect()
        outimg = None
        if "out of memory" in str(error).lower() or "hip out of memory" in str(error).lower():
            ERROR = "CUDA/HIP out of memory. Try reducing tile size or using a smaller model."
        else:
            ERROR = f"Runtime error: {str(error)}"
        print(f"RealESRGAN: Runtime error: {ERROR}")
    except Exception as error:
        torch.cuda.empty_cache()
        gc.collect()
        outimg = None
        ERROR = f"Error during upscaling: {str(error)}"
        print(f"RealESRGAN: General error: {ERROR}")
    
    # Return the upscaled image.
    return outimg, ERROR

def process_single_frame(args):
    """Process a single frame for parallel processing."""
    frame_idx, single_image, scale_factor, gpu_id, tile_number, fp_format, denoise, netscale, tile_pad, pre_pad, models, optimize_memory, fast_mode = args
    
    try:
        # Create a PIL image.
        img_input = tensor2pil(single_image)
        # Copy image.
        imgNEW = img_input.copy()
        # Upscale image.
        print(f"RealESRGAN: Processing frame {frame_idx+1}")
        
        # Clear any existing GPU memory before processing
        torch.cuda.empty_cache()
        gc.collect()
        
        # For parallel processing, don't use cached models to avoid thread safety issues
        imgNEW, frame_error = upscaler_no_cache(imgNEW, scale_factor, gpu_id, tile_number,
                                               fp_format, denoise, netscale, tile_pad,
                                               pre_pad, models, optimize_memory, fast_mode)
        
        # Clear GPU memory after processing
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"RealESRGAN: Exception in frame {frame_idx+1}: {str(e)}")
        imgNEW = None
        frame_error = f"Exception in parallel processing: {str(e)}"
    
    # Check if ERROR or imgNEW is None.
    if frame_error is not None or imgNEW is None:
        print(f"RealESRGAN: Frame {frame_idx+1} failed - ERROR: {frame_error}")
        # Create error image with same dimensions as input
        h, w = img_input.size[1], img_input.size[0]  # PIL size is (width, height)
        # Scale up to match expected output size
        scale_factor_int = int(scale_factor)
        error_h, error_w = h * scale_factor_int, w * scale_factor_int
        # Create an empty image with correct dimensions
        imgNEW = np.zeros([error_h, error_w, 3], dtype=np.uint8)
        # Add text to the image
        text = f"Error - Frame {frame_idx+1}"
        position = (20, error_h // 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(1, error_w // 500)  # Scale font with image size
        color = (255, 0, 0)
        thickness = max(2, error_w // 300)  # Scale thickness with image size
        cv2.putText(imgNEW, text, position, font, font_scale, color, thickness)
    else:
        print(f"RealESRGAN: Frame {frame_idx+1} processed successfully")
    
    # Convert back to tensor
    upscaled_tensor = numpy2tensor_batch(imgNEW)
    return frame_idx, upscaled_tensor, frame_error

# ++++++++++++++++++++++++
# Class RealEsrganUpscaler
# ++++++++++++++++++++++++
class RealEsrganUpscaler:
    '''real esrgan upscaler.'''

    @classmethod
    def INPUT_TYPES(cls):
        '''Define the node input types.'''
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 1024.0, "step": 0.1, "tooltip": "Output scale factor. For RealESRGAN_x4plus with netscale=4, use 2.0 for 2x upscaling."}),
                "tile_number": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 64}),
                "tile_pad": ("INT", {"default": 10, "min": 0, "max": 1024, "step": 1}),
                "pre_pad": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
                "fp_format": (["fp16", "fp32"], {}),
                "denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gpu_id": (GPU_LIST, {}),
                "netscale": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1, "tooltip": "Model architecture scale (4 for RealESRGAN_x4plus, 2 for RealESRGAN_x2plus). Should match the model's native scale."}),
                "models": (MODS, {}),
                "parallel_workers": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Number of parallel workers for batch processing. Higher values may improve performance but use more memory."}),
        "optimize_memory": ("BOOLEAN", {"default": False, "tooltip": "Enable memory optimizations including intelligent tile sizing and memory management. May affect quality if tiles become too small."}),
        "fast_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable fast mode with reduced quality for faster processing. Uses smaller tiles and optimized settings. May cause distortions."}),
            }
        }

    # Set the ComfyUI related variables.
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING",)
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height", "config_str", "error_str",)
    FUNCTION = "realesrgan_upscaler"
    CATEGORY = "💊 RealEsrganUpscaler"
    DESCRIPTION = "Upscaling using RealESRGAN."
    OUTPUT_NODE = True

    def realesrgan_upscaler(self, image, gpu_id, scale_factor, tile_number,
                            fp_format, models, denoise, netscale, tile_pad,
                            pre_pad, parallel_workers=1, optimize_memory=True, fast_mode=False):
        '''RealESRGAN upscaler.'''
        # Set value for paranoia reason.
        gpu_id = int(gpu_id)
        
        # Check if we have a batch of images (video workflow)
        if len(image.shape) == 4:  # Batch of images
            print(f"RealESRGAN: Processing batch of {image.shape[0]} images with {parallel_workers} parallel workers")
            
            # Create progress bar for batch processing
            pbar = ProgressBar(image.shape[0])
            
            # Prepare arguments for parallel processing
            frame_args = []
            for i in range(image.shape[0]):
                single_image = image[i]
                frame_args.append((i, single_image, scale_factor, gpu_id, tile_number,
                                 fp_format, denoise, netscale, tile_pad, pre_pad, models,
                                 optimize_memory, fast_mode))
            
            # Process frames in parallel or sequentially based on parallel_workers
            upscaled_images = [None] * image.shape[0]
            batch_errors = []
            
            if parallel_workers > 1:
                # Parallel processing (no caching to avoid race conditions)
                # Reduce tile size for parallel processing to prevent memory issues
                parallel_tile = min(tile_number, 256)  # More conservative tile size for parallel
                print(f"RealESRGAN: Using reduced tile size {parallel_tile} for parallel processing")
                
                # Update frame args with reduced tile size
                frame_args = [(i, single_image, scale_factor, gpu_id, parallel_tile, fp_format, denoise, netscale, tile_pad, pre_pad, models, optimize_memory, fast_mode) 
                             for i, single_image in enumerate(image)]
                
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    # Submit all tasks
                    future_to_index = {executor.submit(process_single_frame, args): args[0] for args in frame_args}
                    
                    # Collect results as they complete
                    completed = 0
                    for future in future_to_index:
                        frame_idx, upscaled_tensor, frame_error = future.result()
                        
                        # Debug: Check tensor validity
                        if upscaled_tensor is not None:
                            print(f"RealESRGAN: Frame {frame_idx+1} tensor shape: {upscaled_tensor.shape}, dtype: {upscaled_tensor.dtype}")
                            # Check for corrupted tensors (all zeros, NaN, or extreme values)
                            if torch.all(upscaled_tensor == 0):
                                print(f"RealESRGAN: Warning - Frame {frame_idx+1} tensor is all zeros!")
                            elif torch.isnan(upscaled_tensor).any():
                                print(f"RealESRGAN: Warning - Frame {frame_idx+1} tensor contains NaN values!")
                            elif torch.max(upscaled_tensor) > 10 or torch.min(upscaled_tensor) < -10:
                                print(f"RealESRGAN: Warning - Frame {frame_idx+1} tensor has extreme values: min={torch.min(upscaled_tensor):.3f}, max={torch.max(upscaled_tensor):.3f}")
                        else:
                            print(f"RealESRGAN: Warning - Frame {frame_idx+1} tensor is None!")
                        
                        upscaled_images[frame_idx] = upscaled_tensor
                        if frame_error is not None:
                            batch_errors.append(f"Frame {frame_idx+1}: {frame_error}")
                        completed += 1
                        pbar.update(1)
            else:
                # Sequential processing with caching (original behavior)
                for i in range(image.shape[0]):
                    # Extract single image from batch
                    single_image = image[i]
                    # Create a PIL image.
                    img_input = tensor2pil(single_image)
                    # Copy image.
                    imgNEW = img_input.copy()
                    # Upscale image.
                    print(f"RealESRGAN: Processing frame {i+1}/{image.shape[0]}")
                    imgNEW, frame_error = upscaler(imgNEW, scale_factor, gpu_id, tile_number,
                                                 fp_format, denoise, netscale, tile_pad,
                                                 pre_pad, models, optimize_memory, fast_mode)
                    # Check if ERROR or imgNEW is None.
                    if frame_error is not None or imgNEW is None:
                        print(f"RealESRGAN: Frame {i+1} failed - ERROR: {frame_error}")
                        batch_errors.append(f"Frame {i+1}: {frame_error}")
                        # Create error image with same dimensions as input
                        h, w = img_input.size[1], img_input.size[0]  # PIL size is (width, height)
                        # Scale up to match expected output size
                        scale_factor_int = int(scale_factor)
                        error_h, error_w = h * scale_factor_int, w * scale_factor_int
                        # Create an empty image with correct dimensions
                        imgNEW = np.zeros([error_h, error_w, 3], dtype=np.uint8)
                        # Add text to the image
                        text = f"Error - Frame {i+1}"
                        position = (20, error_h // 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = max(1, error_w // 500)  # Scale font with image size
                        color = (255, 0, 0)
                        thickness = max(2, error_w // 300)  # Scale thickness with image size
                        cv2.putText(imgNEW, text, position, font, font_scale, color, thickness)
                    else:
                        print(f"RealESRGAN: Frame {i+1} processed successfully")
                    
                    # Convert back to tensor and add to batch
                    upscaled_tensor = numpy2tensor_batch(imgNEW)
                    upscaled_images[i] = upscaled_tensor
                    
                    # Update progress bar
                    pbar.update(1)
            
            # Combine all errors into a single error string
            if batch_errors:
                ERROR = f"Batch processing errors: {'; '.join(batch_errors)}"
            else:
                ERROR = None
            
            # Validate all tensors before stacking
            valid_tensors = []
            for i, tensor in enumerate(upscaled_images):
                if tensor is not None and tensor.shape[0] > 0:
                    # Ensure tensor is in the correct format
                    if len(tensor.shape) == 3:  # H, W, C format
                        valid_tensors.append(tensor)
                        print(f"RealESRGAN: Frame {i+1} validated - shape: {tensor.shape}")
                    else:
                        print(f"RealESRGAN: Warning - Frame {i+1} has invalid shape: {tensor.shape}")
                        # Create a placeholder tensor with correct shape
                        placeholder = torch.zeros((512, 512, 3), dtype=torch.float32)
                        valid_tensors.append(placeholder)
                else:
                    print(f"RealESRGAN: Warning - Frame {i+1} is None or empty, creating placeholder")
                    # Create a placeholder tensor
                    placeholder = torch.zeros((512, 512, 3), dtype=torch.float32)
                    valid_tensors.append(placeholder)
            
            # Ensure all tensors have the same shape
            if valid_tensors:
                target_shape = valid_tensors[0].shape
                print(f"RealESRGAN: Target tensor shape: {target_shape}")
                
                # Resize all tensors to match the first one
                for i, tensor in enumerate(valid_tensors):
                    if tensor.shape != target_shape:
                        print(f"RealESRGAN: Resizing frame {i+1} from {tensor.shape} to {target_shape}")
                        # Simple resize - this might cause quality loss but prevents stacking errors
                        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)  # HWC to CHW
                        tensor = torch.nn.functional.interpolate(tensor, size=(target_shape[0], target_shape[1]), mode='bilinear', align_corners=False)
                        tensor = tensor.squeeze(0).permute(1, 2, 0)  # CHW back to HWC
                        valid_tensors[i] = tensor
                
                # Stack all upscaled images into a batch
                upscaled_batch = torch.stack(valid_tensors)
                print(f"RealESRGAN: Successfully stacked batch with shape: {upscaled_batch.shape}")
            else:
                print("RealESRGAN: Error - No valid tensors to stack!")
                # Create a fallback batch
                upscaled_batch = torch.zeros((image.shape[0], 512, 512, 3), dtype=torch.float32)
            # Get width and height from the first valid tensor
            if valid_tensors:
                first_image = valid_tensors[0]
                (height, width, channels) = first_image.shape
            else:
                # Fallback dimensions
                height, width, channels = 512, 512, 3
            # Create a mask for the batch
            maskImage = np.zeros((height, width, channels), np.uint8)
            mask = numpy2tensor_batch(maskImage)
            # Create batch mask (repeat for all frames)
            batch_mask = mask.unsqueeze(0).repeat(upscaled_batch.shape[0], 1, 1, 1)
            # Return the upscaled batch
            return (upscaled_batch, batch_mask, width, height, GPU_STR, ERROR,)
        else:
            # Single image processing
            # Create a PIL image.
            img_input = tensor2pil(image)
            # Copy image.
            imgNEW = img_input.copy()
            # Upscale image.
            imgNEW, ERROR = upscaler(imgNEW, scale_factor, gpu_id, tile_number,
                                     fp_format, denoise, netscale, tile_pad,
                                     pre_pad, models)
            # Check if ERROR or imgNEW is None.
            if ERROR is not None or imgNEW is None:
                # Set rows and cols.
                n,m = 512,512
                # Create an empty image.
                imgNEW = np.zeros([n,m,3], dtype=np.uint8)
                # Add text to the image
                text = "Oops, something went wrong!"
                position = (20, 256)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 0, 255)
                thickness = 2
                cv2.putText(imgNEW, text, position, font, font_scale, color, thickness)
            # Get width and height of the new image.
            (height, width, channels) = imgNEW.shape
            # Create a mask.
            maskImage = np.zeros((height, width, channels), np.uint8)
            mask = numpy2tensor(maskImage)
            mask = mask[:, :, :, 1]
            # Create tensors from PIL.
            image_out = numpy2tensor(imgNEW)
            # Return the upscaled image.
            return (image_out, mask, width, height, GPU_STR, ERROR,)
