#!/usr/bin/python
'''RealESRGAN Upscaler node.'''
# pylint: disable=invalid-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=unused-variable

# Import the Python modules.
import warnings
import gc
import pathlib

# Set some strings.
__version__ = "0.0.0.5"

# Import the third party Python modules.
from PIL import Image
import numpy as np
import torch
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Disable future warning.
warnings.filterwarnings("ignore", category=FutureWarning)

# Create context manager.
class ClearCache:
    '''Clear cache class.'''
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()

# Get the number of GPUs.
NUM_GPUS = torch.cuda.device_count()

# Get GPU string.
GPU_STR = ""
for i in range(NUM_GPUS):
    VRAM = torch.cuda.get_device_properties(i).total_memory
    VRAM = str(int(VRAM / 1000**3)) + " GB"
    info = torch.cuda.get_device_name(i)
    print(i, info)
    GPU_STR = GPU_STR + str(i) + "\t" + info + "\t" + VRAM + "\n"
GPU_STR = GPU_STR.lstrip().rstrip()

# Create list with the GPUs numbers.
GPU_LIST = list(range(NUM_GPUS))

# Set some paths.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
MODELS_PATH = ''.join([str(PARENT_PATH), "/models"])

# Set the model.
#MODEL = "4x-UltraSharp.pth"
MODEL = "RealESRGAN_x4plus.pth"

# Set the model path.
MODEL_PATH = ''.join([str(MODELS_PATH), "/RealESRGAN_x4plus.pth"])

# Set file path.
FILE_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

# Download model.
model_file = pathlib.Path(MODEL_PATH)
print(model_file)
if not model_file.is_file():
    response = requests.get(FILE_URL, timeout=20)
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as file:
            file.write(response.content)
        print('File download succeeded!')
    else:
        print('File download failed!')

# -------------------------------
# Convert Tensor to PIL function.
# -------------------------------
def tensor2pil(image):
    '''Tensor to PIL image.'''
    # Return a PIL image.
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# -------------------------------
# Convert PIL to Tensor function.
# -------------------------------
def pil2tensor(image):
    '''PIL image to Tensor.'''
    # Return a tensor.
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# ----------------------------
# RealESRGAN Upscaler function
# ----------------------------
def upscaler(input_img, outscale, gpu_id, tile, fp_fmt):
    """Inference upscaler using Real-ESRGAN.
    """
    # Convert PIL image to Numpy array.
    image = np.array(input_img)
    # Set up the network for the RealESRGAN model.
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # Set the netscale (4) to the depending one of the model (x4).
    netscale = 4
    # Set dni_weight to control the denoise strength.
    dni_weight = None
    # Set some padding values.
    pre_pad = 0
    tile_pad = 10
    # Define the upscaler upsampler.
    with ClearCache():
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=MODEL_PATH,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp_fmt,
            gpu_id=gpu_id)
    # Determine the output image.
    try:
        outimg, _ = upsampler.enhance(image, outscale=outscale)
    except RuntimeError as error:
        torch.cuda.empty_cache()
        #raise Error("An serious error occurred 💥. " +
        #            "CUDA out of memory. Try to set tile size to a smaller number.")
    except UnboundLocalError as error:
        torch.cuda.empty_cache()
        #raise Error("An serious error occurred 💥. " +
        #            "Image size problem occurs. Tile size problem occurs.")
    # Return the upscaled image.
    return outimg

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
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "gpu_id": (GPU_LIST, {}),
                "tile_number": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
                "fp_format": (["fp16", "fp32"], {}),
                "model": ("STRING", {"multiline": False, "default": MODEL}),
            }
        }

    # Set the ComfyUI related variables.
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("IMAGE", "DATA",)
    FUNCTION = "realesrgan_upscaler"
    CATEGORY = "🔍 RealEsrganUpscaler"
    DESCRIPTION = "Upscaling using RealESRGAN."
    OUTPUT_NODE = True

    def realesrgan_upscaler(self, image, gpu_id, scale_factor, tile_number, fp_format, model):
        '''Detect ellipse.'''
        # Set value for paranoia reason.
        gpu_id = int(gpu_id)
        # Create a PIL image.
        img_input = tensor2pil(image)
        # Copy image.
        imgNEW = img_input.copy()
        # Upscale image.
        imgNEW = upscaler(imgNEW, scale_factor, gpu_id, tile_number, fp_format)
        # Create tensors from PIL.
        image_out = pil2tensor(imgNEW)
        # Return the upscaled image.
        return (image_out, GPU_STR)
