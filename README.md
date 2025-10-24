# RealESRGAN Upscaler for ComfyUI [![ComfyUI - Homepage](https://img.shields.io/badge/ComfyUI-Homepage-aa00ee)](https://github.com/comfyanonymous/ComfyUI) [![ComfyUI - Manager](https://img.shields.io/badge/ComfyUI-Manager-2aeeef)](https://github.com/ltdrdata/ComfyUI-Manager)

> [!IMPORTANT]  
> **🚀 Video Workflow Support**: This node now fully supports video upscaling workflows with batch processing, parallel processing, and optimized memory management for both AMD and NVIDIA GPUs.

> [!CAUTION]
> **Installation Fix Required**: Please run the provided PowerShell script to fix a torchvision import issue. See the Installation section for details.

## Overview

This ComfyUI custom node provides high-quality image and video upscaling using RealESRGAN models. It's optimized for both AMD and NVIDIA GPUs, with special support for video workflows and batch processing.

### Key Features

- **🎬 Video Workflow Support**: Full batch processing for video frames
- **⚡ Performance Optimized**: Model caching and parallel processing options
- **🖥️ Multi-GPU Support**: Works with both AMD and NVIDIA GPUs
- **💾 Memory Management**: Intelligent tile sizing and memory optimization
- **📊 Progress Reporting**: Real-time progress updates in ComfyUI
- **🔧 Error Handling**: Comprehensive error reporting and recovery

## Prerequisites

- **ComfyUI**: Latest version recommended
- **Python**: 3.8+ with PyTorch
- **GPU**: AMD or NVIDIA GPU with sufficient VRAM
- **Memory**: Minimum 4GB VRAM (8GB+ recommended for video workflows)

## Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "RealESRGAN Upscaler" or "iGavroche"
3. Install the node

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iGavroche/Ninode-RealESRGAN_Upscaler.git
```

### Fix Required Import Issue
After installation, run the provided fix script:

**Windows (PowerShell):**
```powershell
.\fix_torchvision_import.ps1
```

**Linux/Mac (Bash):**
```bash
./scripts/fix.bash
```

**Manual Fix:**
Find and edit the file `degradations.py` in your basicsr installation:
```python
# Change this line:
from torchvision.transforms.functional_tensor import rgb_to_grayscale
# To this:
from torchvision.transforms.functional import rgb_to_grayscale
```

## Models

The following RealESRGAN models are supported:

- **RealESRGAN_x4plus.pth** (Recommended for 2x-4x upscaling)
- **RealESRGAN_x2plus.pth** (For 2x upscaling)
- **RealESRNet_x4plus.pth** (Alternative 4x model)
- **ESRGAN_SRx4_DF2KOST_official-ff704c30.pth** (High-quality 4x model)

Models are automatically downloaded to the `models/` directory during installation.

## Node Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale_factor` | FLOAT | 2.0 | Output scale factor (2.0 for 2x upscaling with RealESRGAN_x4plus) |
| `netscale` | INT | 4 | Model architecture scale (4 for RealESRGAN_x4plus, 2 for RealESRGAN_x2plus) |
| `tile_number` | INT | 512 | Tile size for processing (reduced automatically for large images) |
| `tile_pad` | INT | 10 | Padding around tiles |
| `pre_pad` | INT | 0 | Pre-padding for processing |
| `fp_format` | STRING | "fp32" | Floating point format ("fp32" or "fp16") |
| `denoise` | FLOAT | 0.0 | Denoising strength (0.0-1.0) |
| `gpu_id` | INT | 0 | GPU device ID |
| `models` | STRING | "RealESRGAN_x4plus.pth" | Model to use |
| `parallel_workers` | INT | 1 | Number of parallel workers (1=sequential with caching, 2-4=parallel) |

## Usage Examples

### Basic Image Upscaling
1. Load an image using a Load Image node
2. Connect to RealESRGAN Upscaler
3. Set `scale_factor` to desired upscaling (e.g., 2.0 for 2x)
4. Set `netscale` to match your model (4 for RealESRGAN_x4plus)
5. Connect output to Save Image node

### Video Workflow
1. Use Video Loader to load video frames
2. Connect to RealESRGAN Upscaler
3. Set `parallel_workers` to 1 for fastest processing (with caching)
4. Set `parallel_workers` to 2-4 for parallel processing (uses more memory)
5. Connect to Video Combine node

### Recommended Settings

**For RealESRGAN_x4plus.pth:**
- `scale_factor`: 2.0 (for 2x upscaling)
- `netscale`: 4
- `tile_number`: 512 (or 256 for parallel processing)

**For RealESRGAN_x2plus.pth:**
- `scale_factor`: 2.0
- `netscale`: 2
- `tile_number`: 512

## Performance Tips

### Sequential Processing (parallel_workers=1)
- **Best for**: Most use cases, especially video workflows
- **Benefits**: Uses model caching, fastest overall processing
- **Memory**: Lower memory usage
- **Speed**: Up to 50% faster than original implementation

### Parallel Processing (parallel_workers=2-4)
- **Best for**: Large batches where memory allows
- **Benefits**: Can process multiple frames simultaneously
- **Memory**: Higher memory usage (each worker loads its own model)
- **Speed**: May be 2-3x faster for large batches, but slower for small batches

### Memory Optimization
- **Large Images**: Tile size is automatically reduced
- **AMD GPUs**: More conservative memory management
- **Parallel Processing**: Uses smaller tile sizes (256 max)
- **Error Recovery**: Graceful handling of memory errors

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: torchvision.transforms.functional_tensor"**
- **Solution**: Run the provided fix script or manually edit `degradations.py`

**"CUDA/HIP out of memory"**
- **Solution**: Reduce `tile_number` or use `parallel_workers=1`

**"size mismatch for conv_first.weight"**
- **Solution**: Ensure `netscale` matches your model (4 for RealESRGAN_x4plus)

**"Error - Frame x" in video output**
- **Solution**: Check error messages in ComfyUI console, ensure proper model settings

### Performance Issues

**Slow processing:**
- Use `parallel_workers=1` for sequential processing with caching
- Ensure `netscale` matches your model
- Check GPU memory usage

**Memory errors:**
- Reduce `tile_number` (try 256 or 128)
- Use `parallel_workers=1`
- Close other applications using GPU memory

## Advanced Configuration

### Environment Variables
```bash
# For AMD GPUs with memory fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For debugging
export CUDA_LAUNCH_BLOCKING=1
```

### Custom Models
Place custom RealESRGAN models in the `models/` directory and select them from the dropdown.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## Credits

- **RealESRGAN**: [Xintao Wang](https://github.com/xinntao/Real-ESRGAN) for the excellent upscaling models
- **ComfyUI**: [ComfyUI Team](https://github.com/comfyanonymous/ComfyUI) for the amazing framework
- **Original Node**: [zentrocdot](https://github.com/zentrocdot/ComfyUI-RealESRGAN_Upscaler) for the initial implementation

## License

This project is licensed under the same terms as the original RealESRGAN project.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the changelog for known issues
3. Open an issue on the [GitHub repository](https://github.com/iGavroche/Ninode-RealESRGAN_Upscaler)

---

**Have fun upscaling! 🚀**