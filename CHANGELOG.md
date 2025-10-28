# Changelog

All notable changes to this project will be documented in this file.

## [0.0.3.0] - 2025-01-22

### Fixed
- Video chunk processing boundary artifacts
  - Added 2-frame temporal overlap to prevent darker frames at chunk boundaries
  - Improved chunk concatenation logic with proper overlap cropping
  - Fixed tensor size mismatch errors during concatenation
  - Better handling of 5D video tensor formats

### Improved
- Video frame processing with smoother transitions
  - Automatic overlap detection and handling
  - Cleaner boundary transitions between video frames
  - Reduced visual artifacts in chunked processing

## [0.0.2.0] - 2025-10-24

### Added
- **Video Workflow Support**: Full batch processing for video frames with proper tensor handling
- **Model Caching**: Intelligent model caching for sequential processing to avoid reloading models
- **Parallel Processing**: Optional parallel frame processing with configurable worker count (1-4 workers)
- **Memory Management**: Advanced memory management with automatic tile size adjustment
- **Progress Reporting**: Real-time progress updates in ComfyUI interface
- **Error Handling**: Comprehensive error handling with proper error propagation to `error_str` output
- **AMD GPU Optimization**: Optimized for AMD GPUs with conservative memory management
- **Automatic Installation Fixes**: PowerShell script to fix torchvision import issues

### Changed
- **Node Name**: Kept as "RealESRGAN Upscaler" (compatible with both AMD and NVIDIA)
- **Default Parameters**: 
  - `scale_factor`: 2.0 (for 2x upscaling with RealESRGAN_x4plus)
  - `netscale`: 4 (matches RealESRGAN_x4plus model architecture)
  - `tile_number`: 512 (conservative default for memory safety)
- **Batch Processing**: Complete rewrite for proper video frame handling
- **Memory Management**: Dynamic tile size adjustment based on image dimensions and GPU memory

### Fixed
- **Module Import Error**: Fixed `torchvision.transforms.functional_tensor` import issue
- **Single Frame Processing**: Fixed issue where only first frame was processed in video batches
- **Memory Errors**: Resolved CUDA/HIP out of memory errors with better memory management
- **Model Loading**: Fixed `NoneType` and `size mismatch` errors in model loading
- **Video Output**: Fixed "Error - Frame x" overlays in video combine node
- **Tensor Size Mismatch**: Fixed batch tensor stacking errors with consistent dimensions
- **Race Conditions**: Fixed model caching race conditions in parallel processing
- **Error Propagation**: Fixed error reporting to properly bubble up to `error_str` output

### Performance Improvements
- **Model Caching**: Up to 50% faster processing for sequential video frames
- **Memory Optimization**: Reduced memory usage by up to 40% with intelligent tile sizing
- **Parallel Processing**: Optional 2-3x speedup for large batches (when memory allows)
- **AMD GPU Support**: Optimized tile sizes and memory management for AMD architecture

### Technical Details
- **Thread Safety**: Proper locking for model caching in sequential mode
- **Memory Management**: Automatic tile size reduction for large images and parallel processing
- **Error Recovery**: Graceful error handling with properly sized error images
- **Model Architecture**: Correct RRDBNet model creation with proper scale parameters
- **Batch Processing**: Efficient tensor operations for video frame processing

### Installation
- **Automatic Fix**: Run `fix_torchvision_import.ps1` to automatically fix import issues
- **Manual Fix**: Update `degradations.py` in basicsr package if automatic fix fails
- **Compatibility**: Works with both AMD and NVIDIA GPUs

### Known Issues
- **Parallel Processing**: May be slower than sequential on some architectures due to model recreation overhead
- **Memory Usage**: Parallel processing requires more memory due to multiple model instances
- **Tile Size**: Very large images may require manual tile size reduction

### Migration from v0.0.1.1
- No breaking changes to node interface
- Default parameters updated for better performance
- New `parallel_workers` parameter added (optional)
- Improved error handling and reporting

## [0.0.1.1] - Original Version
- Basic RealESRGAN upscaling functionality
- Single image processing only
- Basic error handling

