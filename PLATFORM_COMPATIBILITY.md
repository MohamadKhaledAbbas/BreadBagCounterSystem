# Platform Compatibility Guide

This document describes how the BreadBagCounterSystem supports both RDK boards and Windows/other platforms.

## Overview

The codebase uses platform detection to automatically select the appropriate implementations:

- **RDK Platform**: Uses BPU-optimized models with ROS2 for IPC
- **Windows/Linux (non-RDK)**: Uses Ultralytics YOLO models with OpenCV

## Platform Detection

Platform detection is handled by `src/utils/platform.py`:

```python
from src.utils.platform import IS_RDK, IS_WINDOWS

if IS_RDK:
    # Use RDK-specific code
else:
    # Use Windows/generic code
```

Detection logic:
- **RDK**: Checks for `hobot_dnn` or `hobot_dnn_rdkx5` libraries
- **Windows**: Checks `sys.platform == 'win32'`

## Component Implementations

### Detection

| Platform | Implementation | Model Format |
|----------|---------------|--------------|
| RDK | `BpuDetector` | `.bin` (BPU-optimized) |
| Windows | `UltralyticsDetector` | `.pt`, `.onnx`, `.engine` |

### Classification

| Platform | Implementation | Model Format |
|----------|---------------|--------------|
| RDK | `BpuClassifier` | `.bin` (BPU-optimized) |
| Windows | `UltralyticsClassifier` | `.pt`, `.onnx` |

### Frame Sources

| Platform | Source Type | Implementation |
|----------|------------|----------------|
| RDK | ROS2 | `Ros2FrameServer` (subscribes to `/nv12_images`) |
| RDK | OpenCV | `OpenCVFrameSource` |
| Windows | OpenCV | `OpenCVFrameSource` |

**Note**: ROS2 frame sources are only available on RDK platform.

### IPC/Publishing

| Platform | Implementation | Purpose |
|----------|---------------|---------|
| RDK | ROS2 Publishers/Subscribers | Inter-process communication for UI |
| Windows | Stub implementations | No-op stubs for compatibility |

## Configuration

Model paths are automatically selected based on platform in `src/config/settings.py`:

```python
# RDK uses .bin models
detection_model = "data/model/detect_yolo_small_v2_bayese_640x640_nv12.bin"
classification_model = "data/model/classify_yolo_small_v4_bayese_224x224_nv12.bin"

# Windows uses .pt models
detection_model = "data/model/detect_yolo_small_v3.pt"
classification_model = "data/model/classify_yolo_small_v4.pt"
```

Override with environment variables:
```bash
export DETECTION_MODEL="path/to/custom/model.pt"
export CLASS_MODEL="path/to/custom/classifier.pt"
```

## Running the Application

### On RDK

```bash
# Uses BPU models, ROS2 frame source, ROS2 IPC
python3 main.py
```

To view the UI (requires ROS2):
```bash
python3 main_ui.py
```

### On Windows/Linux

```bash
# Uses Ultralytics models, OpenCV frame source
python3 main.py
```

**Note**: `main_ui.py` is not supported on Windows (requires ROS2). The main application displays frames directly when `is_publishing` is enabled in the database config.

## Adding New Platforms

To add support for a new platform:

1. **Update platform detection** in `src/utils/platform.py`:
   ```python
   def is_new_platform():
       # Your detection logic
       return False
   
   IS_NEW_PLATFORM = is_new_platform()
   ```

2. **Add conditional imports** where needed:
   ```python
   from src.utils.platform import IS_NEW_PLATFORM
   
   if IS_NEW_PLATFORM:
       # Import platform-specific modules
   ```

3. **Update configuration** in `src/config/settings.py` to select appropriate model paths

4. **Add platform-specific implementations** if needed (detector, classifier, frame source)

## Dependencies

### RDK Platform
- `hobot_dnn` or `hobot_dnn_rdkx5` - BPU inference
- `rclpy` - ROS2 Python client
- `hbm_img_msgs` - Custom ROS2 messages for image transport
- `cv_bridge` - ROS2/OpenCV bridge
- `opencv-python` - Image processing
- `numpy` - Numerical operations

### Windows/Linux
- `ultralytics` - YOLO inference
- `opencv-python` - Image processing and frame capture
- `numpy` - Numerical operations
- `torch` - PyTorch (for .pt models)

### Common
- `imagehash` - Perceptual hashing
- `scipy` - Scientific computing
- `pillow` - Image processing

## Troubleshooting

### Import Errors on Windows

If you see `ModuleNotFoundError` for ROS2 modules on Windows:
- This is expected! The code handles this gracefully with stubs
- Ensure you're using the latest version with platform detection

### Model Not Found

If you get model file errors:
- Check the model path in config or environment variables
- Ensure the model format matches the platform (.bin for RDK, .pt/.onnx for Windows)
- Models are gitignored - you need to download or train them separately

### ROS2 Frame Source Error on Windows

```
ValueError: ROS2 frame source only available on RDK platform
```

This is expected on Windows. Use OpenCV frame source instead:
- For video files: The app automatically uses OpenCV in development mode
- For cameras: OpenCV frame source works on all platforms

## Architecture

```
main.py
  ├─ Platform Detection (src/utils/platform.py)
  │
  ├─ Detector (platform-specific)
  │  ├─ RDK: BpuDetector
  │  └─ Windows: UltralyticsDetector
  │
  ├─ Classifier (platform-specific)
  │  ├─ RDK: BpuClassifier
  │  └─ Windows: UltralyticsClassifier
  │
  └─ BagCounterApp
     ├─ Frame Source (configurable)
     │  ├─ ROS2 (RDK only)
     │  └─ OpenCV (all platforms)
     │
     └─ IPC Publisher (platform-specific)
        ├─ RDK: ROS2 Publisher
        └─ Windows: Stub (no-op)
```

## Testing

Run the platform compatibility test suite:

```bash
python3 tests/test_platform_compatibility.py
```

This validates:
- ✓ Platform detection
- ✓ Conditional imports work without ROS2
- ✓ Configuration selects correct model paths
- ✓ Detector/classifier selection
- ✓ Frame source factory behavior
- ✓ Interface compatibility

## Best Practices

1. **Always use platform detection** when adding platform-specific code
2. **Provide stub implementations** for missing features on other platforms
3. **Keep interfaces consistent** across implementations
4. **Test on both platforms** when making changes
5. **Document platform-specific behavior** in docstrings
6. **Use environment variables** for configuration overrides

## Future Enhancements

Potential improvements:
- [ ] Add MacOS-specific optimizations
- [ ] Support for TensorRT on Windows
- [ ] Cross-platform UI using Qt or similar
- [ ] Automated model conversion between formats
- [ ] Docker containers for consistent environments
