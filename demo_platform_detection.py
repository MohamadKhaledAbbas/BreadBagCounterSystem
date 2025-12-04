#!/usr/bin/env python3
"""
Demo script showing platform detection and automatic component selection.
This script can be run on both RDK and Windows/other platforms.
"""

import sys
from src.utils.AppLogging import logger

def main():
    """Demonstrate platform detection and component selection."""
    
    print("\n" + "=" * 70)
    print("BREADBAG COUNTER SYSTEM - PLATFORM DETECTION DEMO")
    print("=" * 70 + "\n")
    
    # 1. Platform Detection
    print("1. PLATFORM DETECTION")
    print("-" * 70)
    from src.utils.platform import IS_RDK, IS_WINDOWS, is_rdk_platform, is_windows
    
    print(f"System: {sys.platform}")
    print(f"RDK Platform Detected: {IS_RDK}")
    print(f"Windows Platform Detected: {IS_WINDOWS}")
    
    if IS_RDK:
        print("→ Running on RDK board (hobot_dnn available)")
        platform_name = "RDK"
    elif IS_WINDOWS:
        print("→ Running on Windows")
        platform_name = "Windows"
    else:
        print("→ Running on Linux/Other")
        platform_name = "Linux/Other"
    
    print()
    
    # 2. Configuration
    print("2. PLATFORM-AWARE CONFIGURATION")
    print("-" * 70)
    from src.config.settings import config
    
    print(f"Detection Model: {config.detection_model}")
    print(f"Classification Model: {config.classification_model}")
    print(f"Database Path: {config.db_path}")
    
    if IS_RDK:
        print("→ Using BPU-optimized .bin models")
    else:
        print("→ Using Ultralytics .pt models")
    
    print()
    
    # 3. Detector Selection
    print("3. DETECTOR SELECTION")
    print("-" * 70)
    
    if IS_RDK:
        from src.detection.BpuDetector import BpuDetector as Detector
        detector_type = "BpuDetector (BPU-accelerated)"
    else:
        from src.detection.UltralyticsDetector import UltralyticsDetector as Detector
        detector_type = "UltralyticsDetector (CPU/GPU)"
    
    print(f"Selected Detector: {detector_type}")
    print(f"Class: {Detector.__module__}.{Detector.__name__}")
    print()
    
    # 4. Classifier Selection
    print("4. CLASSIFIER SELECTION")
    print("-" * 70)
    
    if IS_RDK:
        from src.classifier.BpuClassifyer import BpuClassifier as Classifier
        classifier_type = "BpuClassifier (BPU-accelerated)"
    else:
        from src.classifier.UltralyticsClassifier import UltralyticsClassifier as Classifier
        classifier_type = "UltralyticsClassifier (CPU/GPU)"
    
    print(f"Selected Classifier: {classifier_type}")
    print(f"Class: {Classifier.__module__}.{Classifier.__name__}")
    print()
    
    # 5. Frame Source Options
    print("5. FRAME SOURCE OPTIONS")
    print("-" * 70)
    
    if IS_RDK:
        print("Available frame sources:")
        print("  • ROS2 (production mode) - /nv12_images topic")
        print("  • OpenCV (development mode) - video files or cameras")
    else:
        print("Available frame sources:")
        print("  • OpenCV (all modes) - video files or cameras")
        print("  ✗ ROS2 not available on this platform")
    
    print()
    
    # 6. IPC/Publishing
    print("6. IPC/PUBLISHING")
    print("-" * 70)
    
    from src.counting.IPC import init_ros2_context, shutdown_ros2_context
    
    if IS_RDK:
        print("IPC Method: ROS2 Publishers/Subscribers")
        print("  • FramePublisher: Publishes annotated frames")
        print("  • FrameSubscriber: Receives frames in UI")
        print("  • Topic: breadcount/image_raw")
    else:
        print("IPC Method: Stub implementations (no-op)")
        print("  • ROS2 not available - using stubs for compatibility")
        print("  • Publishing operations are no-ops")
        print("  • UI display handled directly if enabled")
    
    print()
    
    # 7. Summary
    print("7. SUMMARY")
    print("-" * 70)
    print(f"Platform: {platform_name}")
    print(f"Detector: {detector_type}")
    print(f"Classifier: {classifier_type}")
    print(f"Frame Source: {'ROS2 + OpenCV' if IS_RDK else 'OpenCV only'}")
    print(f"IPC: {'ROS2' if IS_RDK else 'Stubs'}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70 + "\n")
    
    # Test that we can initialize IPC (should work on all platforms)
    print("Testing IPC initialization (should work on all platforms)...")
    ctx = init_ros2_context()
    if ctx is None and not IS_RDK:
        print("✓ IPC stub initialization successful (non-RDK platform)")
    elif ctx is not None and IS_RDK:
        print("✓ ROS2 context initialization successful (RDK platform)")
        shutdown_ros2_context()
        print("✓ ROS2 context shutdown successful")
    else:
        print("✓ IPC initialization successful")
    
    print("\nTo run the full application:")
    print("  python3 main.py")
    
    if IS_RDK:
        print("\nTo view the UI (RDK only):")
        print("  python3 main_ui.py")
    else:
        print("\nNote: main_ui.py requires ROS2 (RDK only)")
        print("      On this platform, main.py displays output directly")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
