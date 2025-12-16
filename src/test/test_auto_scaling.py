"""
Test auto-scaling functionality for testing mode.

This test verifies that auto-scaling correctly measures processing speed
and calculates appropriate time scale factor.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.tracking.EventCentricTracker import EventConfig, EventCentricTracker
import numpy as np


def test_auto_scaling_activation():
    """Test that auto-scaling activates after warmup period."""
    
    config = EventConfig(
        testing_time_scale_factor=1.0,  # Initial value
        enable_auto_time_scaling=True,  # Enable auto-scaling
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    
    # Simulate frame updates with consistent 200ms intervals (5fps effective)
    # Target is 40ms (25fps), so scale factor should be 200/40 = 5.0
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("  Simulating frames with 200ms intervals (5fps effective)...")
    print("  Target: 40ms per frame (25fps)")
    
    initial_scale = tracker._time_scale_factor
    print(f"  Initial scale factor: {initial_scale}")
    
    # Process warmup frames (100 frames)
    for i in range(102):
        timestamp_ms = i * 200.0  # 200ms per frame = 5fps
        tracker.update([], timestamp_ms, frame, i)
    
    final_scale = tracker._time_scale_factor
    print(f"  Final scale factor: {final_scale}")
    
    # Scale factor should have been calculated and applied
    assert final_scale > initial_scale, f"Auto-scaling didn't activate: {final_scale} <= {initial_scale}"
    assert 4.5 <= final_scale <= 5.5, f"Scale factor {final_scale} outside expected range [4.5, 5.5]"
    
    print(f"✓ Auto-scaling activation test passed")
    print(f"  Calculated scale factor: {final_scale:.2f}x")


def test_auto_scaling_near_realtime():
    """Test that auto-scaling doesn't activate when processing is near real-time."""
    
    config = EventConfig(
        testing_time_scale_factor=1.0,
        enable_auto_time_scaling=True,
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("  Simulating frames with 45ms intervals (22fps, close to target 25fps)...")
    
    # Process frames at near-target speed (45ms vs 40ms target = 1.125x)
    for i in range(102):
        timestamp_ms = i * 45.0
        tracker.update([], timestamp_ms, frame, i)
    
    # Scale factor should remain 1.0 (< 1.2x threshold)
    assert tracker._time_scale_factor == 1.0, f"Auto-scaling activated unnecessarily: {tracker._time_scale_factor}"
    
    print(f"✓ Near real-time test passed")
    print(f"  Scale factor remained at: {tracker._time_scale_factor} (no scaling needed)")


def test_auto_scaling_with_variable_intervals():
    """Test auto-scaling with variable frame intervals (more realistic)."""
    
    config = EventConfig(
        testing_time_scale_factor=1.0,
        enable_auto_time_scaling=True,
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("  Simulating frames with variable intervals averaging 150ms...")
    
    # Simulate variable processing times averaging 150ms (6.67fps)
    import random
    random.seed(42)
    
    timestamp_ms = 0.0
    for i in range(102):
        # Variable interval: 130-170ms, avg=150ms
        interval = 130.0 + random.random() * 40.0
        timestamp_ms += interval
        tracker.update([], timestamp_ms, frame, i)
    
    # Should calculate scale factor around 150/40 = 3.75
    assert tracker._time_scale_factor > 1.0, "Auto-scaling didn't activate"
    assert 3.0 <= tracker._time_scale_factor <= 4.5, f"Scale factor {tracker._time_scale_factor} outside expected range"
    
    print(f"✓ Variable intervals test passed")
    print(f"  Calculated scale factor: {tracker._time_scale_factor:.2f}x")


def test_scaled_thresholds_after_auto_scaling():
    """Test that thresholds are properly scaled after auto-scaling activates."""
    
    config = EventConfig(
        association_time_ms=400.0,
        ghost_timeout_ms=1000.0,
        testing_time_scale_factor=1.0,
        enable_auto_time_scaling=True,
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Before auto-scaling
    initial_assoc = tracker._scaled_association_time_ms
    initial_ghost = tracker._scaled_ghost_timeout_ms
    
    print(f"  Before auto-scaling:")
    print(f"    association_time: {initial_assoc}ms")
    print(f"    ghost_timeout: {initial_ghost}ms")
    
    # Trigger auto-scaling with 200ms intervals
    for i in range(102):
        timestamp_ms = i * 200.0
        tracker.update([], timestamp_ms, frame, i)
    
    # After auto-scaling
    final_assoc = tracker._scaled_association_time_ms
    final_ghost = tracker._scaled_ghost_timeout_ms
    
    print(f"  After auto-scaling (factor={tracker._time_scale_factor:.2f}):")
    print(f"    association_time: {final_assoc}ms")
    print(f"    ghost_timeout: {final_ghost}ms")
    
    # Thresholds should be scaled
    assert final_assoc > initial_assoc, "Association time not scaled"
    assert final_ghost > initial_ghost, "Ghost timeout not scaled"
    
    # Check scaling is approximately correct (5.0x for 200ms intervals)
    expected_assoc = 400.0 * 5.0
    expected_ghost = 1000.0 * 5.0
    assert abs(final_assoc - expected_assoc) < 100, f"Association time scaling incorrect: {final_assoc} vs {expected_assoc}"
    assert abs(final_ghost - expected_ghost) < 200, f"Ghost timeout scaling incorrect: {final_ghost} vs {expected_ghost}"
    
    print(f"✓ Scaled thresholds after auto-scaling test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Auto-Scaling Functionality")
    print("=" * 60)
    print()
    
    try:
        test_auto_scaling_activation()
        print()
        test_auto_scaling_near_realtime()
        print()
        test_auto_scaling_with_variable_intervals()
        print()
        test_scaled_thresholds_after_auto_scaling()
        print()
        print("=" * 60)
        print("✓ All auto-scaling tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
