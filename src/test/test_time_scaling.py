"""
Test time scaling functionality for testing mode.

This test verifies that time-based thresholds are properly scaled
when testing_time_scale_factor is configured.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.tracking.EventCentricTracker import EventConfig, EventCentricTracker


def test_manual_time_scaling():
    """Test that manual time scaling factor is applied to all time-based thresholds."""
    
    # Create config with manual scaling factor
    config = EventConfig(
        association_time_ms=400.0,
        ghost_timeout_ms=1000.0,
        max_event_lifetime_ms=10000.0,
        suppression_duration_ms=1500.0,
        min_event_creation_interval_ms=400.0,
        open_to_closing_time_ms=100.0,
        closing_stability_time_ms=150.0,
        closed_stability_time_ms=200.0,
        max_prediction_time_ms=500.0,
        min_gap_duration_for_logging_ms=500.0,
        testing_time_scale_factor=5.0,  # 5x slower processing
        enable_auto_time_scaling=False,  # Disable auto for predictable test
    )
    
    # Create tracker
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    
    # Verify scaled thresholds
    assert tracker._time_scale_factor == 5.0, "Scale factor not set correctly"
    assert tracker._scaled_association_time_ms == 400.0 * 5.0, "association_time_ms not scaled"
    assert tracker._scaled_ghost_timeout_ms == 1000.0 * 5.0, "ghost_timeout_ms not scaled"
    assert tracker._scaled_max_event_lifetime_ms == 10000.0 * 5.0, "max_event_lifetime_ms not scaled"
    assert tracker._scaled_suppression_duration_ms == 1500.0 * 5.0, "suppression_duration_ms not scaled"
    assert tracker._scaled_min_event_creation_interval_ms == 400.0 * 5.0, "min_event_creation_interval_ms not scaled"
    assert tracker._scaled_open_to_closing_time_ms == 100.0 * 5.0, "open_to_closing_time_ms not scaled"
    assert tracker._scaled_closing_stability_time_ms == 150.0 * 5.0, "closing_stability_time_ms not scaled"
    assert tracker._scaled_closed_stability_time_ms == 200.0 * 5.0, "closed_stability_time_ms not scaled"
    assert tracker._scaled_max_prediction_time_ms == 500.0 * 5.0, "max_prediction_time_ms not scaled"
    assert tracker._scaled_min_gap_duration_for_logging_ms == 500.0 * 5.0, "min_gap_duration_for_logging_ms not scaled"
    
    print("✓ Manual time scaling test passed")
    print(f"  Scale factor: {tracker._time_scale_factor}")
    print(f"  Scaled association_time: {tracker._scaled_association_time_ms}ms (base: {config.association_time_ms}ms)")
    print(f"  Scaled ghost_timeout: {tracker._scaled_ghost_timeout_ms}ms (base: {config.ghost_timeout_ms}ms)")


def test_no_scaling():
    """Test that no scaling is applied when factor is 1.0."""
    
    config = EventConfig(
        association_time_ms=400.0,
        ghost_timeout_ms=1000.0,
        testing_time_scale_factor=1.0,  # No scaling
        enable_auto_time_scaling=False,
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    
    # Verify no scaling applied
    assert tracker._time_scale_factor == 1.0
    assert tracker._scaled_association_time_ms == config.association_time_ms
    assert tracker._scaled_ghost_timeout_ms == config.ghost_timeout_ms
    
    print("✓ No scaling test passed")
    print(f"  Scale factor: {tracker._time_scale_factor}")
    print(f"  Thresholds unchanged from base configuration")


def test_scaled_config_propagation():
    """Test that scaled config is properly created for BreadBagEvent instances."""
    
    config = EventConfig(
        association_time_ms=400.0,
        ghost_timeout_ms=1000.0,
        testing_time_scale_factor=3.0,
        enable_auto_time_scaling=False,
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    
    # Get scaled config
    scaled_config = tracker._get_scaled_config()
    
    # Verify scaled config has scaled values
    assert scaled_config.association_time_ms == 400.0 * 3.0
    assert scaled_config.ghost_timeout_ms == 1000.0 * 3.0
    
    # Verify original config unchanged
    assert config.association_time_ms == 400.0
    assert config.ghost_timeout_ms == 1000.0
    
    print("✓ Scaled config propagation test passed")
    print(f"  Original association_time: {config.association_time_ms}ms")
    print(f"  Scaled association_time: {scaled_config.association_time_ms}ms")


def test_auto_scaling_disabled():
    """Test that auto-scaling doesn't activate when disabled."""
    
    config = EventConfig(
        testing_time_scale_factor=2.0,
        enable_auto_time_scaling=False,  # Disabled
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    
    # Simulate frame updates
    import numpy as np
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    for i in range(150):  # More than warmup frames
        timestamp_ms = i * 200.0  # Simulate 5fps
        tracker.update([], timestamp_ms, frame, i)
    
    # Scale factor should remain at initial value
    assert tracker._time_scale_factor == 2.0, "Scale factor changed despite auto-scaling disabled"
    
    print("✓ Auto-scaling disabled test passed")
    print(f"  Scale factor remained at: {tracker._time_scale_factor}")


def test_frame_based_parameters_not_affected():
    """Test that frame-based parameters are not affected by time scaling."""
    
    config = EventConfig(
        commit_idle_frames=25,
        out_of_zone_grace_frames=5,
        testing_time_scale_factor=5.0,
        enable_auto_time_scaling=False,
    )
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    
    # Verify frame-based parameters unchanged
    assert tracker.config.commit_idle_frames == 25, "Frame-based parameter incorrectly scaled"
    assert tracker.config.out_of_zone_grace_frames == 5, "Frame-based parameter incorrectly scaled"
    
    print("✓ Frame-based parameters test passed")
    print(f"  commit_idle_frames: {tracker.config.commit_idle_frames} (unchanged)")
    print(f"  out_of_zone_grace_frames: {tracker.config.out_of_zone_grace_frames} (unchanged)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Time Scaling Functionality")
    print("=" * 60)
    print()
    
    try:
        test_manual_time_scaling()
        print()
        test_no_scaling()
        print()
        test_scaled_config_propagation()
        print()
        test_auto_scaling_disabled()
        print()
        test_frame_based_parameters_not_affected()
        print()
        print("=" * 60)
        print("✓ All tests passed!")
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
