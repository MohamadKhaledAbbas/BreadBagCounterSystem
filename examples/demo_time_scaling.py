#!/usr/bin/env python3
"""
Demo: Time Scaling for Testing Mode

This script demonstrates how time scaling works to maintain equivalent
behavior between slow testing environments and fast production environments.

It simulates:
1. Production (25fps, real-time)
2. Testing without scaling (5fps, incorrect behavior)
3. Testing with scaling (5fps, correct behavior)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tracking.EventCentricTracker import EventConfig, EventCentricTracker
import numpy as np


def simulate_scenario(name, config, fps, num_frames=50):
    """Simulate a tracking scenario with given config and FPS."""
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"{'='*60}")
    print(f"FPS: {fps}, Frame time: {1000/fps:.1f}ms")
    print(f"Scale factor: {config.testing_time_scale_factor}")
    print()
    
    tracker = EventCentricTracker(config=config, open_class_id=1, closed_class_id=0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Simulate detection appearing and disappearing
    results = []
    
    # Detection appears at frame 0
    detection = {
        'box': [500, 300, 600, 400],
        'class_id': 1,  # Open bag
        'conf': 0.8
    }
    
    # Process frames
    for frame_idx in range(num_frames):
        timestamp_ms = frame_idx * (1000.0 / fps)
        
        # Detection present for first 10 frames, then disappears
        if frame_idx < 10:
            detections = [detection]
            status = "DETECTED"
        else:
            detections = []
            status = "LOST"
        
        ready_events = tracker.update(detections, timestamp_ms, frame, frame_idx)
        
        active_count = len(tracker.active_events)
        committed_count = len(ready_events)
        
        if frame_idx % 10 == 0 or committed_count > 0 or (frame_idx == 9):
            print(f"Frame {frame_idx:3d} ({timestamp_ms:7.0f}ms): "
                  f"{status:8s} | Active: {active_count}, Committed: {committed_count}")
        
        if committed_count > 0:
            results.append({
                'frame': frame_idx,
                'timestamp_ms': timestamp_ms,
                'committed': committed_count
            })
    
    print()
    print("Results:")
    if results:
        for r in results:
            print(f"  Event committed at frame {r['frame']} ({r['timestamp_ms']:.0f}ms)")
    else:
        print("  No events committed (expired too quickly or still active)")
    
    stats = tracker.get_tracker_stats()
    print(f"\nFinal Stats:")
    print(f"  Created: {stats['events_created']}")
    print(f"  Committed: {stats['events_committed']}")
    print(f"  Expired: {stats['events_expired']}")
    print(f"  Active: {stats['active_events']}")
    
    return results


def main():
    print("=" * 60)
    print("Time Scaling Demo")
    print("=" * 60)
    print()
    print("This demo shows how time scaling ensures equivalent behavior")
    print("between production (fast) and testing (slow) environments.")
    print()
    
    # Scenario 1: Production - 25fps, no scaling needed
    config_production = EventConfig(
        ghost_timeout_ms=1000.0,  # 1 second ghost timeout
        testing_time_scale_factor=1.0,
        enable_auto_time_scaling=False,
    )
    results_prod = simulate_scenario(
        "Production (25fps, real-time)",
        config_production,
        fps=25,
        num_frames=50
    )
    
    # Scenario 2: Testing without scaling - 5fps, timeouts expire too quickly
    config_testing_no_scale = EventConfig(
        ghost_timeout_ms=1000.0,  # Same 1 second ghost timeout
        testing_time_scale_factor=1.0,  # No scaling!
        enable_auto_time_scaling=False,
    )
    results_test_no_scale = simulate_scenario(
        "Testing WITHOUT scaling (5fps, incorrect)",
        config_testing_no_scale,
        fps=5,
        num_frames=50
    )
    
    # Scenario 3: Testing with scaling - 5fps, timeouts scaled appropriately
    config_testing_with_scale = EventConfig(
        ghost_timeout_ms=1000.0,  # Base 1 second ghost timeout
        testing_time_scale_factor=5.0,  # Scale by 5x for 5fps (vs 25fps target)
        enable_auto_time_scaling=False,
    )
    results_test_with_scale = simulate_scenario(
        "Testing WITH scaling (5fps, correct)",
        config_testing_with_scale,
        fps=5,
        num_frames=50
    )
    
    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("Ghost timeout: 1000ms (1 second)")
    print("Detection lost after frame 9")
    print()
    print(f"Production (25fps):     {len(results_prod)} events committed")
    print(f"Testing no scale (5fps): {len(results_test_no_scale)} events committed")
    print(f"Testing w/ scale (5fps): {len(results_test_with_scale)} events committed")
    print()
    
    if len(results_prod) == len(results_test_with_scale):
        print("✓ With scaling: Testing matches production behavior!")
    else:
        print("⚠ Results differ - check configuration")
    
    if len(results_test_no_scale) != len(results_prod):
        print("⚠ Without scaling: Testing does NOT match production")
    
    print()
    print("Key Insight:")
    print("  - At 25fps, 1000ms ghost timeout = 25 frames")
    print("  - At 5fps WITHOUT scaling, 1000ms = only 5 frames (expires too early!)")
    print("  - At 5fps WITH 5x scaling, 5000ms = 25 frames (equivalent!)")


if __name__ == "__main__":
    main()
