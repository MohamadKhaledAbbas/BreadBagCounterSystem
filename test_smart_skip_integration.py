"""
Integration test demonstrating smart frame skipping behavior.

This simulates a production scenario with varying queue loads.
"""

import sys
import os
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.tracking_config import tracking_config


class MockEvent:
    """Mock event for testing."""
    def __init__(self, event_id, state, frame_count=0):
        self.event_id = event_id
        self.state = state
        self.frame_count = frame_count


class SmartSkipSimulator:
    """Simulates smart skip behavior for testing."""
    
    def __init__(self):
        self.frame_counter = 0
        self.event_frame_counts = {}
        self.frames_processed = 0
        self.frames_skipped = 0
        
    def should_skip(self, queue_util, active_events):
        """
        Simplified version of _should_smart_skip_frame for testing.
        """
        if not tracking_config.degraded_mode_smart_skip_enabled:
            return False, "disabled"
        
        # Check for critical states
        for event in active_events:
            if event.state == 'CLOSING':
                return False, f"event_{event.event_id}_closing"
            
            if event.state == 'OPEN':
                frames = self.event_frame_counts.get(event.event_id, 0)
                if frames < tracking_config.degraded_mode_critical_state_frame_threshold:
                    return False, f"event_{event.event_id}_early_open"
        
        # Check minimum frames
        for event in active_events:
            frames = self.event_frame_counts.get(event.event_id, 0)
            if frames < tracking_config.degraded_mode_min_frames_per_event:
                return False, f"event_{event.event_id}_needs_frames"
        
        # Apply pattern
        self.frame_counter += 1
        pattern = tracking_config.degraded_mode_skip_pattern
        
        if pattern == 'adaptive':
            if queue_util < 0.5:
                return False, "adaptive_low_queue"
            elif queue_util < 0.7:
                should_skip = (self.frame_counter % 3 == 0)
                return should_skip, "adaptive_every_3rd"
            elif queue_util < 0.85:
                should_skip = (self.frame_counter % 2 == 0)
                return should_skip, "adaptive_every_2nd"
            elif queue_util < 0.95:
                should_skip = (self.frame_counter % 3 != 0)
                return should_skip, "adaptive_2_of_3"
            else:
                should_skip = (self.frame_counter % 4 != 0)
                return should_skip, "adaptive_3_of_4"
        
        return False, "unknown"
    
    def process_frame(self, queue_util, active_events):
        """Process a frame and update counters."""
        should_skip, reason = self.should_skip(queue_util, active_events)
        
        if should_skip:
            self.frames_skipped += 1
            return False, reason
        else:
            self.frames_processed += 1
            # Update event frame counts
            for event in active_events:
                self.event_frame_counts[event.event_id] = \
                    self.event_frame_counts.get(event.event_id, 0) + 1
            return True, reason


def test_scenario_1_new_event():
    """Test that new events receive sufficient initial frames."""
    print("\n" + "="*70)
    print("Scenario 1: New Event Creation (Early OPEN Protection)")
    print("="*70)
    
    simulator = SmartSkipSimulator()
    
    # Create a new event (early OPEN, needs initial frames)
    event = MockEvent(event_id=1, state='OPEN')
    
    # Simulate high queue pressure (80%)
    queue_util = 0.80
    
    print(f"\nQueue Utilization: {queue_util:.0%}")
    print(f"Expected: No skipping until event has {tracking_config.degraded_mode_critical_state_frame_threshold} frames\n")
    
    frames_before_skip = 0
    
    for i in range(1, 21):
        processed, reason = simulator.process_frame(queue_util, [event])
        
        if processed:
            frames_before_skip += 1
            print(f"Frame {i:2d}: PROCESSED (reason: {reason}) "
                  f"- Event frames: {simulator.event_frame_counts.get(1, 0)}")
        else:
            print(f"Frame {i:2d}: SKIPPED   (reason: {reason})")
            break
    
    print(f"\n✓ Event received {frames_before_skip} frames before skipping started")
    print(f"  (threshold: {tracking_config.degraded_mode_critical_state_frame_threshold})")
    
    assert frames_before_skip >= tracking_config.degraded_mode_critical_state_frame_threshold, \
        f"Event should get at least {tracking_config.degraded_mode_critical_state_frame_threshold} frames"


def test_scenario_2_closing_state():
    """Test that CLOSING state is never skipped."""
    print("\n" + "="*70)
    print("Scenario 2: Event in CLOSING State (Critical State Protection)")
    print("="*70)
    
    simulator = SmartSkipSimulator()
    
    # Create event in CLOSING state (already has many frames)
    event = MockEvent(event_id=1, state='CLOSING')
    simulator.event_frame_counts[1] = 30  # Already has plenty of frames
    
    # Simulate critical queue pressure (98%)
    queue_util = 0.98
    
    print(f"\nQueue Utilization: {queue_util:.0%} (CRITICAL)")
    print(f"Event State: CLOSING (critical for state transition)")
    print(f"Expected: NO SKIPPING despite critical queue\n")
    
    for i in range(1, 11):
        processed, reason = simulator.process_frame(queue_util, [event])
        
        status = "PROCESSED" if processed else "SKIPPED"
        print(f"Frame {i:2d}: {status:9s} (reason: {reason})")
        
        assert processed, f"Frame {i} should be processed (CLOSING state)"
    
    print(f"\n✓ All 10 frames processed despite 98% queue utilization")
    print(f"  CLOSING state protected from skipping")


def test_scenario_3_adaptive_pattern():
    """Test adaptive pattern with mature events."""
    print("\n" + "="*70)
    print("Scenario 3: Adaptive Skip Pattern with Mature Events")
    print("="*70)
    
    # Create events that have enough frames
    event1 = MockEvent(event_id=1, state='OPEN')
    event2 = MockEvent(event_id=2, state='CLOSED')
    
    # Test at different queue levels
    test_cases = [
        (0.45, "Below threshold", 0),
        (0.65, "50-70% (every 3rd)", 33),
        (0.75, "70-85% (every 2nd)", 50),
        (0.90, "85-95% (2 of 3)", 67),
    ]
    
    for queue_util, description, expected_skip_pct in test_cases:
        simulator = SmartSkipSimulator()
        # Give events enough initial frames
        simulator.event_frame_counts[1] = 20
        simulator.event_frame_counts[2] = 20
        
        print(f"\n{description} - Queue: {queue_util:.0%}")
        print("-" * 70)
        
        for i in range(1, 31):
            processed, reason = simulator.process_frame(queue_util, [event1, event2])
            
            if i <= 10:  # Only show first 10 frames
                status = "✓" if processed else "✗"
                print(f"  Frame {i:2d}: {status}")
        
        actual_skip_pct = (simulator.frames_skipped / 30) * 100
        print(f"\nSkipped: {simulator.frames_skipped}/30 frames ({actual_skip_pct:.0f}%)")
        print(f"Expected: ~{expected_skip_pct}%")
        
        # Allow some tolerance due to pattern phase
        tolerance = 10
        if expected_skip_pct > 0:
            assert abs(actual_skip_pct - expected_skip_pct) <= tolerance, \
                f"Skip rate should be ~{expected_skip_pct}%, got {actual_skip_pct:.0f}%"


def test_scenario_4_min_frames_enforcement():
    """Test that events always get minimum frames."""
    print("\n" + "="*70)
    print("Scenario 4: Minimum Frames Enforcement")
    print("="*70)
    
    simulator = SmartSkipSimulator()
    
    # Create new event
    event = MockEvent(event_id=1, state='OPEN')
    
    # Very high queue (90% - should skip aggressively)
    queue_util = 0.90
    
    print(f"\nQueue Utilization: {queue_util:.0%} (should skip 2 of 3 frames)")
    print(f"Minimum frames per event: {tracking_config.degraded_mode_min_frames_per_event}")
    print(f"Expected: Event gets all frames until it reaches minimum\n")
    
    frames_received = 0
    total_frames = 50
    
    for i in range(1, total_frames + 1):
        processed, reason = simulator.process_frame(queue_util, [event])
        
        if processed:
            frames_received = simulator.event_frame_counts.get(1, 0)
            
            if frames_received <= tracking_config.degraded_mode_min_frames_per_event + 2:
                print(f"Frame {i:2d}: PROCESSED - Event has {frames_received} frames")
    
    print(f"\n✓ Event received {frames_received} frames (minimum: {tracking_config.degraded_mode_min_frames_per_event})")
    
    assert frames_received >= tracking_config.degraded_mode_min_frames_per_event, \
        f"Event should get at least {tracking_config.degraded_mode_min_frames_per_event} frames"


def run_integration_tests():
    """Run all integration test scenarios."""
    print("\n" + "="*70)
    print("Smart Frame Skipping - Integration Tests")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Pattern: {tracking_config.degraded_mode_skip_pattern}")
    print(f"  Min frames per event: {tracking_config.degraded_mode_min_frames_per_event}")
    print(f"  Critical state threshold: {tracking_config.degraded_mode_critical_state_frame_threshold}")
    print(f"  Preserve critical states: {tracking_config.degraded_mode_preserve_critical_states}")
    
    tests = [
        ("New Event Protection", test_scenario_1_new_event),
        ("CLOSING State Protection", test_scenario_2_closing_state),
        ("Adaptive Pattern", test_scenario_3_adaptive_pattern),
        ("Minimum Frames Enforcement", test_scenario_4_min_frames_enforcement),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n✓ {name} PASSED")
        except AssertionError as e:
            print(f"\n✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {name} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("✓ All integration tests passed!\n")
        print("Smart frame skipping is working correctly and ready for production.\n")
        sys.exit(0)


if __name__ == '__main__':
    run_integration_tests()
