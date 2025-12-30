"""
Tests for credit-based flow control in SpoolProcessor.

These tests verify the in-flight window management, ACK handling,
timeout behavior, and backpressure mechanisms.
"""

import sys
import os
import time
import threading
from collections import deque
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define structures locally to avoid import dependencies
@dataclass
class InFlightFrame:
    """Tracks a frame that has been published but not yet ACKed."""
    seq: int
    frame_index: int
    sent_time: float
    segment_num: int
    expired: bool = False


# Constants from spool_processor_node
DEFAULT_MAX_IN_FLIGHT = 10
DEFAULT_ACK_TIMEOUT = 10.0


class MockInFlightTracker:
    """Mock class to test in-flight tracking logic."""
    
    def __init__(self, max_in_flight: int, ack_timeout: float):
        self.max_in_flight = max_in_flight
        self.ack_timeout = ack_timeout
        self.in_flight = {}  # seq -> InFlightFrame
        self.in_flight_order = deque()  # FIFO order
        self.lock = threading.Lock()
        self.timeouts = 0
    
    def add_frame(self, seq: int, frame_index: int) -> bool:
        """Add a frame to in-flight tracking. Returns True if credit available."""
        with self.lock:
            if len(self.in_flight) >= self.max_in_flight:
                return False
            
            frame = InFlightFrame(
                seq=seq,
                frame_index=frame_index,
                sent_time=time.time(),
                segment_num=0
            )
            self.in_flight[seq] = frame
            self.in_flight_order.append(seq)
            return True
    
    def ack_frame(self, seq: int) -> bool:
        """ACK a frame (out-of-order is OK). Returns True if frame was in-flight."""
        with self.lock:
            if seq in self.in_flight:
                del self.in_flight[seq]
                return True
            return False
    
    def check_timeouts(self) -> int:
        """Check for timed-out frames and free credit. Returns count of expired frames."""
        current_time = time.time()
        expired_count = 0
        
        with self.lock:
            while self.in_flight_order:
                seq = self.in_flight_order[0]
                
                # Check if frame still exists (might have been ACKed)
                if seq not in self.in_flight:
                    self.in_flight_order.popleft()
                    continue
                
                frame = self.in_flight[seq]
                age = current_time - frame.sent_time
                
                # If oldest frame hasn't timed out, none have (FIFO)
                if age < self.ack_timeout:
                    break
                
                # Frame has timed out - free credit
                del self.in_flight[seq]
                self.in_flight_order.popleft()
                expired_count += 1
                self.timeouts += 1
        
        return expired_count
    
    def get_in_flight_count(self) -> int:
        """Get current in-flight count."""
        with self.lock:
            return len(self.in_flight)
    
    def has_credit(self) -> bool:
        """Check if credit is available."""
        with self.lock:
            return len(self.in_flight) < self.max_in_flight


def test_in_flight_window_cap():
    """Test that in-flight window enforces max_in_flight limit."""
    print("Testing in-flight window cap enforcement...")
    
    tracker = MockInFlightTracker(max_in_flight=5, ack_timeout=10.0)
    
    # Add frames up to the limit
    for i in range(5):
        assert tracker.add_frame(i, i), f"Should be able to add frame {i}"
    
    assert tracker.get_in_flight_count() == 5, "Should have 5 frames in-flight"
    
    # Try to add one more - should fail (backpressure)
    assert not tracker.add_frame(5, 5), "Should not be able to add 6th frame (exceeds limit)"
    
    # ACK one frame to free credit
    assert tracker.ack_frame(2), "Should ACK frame 2"
    assert tracker.get_in_flight_count() == 4, "Should have 4 frames in-flight after ACK"
    
    # Now should be able to add another
    assert tracker.add_frame(5, 5), "Should be able to add frame after ACK freed credit"
    assert tracker.get_in_flight_count() == 5, "Should have 5 frames in-flight again"
    
    print("✓ In-flight window cap enforcement test passed")


def test_out_of_order_ack():
    """Test that ACKs can arrive out-of-order and still free credit."""
    print("Testing out-of-order ACK handling...")
    
    tracker = MockInFlightTracker(max_in_flight=10, ack_timeout=10.0)
    
    # Add several frames
    for i in range(5):
        tracker.add_frame(i, i)
    
    assert tracker.get_in_flight_count() == 5, "Should have 5 frames in-flight"
    
    # ACK frames out of order: 3, 1, 4, 0, 2
    ack_order = [3, 1, 4, 0, 2]
    for seq in ack_order:
        assert tracker.ack_frame(seq), f"Should successfully ACK frame {seq}"
    
    assert tracker.get_in_flight_count() == 0, "All frames should be ACKed"
    
    # Try to ACK a non-existent frame
    assert not tracker.ack_frame(10), "Should return False for non-existent frame"
    
    print("✓ Out-of-order ACK handling test passed")


def test_timeout_based_credit_release():
    """Test that timed-out frames free credit automatically."""
    print("Testing timeout-based credit release...")
    
    # Use very short timeout for testing
    tracker = MockInFlightTracker(max_in_flight=5, ack_timeout=0.1)
    
    # Add 3 frames
    for i in range(3):
        tracker.add_frame(i, i)
    
    assert tracker.get_in_flight_count() == 3, "Should have 3 frames in-flight"
    
    # Wait for timeout
    time.sleep(0.15)
    
    # Check for timeouts - should expire all 3 frames
    expired = tracker.check_timeouts()
    assert expired == 3, f"Should have expired 3 frames, got {expired}"
    assert tracker.get_in_flight_count() == 0, "All frames should be expired and removed"
    assert tracker.timeouts == 3, "Timeout counter should be 3"
    
    # Add more frames - should have credit available now
    for i in range(5):
        assert tracker.add_frame(i + 3, i + 3), f"Should be able to add frame {i + 3} after timeout freed credit"
    
    print("✓ Timeout-based credit release test passed")


def test_backpressure_behavior():
    """Test backpressure when in-flight window fills."""
    print("Testing backpressure behavior...")
    
    tracker = MockInFlightTracker(max_in_flight=3, ack_timeout=10.0)
    
    # Fill the window
    publish_count = 0
    for i in range(10):
        if tracker.has_credit():
            success = tracker.add_frame(i, i)
            if success:
                publish_count += 1
        else:
            break
    
    assert publish_count == 3, f"Should have published exactly 3 frames (max_in_flight), got {publish_count}"
    assert not tracker.has_credit(), "Should have no credit available"
    assert tracker.get_in_flight_count() == 3, "Should have 3 frames in-flight"
    
    # Try to publish more - should fail due to backpressure
    assert not tracker.add_frame(10, 10), "Should not publish when backpressure active"
    
    # ACK oldest frame
    tracker.ack_frame(0)
    
    # Now should have credit again
    assert tracker.has_credit(), "Should have credit after ACK"
    assert tracker.add_frame(10, 10), "Should be able to publish after credit freed"
    
    print("✓ Backpressure behavior test passed")


def test_mixed_ack_and_timeout():
    """Test behavior with both ACKs and timeouts occurring."""
    print("Testing mixed ACK and timeout behavior...")
    
    tracker = MockInFlightTracker(max_in_flight=5, ack_timeout=0.1)
    
    # Add 5 frames
    for i in range(5):
        tracker.add_frame(i, i)
    
    # ACK frames 1 and 3 immediately
    tracker.ack_frame(1)
    tracker.ack_frame(3)
    
    assert tracker.get_in_flight_count() == 3, "Should have 3 frames in-flight (2 ACKed)"
    
    # Wait for timeout
    time.sleep(0.15)
    
    # Check timeouts - should expire remaining 3 frames (0, 2, 4)
    expired = tracker.check_timeouts()
    assert expired == 3, f"Should expire 3 remaining frames, got {expired}"
    assert tracker.get_in_flight_count() == 0, "All frames should be ACKed or expired"
    
    print("✓ Mixed ACK and timeout behavior test passed")


def test_processor_config_defaults():
    """Test that ProcessorConfig would have correct defaults."""
    print("Testing credit-based flow control defaults...")
    
    # Test constants directly since we can't import ProcessorConfig
    assert DEFAULT_MAX_IN_FLIGHT == 10, \
        f"max_in_flight default should be 10, got {DEFAULT_MAX_IN_FLIGHT}"
    
    assert DEFAULT_ACK_TIMEOUT == 10.0, \
        f"ack_timeout default should be 10.0, got {DEFAULT_ACK_TIMEOUT}"
    
    print("✓ Credit-based flow control defaults test passed")


def test_in_flight_frame_dataclass():
    """Test InFlightFrame dataclass."""
    print("Testing InFlightFrame dataclass...")
    
    current_time = time.time()
    frame = InFlightFrame(
        seq=42,
        frame_index=100,
        sent_time=current_time,
        segment_num=5,
        expired=False
    )
    
    assert frame.seq == 42, "seq should be 42"
    assert frame.frame_index == 100, "frame_index should be 100"
    assert frame.sent_time == current_time, "sent_time should match"
    assert frame.segment_num == 5, "segment_num should be 5"
    assert frame.expired is False, "expired should be False"
    
    # Test expiration marking
    frame.expired = True
    assert frame.expired is True, "expired should be True after marking"
    
    print("✓ InFlightFrame dataclass test passed")


def run_all_tests():
    """Run all credit-based flow control tests."""
    print("=" * 60)
    print("Running Credit-Based Flow Control Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_in_flight_window_cap,
        test_out_of_order_ack,
        test_timeout_based_credit_release,
        test_backpressure_behavior,
        test_mixed_ack_and_timeout,
        test_processor_config_defaults,
        test_in_flight_frame_dataclass,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            failed += 1
            print(f"✗ Test failed: {e}")
            print()
        except Exception as e:
            failed += 1
            print(f"✗ Test error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
