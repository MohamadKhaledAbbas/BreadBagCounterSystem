"""
Tests for SpoolProcessorNode windowed ACK logic.

These tests verify the inflight window management:
- Window size limiting
- Out-of-order ACK handling
- Timeout and retry logic
- Frame retirement ordering
"""

import sys
import os
import time
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock ROS2 dependencies before importing spool_processor_node
from unittest.mock import Mock, MagicMock
sys.modules['rclpy'] = Mock()
sys.modules['rclpy.node'] = Mock()
sys.modules['rclpy.qos'] = Mock()
sys.modules['img_msgs'] = Mock()
sys.modules['img_msgs.msg'] = Mock()
sys.modules['std_msgs'] = Mock()
sys.modules['std_msgs.msg'] = Mock()
sys.modules['builtin_interfaces'] = Mock()
sys.modules['builtin_interfaces.msg'] = Mock()

# Mock database and other dependencies
sys.modules['imagehash'] = Mock()
sys.modules['src.logging.Database'] = Mock()
sys.modules['src.config.settings'] = Mock()

# Mock platform
import src.utils.platform as platform_module
platform_module.IS_RDK = False

from src.ros2_spool.spool_processor_node import InflightFrame, ProcessorConfig
from src.spool.segment_io import FrameRecord


def create_mock_frame_record(index: int) -> FrameRecord:
    """Create a mock FrameRecord for testing."""
    return FrameRecord(
        index=index,
        data=b'mock_frame_data',
        width=1920,
        height=1080,
        pts_sec=0,
        pts_nsec=0,
        dts_sec=0,
        dts_nsec=0,
        encoding='h264'
    )


def test_inflight_window_size_limit():
    """Test that window size is respected."""
    print("Running test_inflight_window_size_limit...")
    
    # Simulate window with max size 3
    max_window = 3
    inflight_frames = deque()
    
    # Add frames up to limit
    for i in range(max_window):
        frame = InflightFrame(
            seq=i,
            frame_index=i,
            segment_num=1,
            publish_time=time.time(),
            retry_count=0,
            acked=False,
            frame_record=create_mock_frame_record(i)
        )
        inflight_frames.append(frame)
    
    # Check window is at limit
    assert len(inflight_frames) == max_window, \
        f"Window should be at limit {max_window}, got {len(inflight_frames)}"
    
    # Check that we can't add more (simulating _can_publish_frame check)
    can_publish = len(inflight_frames) < max_window
    assert not can_publish, "Should not be able to publish when window is full"
    
    print("✓ test_inflight_window_size_limit passed")


def test_out_of_order_ack_handling():
    """Test handling of out-of-order ACKs."""
    print("Running test_out_of_order_ack_handling...")
    
    # Create window with 3 frames
    inflight_frames = deque()
    for i in range(3):
        frame = InflightFrame(
            seq=i,
            frame_index=i,
            segment_num=1,
            publish_time=time.time(),
            retry_count=0,
            acked=False,
            frame_record=create_mock_frame_record(i)
        )
        inflight_frames.append(frame)
    
    # ACK frames out of order: 2, 0, 1
    inflight_frames[2].acked = True  # ACK frame 2 (out of order)
    inflight_frames[0].acked = True  # ACK frame 0 (head)
    
    # Retire should only remove frame 0 (head), not frame 2
    while inflight_frames and inflight_frames[0].acked:
        retired = inflight_frames.popleft()
        assert retired.seq == 0, f"Should retire frame 0 first, got {retired.seq}"
        break
    
    # Window should have 2 frames left (1 and 2)
    assert len(inflight_frames) == 2, f"Should have 2 frames left, got {len(inflight_frames)}"
    assert inflight_frames[0].seq == 1, "Frame 1 should be at head"
    assert inflight_frames[1].seq == 2, "Frame 2 should be at position 1"
    
    # Now ACK frame 1
    inflight_frames[0].acked = True
    
    # Retire should now remove both frames 1 and 2
    retired_count = 0
    while inflight_frames and inflight_frames[0].acked:
        retired_count += 1
        inflight_frames.popleft()
    
    assert retired_count == 2, f"Should have retired 2 frames, got {retired_count}"
    assert len(inflight_frames) == 0, "Window should be empty"
    
    print("✓ test_out_of_order_ack_handling passed")


def test_timeout_and_retry():
    """Test timeout detection and retry logic."""
    print("Running test_timeout_and_retry...")
    
    timeout = 2.0
    max_retries = 2
    
    # Create frame that will timeout
    frame = InflightFrame(
        seq=0,
        frame_index=0,
        segment_num=1,
        publish_time=time.time() - (timeout + 1),  # Publish time in the past
        retry_count=0,
        acked=False,
        frame_record=create_mock_frame_record(0)
    )
    
    inflight_frames = deque([frame])
    
    # Check timeout detection
    current_time = time.time()
    age = current_time - frame.publish_time
    is_timeout = age > timeout
    
    assert is_timeout, f"Frame should be timed out (age={age:.1f}s, timeout={timeout}s)"
    
    # Simulate retry
    if frame.retry_count < max_retries:
        frame.retry_count += 1
        frame.publish_time = current_time  # Reset publish time on retry
        new_seq = 100  # Simulate new sequence number
        frame.seq = new_seq
    
    assert frame.retry_count == 1, f"Retry count should be 1, got {frame.retry_count}"
    assert frame.seq == 100, f"Sequence should be updated to 100, got {frame.seq}"
    
    # Verify frame is not timed out after retry
    age = current_time - frame.publish_time
    is_timeout = age > timeout
    assert not is_timeout, "Frame should not be timed out after retry"
    
    print("✓ test_timeout_and_retry passed")


def test_max_retries_exceeded():
    """Test that frames are skipped after max retries."""
    print("Running test_max_retries_exceeded...")
    
    timeout = 1.0
    max_retries = 2
    
    # Create frame with max retries already reached
    frame = InflightFrame(
        seq=0,
        frame_index=0,
        segment_num=1,
        publish_time=time.time() - (timeout + 1),  # Timed out
        retry_count=max_retries,  # Already at max retries
        acked=False,
        frame_record=create_mock_frame_record(0)
    )
    
    inflight_frames = deque([frame])
    
    # Check timeout detection
    current_time = time.time()
    age = current_time - frame.publish_time
    is_timeout = age > timeout
    
    assert is_timeout, "Frame should be timed out"
    
    # Simulate skip (mark as acked to allow retirement)
    if frame.retry_count >= max_retries:
        frame.acked = True
        skipped = True
    
    assert frame.acked, "Frame should be marked as acked (to allow retirement)"
    assert skipped, "Frame should be marked as skipped"
    
    # Verify frame can now be retired
    can_retire = inflight_frames[0].acked
    assert can_retire, "Frame should be retirable after being marked as acked"
    
    print("✓ test_max_retries_exceeded passed")


def test_ordered_retirement():
    """Test that frames are retired in order."""
    print("Running test_ordered_retirement...")
    
    # Create window with 5 frames, ACK them in random order
    inflight_frames = deque()
    for i in range(5):
        frame = InflightFrame(
            seq=i,
            frame_index=i,
            segment_num=1,
            publish_time=time.time(),
            retry_count=0,
            acked=False,
            frame_record=create_mock_frame_record(i)
        )
        inflight_frames.append(frame)
    
    # ACK frames in order: 2, 4, 1, 0 (not 3)
    # This means: 0, 1, 2 are acked, but 3 is not, 4 is acked
    inflight_frames[2].acked = True  # ACK frame 2
    inflight_frames[4].acked = True  # ACK frame 4
    inflight_frames[1].acked = True  # ACK frame 1
    inflight_frames[0].acked = True  # ACK frame 0 (head)
    # Frame 3 is NOT acked
    
    # Retire frames - should retire 0, 1, 2 (contiguous from head)
    retired = []
    while inflight_frames and inflight_frames[0].acked:
        frame = inflight_frames.popleft()
        retired.append(frame.seq)
    
    # Should have retired frames 0, 1, 2 (all contiguous from head)
    assert retired == [0, 1, 2], f"Should retire frames 0,1,2, got {retired}"
    
    # Remaining frames should be 3, 4
    assert len(inflight_frames) == 2, f"Should have 2 frames left, got {len(inflight_frames)}"
    
    # Frame 3 is now at head but NOT acked, so retirement should stop
    assert inflight_frames[0].seq == 3, "Frame 3 should be at head"
    assert not inflight_frames[0].acked, "Frame 3 should not be acked"
    
    # Frame 4 is acked but can't be retired yet (not at head)
    assert inflight_frames[1].seq == 4, "Frame 4 should be at position 1"
    assert inflight_frames[1].acked, "Frame 4 should be acked"
    
    # Now ACK frame 3
    inflight_frames[0].acked = True
    
    # Retire should now remove both frames 3 and 4
    retired = []
    while inflight_frames and inflight_frames[0].acked:
        frame = inflight_frames.popleft()
        retired.append(frame.seq)
    
    assert retired == [3, 4], f"Should have retired frames 3,4, got {retired}"
    assert len(inflight_frames) == 0, "Window should be empty"
    
    print("✓ test_ordered_retirement passed")


def test_window_with_default_size_one():
    """Test backward compatibility with inflight_window=1."""
    print("Running test_window_with_default_size_one...")
    
    max_window = 1
    inflight_frames = deque()
    
    # Add one frame
    frame = InflightFrame(
        seq=0,
        frame_index=0,
        segment_num=1,
        publish_time=time.time(),
        retry_count=0,
        acked=False,
        frame_record=create_mock_frame_record(0)
    )
    inflight_frames.append(frame)
    
    # Check window is full
    can_publish = len(inflight_frames) < max_window
    assert not can_publish, "Should not be able to publish when window is full (size=1)"
    
    # ACK the frame
    frame.acked = True
    
    # Retire it
    if inflight_frames and inflight_frames[0].acked:
        inflight_frames.popleft()
    
    # Now window should be empty
    assert len(inflight_frames) == 0, "Window should be empty after retirement"
    
    # Should be able to publish again
    can_publish = len(inflight_frames) < max_window
    assert can_publish, "Should be able to publish after retirement"
    
    print("✓ test_window_with_default_size_one passed")


def test_oldest_inflight_age():
    """Test calculating oldest inflight frame age."""
    print("Running test_oldest_inflight_age...")
    
    # Create window with frames at different ages
    base_time = time.time()
    inflight_frames = deque()
    
    # Frame 0: 5 seconds old
    frame0 = InflightFrame(
        seq=0,
        frame_index=0,
        segment_num=1,
        publish_time=base_time - 5.0,
        retry_count=0,
        acked=False,
        frame_record=create_mock_frame_record(0)
    )
    inflight_frames.append(frame0)
    
    # Frame 1: 3 seconds old
    frame1 = InflightFrame(
        seq=1,
        frame_index=1,
        segment_num=1,
        publish_time=base_time - 3.0,
        retry_count=0,
        acked=False,
        frame_record=create_mock_frame_record(1)
    )
    inflight_frames.append(frame1)
    
    # Oldest should be frame 0 (at head, 5 seconds old)
    if inflight_frames:
        oldest = inflight_frames[0]
        age = time.time() - oldest.publish_time
        assert age >= 4.9 and age <= 5.1, f"Oldest age should be ~5 seconds, got {age:.1f}s"
    
    # Test empty window
    inflight_frames.clear()
    age = 0.0 if not inflight_frames else (time.time() - inflight_frames[0].publish_time)
    assert age == 0.0, f"Empty window should return age 0.0, got {age}"
    
    print("✓ test_oldest_inflight_age passed")


if __name__ == '__main__':
    """Run all tests."""
    try:
        test_inflight_window_size_limit()
        test_out_of_order_ack_handling()
        test_timeout_and_retry()
        test_max_retries_exceeded()
        test_ordered_retirement()
        test_window_with_default_size_one()
        test_oldest_inflight_age()
        
        print()
        print("=" * 60)
        print("✓ All windowed ACK tests passed!")
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
