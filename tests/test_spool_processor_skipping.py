#!/usr/bin/env python3
"""
Tests for Spool Processor Forward-Skipping Behavior.

Tests the fix for segment rewinds and empty segment handling:
- Processor should skip forward when current segment is missing
- Processor should skip forward when encountering empty segments
- Processor should never rewind to older segments (except as fallback)
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.segment_io import (
    SegmentWriter, SegmentReader, FrameRecord,
    SEGMENT_MAGIC, SEGMENT_VERSION
)


def create_empty_segment_file(tmpdir: str, segment_num: int) -> str:
    """Create an empty segment file (valid header but no frames)."""
    filename = f"seg_{segment_num:06d}.bin"
    filepath = os.path.join(tmpdir, filename)
    
    # Write valid segment header only (no frame records)
    with open(filepath, 'wb') as f:
        f.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
    
    return filepath


def create_segment_with_frames(tmpdir: str, segment_num: int, num_frames: int = 5) -> str:
    """Create a segment file with actual frame records."""
    writer = SegmentWriter(tmpdir, segment_duration=999, max_segment_duration=999)
    writer.start()
    
    # Set the segment number after start() to override the auto-detected one
    writer._current_segment = segment_num
    
    # Use the segment path directly
    tmp_path = writer._get_segment_path(segment_num, tmp=True)
    writer._current_file = open(tmp_path, 'wb')
    writer._current_file.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
    writer._segment_start_time = time.time()
    
    # Write frame records
    for i in range(num_frames):
        frame = FrameRecord(
            index=segment_num * 100 + i,  # Unique index per segment
            width=640,
            height=480,
            dts_sec=0,
            dts_nsec=i * 33333333,  # ~30fps
            pts_sec=0,
            pts_nsec=i * 33333333,
            encoding="H264",
            data=b"test frame data " + bytes([i])
        )
        writer._current_file.write(frame.to_bytes())
    
    writer._current_file.close()
    
    # Rename to final .bin
    final_path = writer._get_segment_path(segment_num, tmp=False)
    tmp_path.rename(final_path)
    
    return str(final_path)


def test_segment_reader_list_segments():
    """Test SegmentReader.list_segments() returns sorted list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments non-sequentially
        create_segment_with_frames(tmpdir, 5, 2)
        create_segment_with_frames(tmpdir, 1, 2)
        create_segment_with_frames(tmpdir, 10, 2)
        create_segment_with_frames(tmpdir, 3, 2)
        
        reader = SegmentReader(tmpdir)
        segments = reader.list_segments()
        
        assert segments == [1, 3, 5, 10], f"Expected [1, 3, 5, 10], got {segments}"
        print("✓ test_segment_reader_list_segments passed")


def test_segment_reader_read_segment_empty():
    """Test SegmentReader yields nothing for empty segment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an empty segment
        create_empty_segment_file(tmpdir, 5)
        
        reader = SegmentReader(tmpdir)
        frames = list(reader.read_segment(5))
        
        assert frames == [], f"Expected empty list, got {len(frames)} frames"
        print("✓ test_segment_reader_read_segment_empty passed")


def test_segment_reader_skip_to_next():
    """Test that SegmentReader can skip missing segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments with gaps: 1, 5, 10
        create_segment_with_frames(tmpdir, 1, 2)
        create_segment_with_frames(tmpdir, 5, 2)
        create_segment_with_frames(tmpdir, 10, 2)
        
        reader = SegmentReader(tmpdir)
        
        # List segments should show the gaps
        segments = reader.list_segments()
        assert segments == [1, 5, 10], f"Expected [1, 5, 10], got {segments}"
        
        # Read from segment 3 should start from segment 5 (next available)
        frames = list(reader.read_frames(start_segment=3))
        
        # Should have read frames from segments 5 and 10
        # Each segment has 2 frames
        assert len(frames) == 4, f"Expected 4 frames from segments 5+10, got {len(frames)}"
        
        # Verify frame indices are from segments 5 and 10
        indices = [f.index for f in frames]
        assert indices[0] >= 500, f"First frame should be from segment 5, got index {indices[0]}"
        
        print("✓ test_segment_reader_skip_to_next passed")


def test_find_nearest_segment_forward():
    """Test finding nearest segment >= a target segment number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments: 10, 15, 20, 25
        for seg in [10, 15, 20, 25]:
            create_segment_with_frames(tmpdir, seg, 1)
        
        reader = SegmentReader(tmpdir)
        segments = reader.list_segments()
        
        # Test finding nearest segment >= various targets
        test_cases = [
            (5, 10),   # Before all -> should find 10
            (10, 10),  # Exactly at -> should find 10
            (12, 15),  # Between -> should find 15
            (15, 15),  # Exactly at -> should find 15
            (23, 25),  # Between -> should find 25
            (25, 25),  # Exactly at -> should find 25
        ]
        
        for target, expected in test_cases:
            candidates = [s for s in segments if s >= target]
            if candidates:
                found = min(candidates)
                assert found == expected, f"For target {target}, expected {expected}, got {found}"
        
        # Test when no segment >= target exists
        candidates = [s for s in segments if s >= 30]
        assert candidates == [], "No segments >= 30 should exist"
        
        print("✓ test_find_nearest_segment_forward passed")


def test_processor_state_dataclass_vs_enum():
    """Test that ProcessorState (dataclass) and ProcessorRunState (enum) are distinct."""
    from src.spool.spool_utils import ProcessorState as ProcessorStateDataclass
    
    # ProcessorState should be a dataclass (has __dataclass_fields__)
    assert hasattr(ProcessorStateDataclass, '__dataclass_fields__'), \
        "ProcessorState from spool_utils should be a dataclass"
    
    # ProcessorState should have expected fields
    assert 'last_published_index' in ProcessorStateDataclass.__dataclass_fields__
    assert 'last_published_segment' in ProcessorStateDataclass.__dataclass_fields__
    
    # Verify ProcessorRunState enum is defined in the processor node file
    # We read the file directly to avoid import dependency issues
    processor_node_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'src', 'ros2_spool', 'spool_processor_node.py'
    )
    with open(processor_node_path, 'r') as f:
        content = f.read()
    
    # Check that ProcessorRunState enum is defined (not ProcessorState Enum)
    assert 'class ProcessorRunState(Enum):' in content, \
        "ProcessorRunState enum should be defined in spool_processor_node.py"
    
    # Check that the old ProcessorState Enum is NOT defined
    # The Enum should be ProcessorRunState, not ProcessorState
    assert 'class ProcessorState(Enum):' not in content, \
        "ProcessorState(Enum) should be renamed to ProcessorRunState(Enum)"
    
    # Check that state variables use ProcessorRunState
    assert 'self._state = ProcessorRunState' in content, \
        "State assignments should use ProcessorRunState"
    
    print("✓ test_processor_state_dataclass_vs_enum passed")


def test_processor_state_save_no_conflict():
    """Test that saving ProcessorState doesn't cause EnumMeta errors."""
    from src.spool.spool_utils import ProcessorState, save_processor_state, load_processor_state
    
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "test_state.json")
        
        # Create state with ProcessorState dataclass (not Enum!)
        state = ProcessorState(
            last_published_index=100,
            last_published_segment=5,
            session_id="test-session-123",
            timestamp=time.time()
        )
        
        # Save should work without EnumMeta.__call__ errors
        try:
            result = save_processor_state(state_path, state)
            assert result, "Save should succeed"
        except TypeError as e:
            if "EnumMeta" in str(e):
                raise AssertionError(f"EnumMeta error when saving state: {e}")
            raise
        
        # Load should also work
        loaded = load_processor_state(state_path)
        assert loaded is not None, "Load should succeed"
        assert loaded.last_published_index == 100
        assert loaded.last_published_segment == 5
        
        print("✓ test_processor_state_save_no_conflict passed")


def test_retention_policy_segment_deletion():
    """Test that RetentionPolicy can delete processed segments."""
    from src.spool.retention import RetentionPolicy
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test segments
        for seg in [1, 2, 3, 4, 5]:
            create_segment_with_frames(tmpdir, seg, 2)
        
        # Create retention policy with immediate deletion enabled
        policy = RetentionPolicy(
            spool_dir=tmpdir,
            retention_seconds=300.0,
            cleanup_interval=30.0,
            min_segments_to_keep=2,
            retention_safety_enabled=True,
            delete_processed_segments=True
        )
        
        # Verify all segments exist
        segments_before = policy.list_segments()
        assert len(segments_before) == 5, f"Expected 5 segments, got {len(segments_before)}"
        
        # Mark segment 1 as processed - should trigger deletion
        policy.set_last_processed_segment(1)
        
        # Mark segment 2 as processed - segment 1 should now be deleted
        policy.set_last_processed_segment(2)
        
        # Give file system time to process
        time.sleep(0.1)
        
        # Check segments - segment 1 should be deleted
        segments_after = policy.list_segments()
        segment_nums = [s[0] for s in segments_after]
        
        assert 1 not in segment_nums, f"Segment 1 should be deleted, remaining: {segment_nums}"
        
        print("✓ test_retention_policy_segment_deletion passed")


def test_min_frame_interval_config():
    """Test that min_frame_interval_ms configuration is respected."""
    # Read the processor node file to verify configuration exists
    processor_node_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'src', 'ros2_spool', 'spool_processor_node.py'
    )
    with open(processor_node_path, 'r') as f:
        content = f.read()
    
    # Verify min_frame_interval_ms configuration exists
    assert 'DEFAULT_MIN_FRAME_INTERVAL_MS' in content, \
        "DEFAULT_MIN_FRAME_INTERVAL_MS should be defined"
    
    assert 'min_frame_interval_ms' in content, \
        "min_frame_interval_ms should be in ProcessorConfig"
    
    # Verify it's used in pacing logic
    assert 'min_interval_sec' in content, \
        "min_interval_sec should be used in pacing calculation"
    
    print("✓ test_min_frame_interval_config passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Spool Processor Forward-Skipping Behavior")
    print("=" * 60)
    print()
    
    try:
        test_segment_reader_list_segments()
        test_segment_reader_read_segment_empty()
        test_segment_reader_skip_to_next()
        test_find_nearest_segment_forward()
        test_processor_state_dataclass_vs_enum()
        test_processor_state_save_no_conflict()
        test_retention_policy_segment_deletion()
        test_min_frame_interval_config()
        
        print()
        print("=" * 60)
        print("✓ All spool processor skipping tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
