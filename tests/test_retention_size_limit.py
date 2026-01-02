#!/usr/bin/env python3
"""
Tests for Retention Policy Size Limit Enforcement.

Tests the fix for size limit not being respected:
- Size limit is properly configured and passed to RetentionPolicy
- Segments are deleted when size exceeds limit
- Processor state file is used to determine safe segments to delete
"""

import os
import sys
import time
import tempfile
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.segment_io import SegmentWriter, FrameRecord, SEGMENT_MAGIC, SEGMENT_VERSION
from src.spool.retention import RetentionPolicy


def create_segment_with_size(tmpdir: str, segment_num: int, size_mb: float) -> str:
    """Create a segment file with approximately the specified size in MB."""
    writer = SegmentWriter(tmpdir, segment_duration=999, max_segment_duration=999)
    writer.start()
    
    writer._current_segment = segment_num
    tmp_path = writer._get_segment_path(segment_num, tmp=True)
    writer._current_file = open(tmp_path, 'wb')
    writer._current_file.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
    writer._segment_start_time = time.time()
    
    # Calculate how many frames we need to reach target size
    # Each frame has ~54 byte header + data
    target_bytes = int(size_mb * 1024 * 1024)
    frame_data_size = max(100, (target_bytes // 10) - 54)  # Split into ~10 frames
    
    for i in range(10):
        frame = FrameRecord(
            index=segment_num * 100 + i,
            width=640,
            height=480,
            dts_sec=0,
            dts_nsec=i * 33333333,
            pts_sec=0,
            pts_nsec=i * 33333333,
            encoding="H264",
            data=b"X" * frame_data_size  # Fill with data to reach size
        )
        writer._current_file.write(frame.to_bytes())
    
    writer._current_file.close()
    
    final_path = writer._get_segment_path(segment_num, tmp=False)
    tmp_path.rename(final_path)
    
    # Verify size
    actual_size = final_path.stat().st_size
    print(f"Created segment {segment_num}: {actual_size / 1024 / 1024:.2f}MB")
    
    return str(final_path)


def create_processor_state_file(spool_dir: str, last_segment: int, last_index: int):
    """Create a processor state file indicating what's been processed."""
    state_file = Path(spool_dir) / "processor_state.json"
    state = {
        "last_published_index": last_index,
        "last_published_segment": last_segment,
        "session_id": "test-session",
        "timestamp": time.time()
    }
    with open(state_file, 'w') as f:
        json.dump(state, f)
    print(f"Created processor state: segment={last_segment}, index={last_index}")


def test_size_limit_enforced():
    """Test that retention policy enforces size limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments totaling ~3MB (exceeds 2MB limit)
        # Segment 1: 0.5MB
        # Segment 2: 0.5MB
        # Segment 3: 1.0MB
        # Segment 4: 1.0MB
        # Total: 3MB > 2MB limit
        
        create_segment_with_size(tmpdir, 1, 0.5)
        create_segment_with_size(tmpdir, 2, 0.5)
        create_segment_with_size(tmpdir, 3, 1.0)
        create_segment_with_size(tmpdir, 4, 1.0)
        
        # Create processor state indicating segment 3 is being processed
        # (so segments 1-2 are safe to delete)
        create_processor_state_file(tmpdir, 3, 300)
        
        # Create retention policy with 2MB limit
        policy = RetentionPolicy(
            spool_dir=tmpdir,
            retention_seconds=3600.0,  # 1 hour - so age won't trigger deletion
            cleanup_interval=10.0,
            min_segments_to_keep=1,  # Keep at least 1
            retention_safety_enabled=True,
            max_spool_size_bytes=2 * 1024 * 1024,  # 2MB limit
            delete_processed_segments=False
        )
        
        # Check initial size
        total_size = policy._get_total_spool_size()
        print(f"Initial total size: {total_size / 1024 / 1024:.2f}MB")
        assert total_size > 2 * 1024 * 1024, f"Expected >2MB, got {total_size / 1024 / 1024:.2f}MB"
        
        # Run cleanup
        deleted, bytes_freed = policy.cleanup_once()
        
        print(f"Cleanup result: deleted={deleted}, freed={bytes_freed / 1024 / 1024:.2f}MB")
        
        # Should have deleted old segments to get under limit
        assert deleted > 0, f"Expected some segments deleted, got {deleted}"
        
        # Check final size
        final_size = policy._get_total_spool_size()
        print(f"Final total size: {final_size / 1024 / 1024:.2f}MB")
        
        # Should be at or under 2MB limit (with small tolerance for rounding)
        limit_with_tolerance = 2 * 1024 * 1024 + 1024  # 2MB + 1KB tolerance
        assert final_size <= limit_with_tolerance, \
            f"Size still exceeds limit: {final_size / 1024 / 1024:.2f}MB > 2MB"
        
        print("✓ test_size_limit_enforced passed")


def test_processor_state_protects_current_segment():
    """Test that segments being processed are protected even during size cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 4 segments, each ~1MB
        for seg in [1, 2, 3, 4]:
            create_segment_with_size(tmpdir, seg, 1.0)
        
        # Processor is currently on segment 3
        create_processor_state_file(tmpdir, 3, 300)
        
        # Create retention policy with 2MB limit
        policy = RetentionPolicy(
            spool_dir=tmpdir,
            retention_seconds=3600.0,
            cleanup_interval=10.0,
            min_segments_to_keep=1,
            retention_safety_enabled=True,
            max_spool_size_bytes=2 * 1024 * 1024,
            delete_processed_segments=False
        )
        
        # Run cleanup
        deleted, bytes_freed = policy.cleanup_once()
        
        print(f"Deleted {deleted} segments, freed {bytes_freed / 1024 / 1024:.2f}MB")
        
        # Check which segments remain
        segments = policy.list_segments()
        segment_nums = [seg[0] for seg in segments]
        
        print(f"Remaining segments: {segment_nums}")
        
        # Segment 3 (current) and 4 (newer) should be protected
        assert 3 in segment_nums, "Current segment 3 should be protected"
        assert 4 in segment_nums, "Newer segment 4 should be protected"
        
        # Segments 1 and 2 should have been deleted
        assert 1 not in segment_nums, "Old segment 1 should be deleted"
        assert 2 not in segment_nums, "Old segment 2 should be deleted"
        
        print("✓ test_processor_state_protects_current_segment passed")


def test_no_processor_state_safe_behavior():
    """Test that without processor state, retention is conservative."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments exceeding limit
        for seg in [1, 2, 3, 4]:
            create_segment_with_size(tmpdir, seg, 1.0)
        
        # NO processor state file - don't know what's safe
        
        # Create retention policy with 2MB limit
        policy = RetentionPolicy(
            spool_dir=tmpdir,
            retention_seconds=3600.0,
            cleanup_interval=10.0,
            min_segments_to_keep=2,  # Keep at least 2
            retention_safety_enabled=True,
            max_spool_size_bytes=2 * 1024 * 1024,
            delete_processed_segments=False
        )
        
        # Run cleanup
        deleted, bytes_freed = policy.cleanup_once()
        
        print(f"Deleted {deleted} segments without processor state")
        
        # Should still delete something to respect size limit,
        # but will be conservative and keep min_segments_to_keep
        segments = policy.list_segments()
        print(f"Remaining segments: {len(segments)}")
        
        # Should have at least min_segments_to_keep
        assert len(segments) >= 2, f"Should keep at least 2 segments, got {len(segments)}"
        
        print("✓ test_no_processor_state_safe_behavior passed")


def test_size_limit_parameter_passed():
    """Test that size limit parameter is correctly used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with different size limits
        policy_2mb = RetentionPolicy(
            spool_dir=tmpdir,
            max_spool_size_bytes=2 * 1024 * 1024
        )
        assert policy_2mb.max_spool_size_bytes == 2 * 1024 * 1024
        
        policy_1gb = RetentionPolicy(
            spool_dir=tmpdir,
            max_spool_size_bytes=1024 * 1024 * 1024
        )
        assert policy_1gb.max_spool_size_bytes == 1024 * 1024 * 1024
        
        # Test default value
        policy_default = RetentionPolicy(spool_dir=tmpdir)
        assert policy_default.max_spool_size_bytes == 2_147_483_648  # 2GB default
        
        print("✓ test_size_limit_parameter_passed passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Retention Policy Size Limit Enforcement")
    print("=" * 60)
    print()
    
    try:
        test_size_limit_parameter_passed()
        test_size_limit_enforced()
        test_processor_state_protects_current_segment()
        test_no_processor_state_safe_behavior()
        
        print()
        print("=" * 60)
        print("✓ All retention size limit tests passed!")
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
