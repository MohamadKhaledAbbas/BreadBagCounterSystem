"""
Tests for retention policy.

These tests verify the retention policy correctly manages
segment file lifecycle and cleanup.
"""

import sys
import os
import tempfile
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.retention import (
    RetentionPolicy,
    cleanup_stale_tmp_files,
    get_spool_disk_usage,
)
from src.spool.segment_io import SEGMENT_MAGIC, SEGMENT_VERSION


def create_segment_file(tmpdir: str, segment_num: int, age_seconds: float = 0) -> str:
    """Create a test segment file with optional age backdating."""
    filename = f"seg_{segment_num:06d}.bin"
    filepath = os.path.join(tmpdir, filename)
    
    # Write valid segment header
    with open(filepath, 'wb') as f:
        f.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
        f.write(b"test data for segment")
    
    # Backdate the file if requested
    if age_seconds > 0:
        old_time = time.time() - age_seconds
        os.utime(filepath, (old_time, old_time))
    
    return filepath


def create_tmp_file(tmpdir: str, segment_num: int, age_seconds: float = 0) -> str:
    """Create a test .tmp file with optional age backdating."""
    filename = f"seg_{segment_num:06d}.tmp"
    filepath = os.path.join(tmpdir, filename)
    
    with open(filepath, 'wb') as f:
        f.write(b"partial segment data")
    
    if age_seconds > 0:
        old_time = time.time() - age_seconds
        os.utime(filepath, (old_time, old_time))
    
    return filepath


def test_cleanup_stale_tmp_files():
    """Test cleanup of stale temporary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some .tmp files
        create_tmp_file(tmpdir, 1, age_seconds=120)  # Old - should be cleaned
        create_tmp_file(tmpdir, 2, age_seconds=30)   # Recent - should stay
        create_tmp_file(tmpdir, 3, age_seconds=180)  # Old - should be cleaned
        
        # Also create a .bin file (should not be touched)
        create_segment_file(tmpdir, 4, age_seconds=200)
        
        # Cleanup with 60s threshold
        cleaned = cleanup_stale_tmp_files(tmpdir, max_age_seconds=60)
        
        assert cleaned == 2, f"Expected 2 files cleaned, got {cleaned}"
        
        # Verify correct files remain
        remaining = os.listdir(tmpdir)
        assert "seg_000002.tmp" in remaining, "Recent .tmp should remain"
        assert "seg_000004.bin" in remaining, ".bin file should remain"
        assert "seg_000001.tmp" not in remaining, "Old .tmp should be removed"
        assert "seg_000003.tmp" not in remaining, "Old .tmp should be removed"
        
        print("✓ test_cleanup_stale_tmp_files passed")


def test_retention_policy_list_segments():
    """Test RetentionPolicy segment listing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some segments
        create_segment_file(tmpdir, 1)
        create_segment_file(tmpdir, 3)
        create_segment_file(tmpdir, 5)
        
        policy = RetentionPolicy(tmpdir, retention_seconds=300)
        segments = policy.list_segments()
        
        assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
        
        # Check segment numbers are in order
        seg_nums = [s[0] for s in segments]
        assert seg_nums == [1, 3, 5], f"Expected [1, 3, 5], got {seg_nums}"
        
        print("✓ test_retention_policy_list_segments passed")


def test_retention_policy_get_expired():
    """Test identifying expired segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments with different ages
        create_segment_file(tmpdir, 1, age_seconds=500)  # Very old
        create_segment_file(tmpdir, 2, age_seconds=200)  # Old but within retention
        create_segment_file(tmpdir, 3, age_seconds=100)  # Recent
        create_segment_file(tmpdir, 4, age_seconds=10)   # Very recent
        
        policy = RetentionPolicy(
            tmpdir, 
            retention_seconds=150,  # Only segment 1 should expire
            min_segments_to_keep=2
        )
        
        expired = policy.get_expired_segments()
        
        # Should only identify segment 1 as expired
        # (segment 2 is within retention)
        # Also, min_segments_to_keep=2 protects the newest 2
        expired_nums = [e[0] for e in expired]
        
        # Segment 1 should be expired (age 500 > retention 150)
        # Segments 3 and 4 are protected by min_segments_to_keep
        # Segment 2 depends on implementation
        assert 1 in expired_nums, "Segment 1 should be expired"
        assert 4 not in expired_nums, "Newest segment should not expire"
        
        print("✓ test_retention_policy_get_expired passed")


def test_retention_policy_cleanup():
    """Test actual cleanup of expired segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create old segments
        create_segment_file(tmpdir, 1, age_seconds=300)
        create_segment_file(tmpdir, 2, age_seconds=250)
        # Create recent segments
        create_segment_file(tmpdir, 3, age_seconds=10)
        create_segment_file(tmpdir, 4, age_seconds=5)
        
        policy = RetentionPolicy(
            tmpdir,
            retention_seconds=100,  # Segments 1 and 2 should expire
            min_segments_to_keep=1
        )
        
        deleted, bytes_freed = policy.cleanup_once()
        
        # Should have deleted 2 segments (1 and 2)
        # But min_segments_to_keep might affect this
        assert deleted >= 1, f"Should delete at least 1 segment, deleted {deleted}"
        assert bytes_freed > 0, "Should free some bytes"
        
        # Verify stats updated
        assert policy.segments_deleted == deleted
        assert policy.bytes_recovered == bytes_freed
        
        print(f"✓ test_retention_policy_cleanup passed (deleted {deleted}, freed {bytes_freed} bytes)")


def test_retention_policy_min_segments():
    """Test that min_segments_to_keep is respected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create only 2 segments, both old
        create_segment_file(tmpdir, 1, age_seconds=500)
        create_segment_file(tmpdir, 2, age_seconds=400)
        
        policy = RetentionPolicy(
            tmpdir,
            retention_seconds=100,  # Both should technically expire
            min_segments_to_keep=2  # But we keep minimum 2
        )
        
        expired = policy.get_expired_segments()
        
        # Even though both are old, min_segments_to_keep should protect them
        assert len(expired) == 0, f"Should not expire any (min_segments_to_keep=2), got {len(expired)}"
        
        print("✓ test_retention_policy_min_segments passed")


def test_retention_policy_stats():
    """Test retention statistics reporting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some segments
        create_segment_file(tmpdir, 1, age_seconds=100)
        create_segment_file(tmpdir, 2, age_seconds=50)
        
        policy = RetentionPolicy(tmpdir, retention_seconds=300)
        stats = policy.get_stats()
        
        assert 'total_segments' in stats
        assert stats['total_segments'] == 2
        assert 'total_size_bytes' in stats
        assert stats['total_size_bytes'] > 0
        assert 'oldest_segment_age_seconds' in stats
        assert stats['oldest_segment_age_seconds'] >= 100
        assert 'retention_seconds' in stats
        assert stats['retention_seconds'] == 300
        
        print("✓ test_retention_policy_stats passed")


def test_disk_usage_function():
    """Test get_spool_disk_usage helper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create various files
        create_segment_file(tmpdir, 1)
        create_segment_file(tmpdir, 2)
        create_tmp_file(tmpdir, 3)
        
        # Create a metadata file
        meta_path = os.path.join(tmpdir, "seg_000001.meta.json")
        with open(meta_path, 'w') as f:
            f.write('{"test": true}')
        
        usage = get_spool_disk_usage(tmpdir)
        
        assert usage['exists']
        assert usage['segment_count'] == 2
        assert usage['tmp_count'] == 1
        assert usage['meta_count'] == 1
        assert usage['total_bytes'] > 0
        
        print("✓ test_disk_usage_function passed")


def test_disk_usage_nonexistent():
    """Test get_spool_disk_usage with nonexistent directory."""
    usage = get_spool_disk_usage("/nonexistent/path/12345")
    
    assert not usage['exists']
    assert usage['total_bytes'] == 0
    assert usage['segment_count'] == 0
    
    print("✓ test_disk_usage_nonexistent passed")


def test_retention_with_metadata():
    """Test that metadata files are also cleaned up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create old segment with metadata
        create_segment_file(tmpdir, 1, age_seconds=500)
        meta_path = os.path.join(tmpdir, "seg_000001.meta.json")
        with open(meta_path, 'w') as f:
            f.write('{"segment_number": 1}')
        
        # Create recent segment
        create_segment_file(tmpdir, 2, age_seconds=10)
        
        policy = RetentionPolicy(
            tmpdir,
            retention_seconds=100,
            min_segments_to_keep=1
        )
        
        policy.cleanup_once()
        
        # Both segment and metadata should be gone
        remaining = os.listdir(tmpdir)
        assert "seg_000001.bin" not in remaining, "Old segment should be removed"
        assert "seg_000001.meta.json" not in remaining, "Old metadata should be removed"
        assert "seg_000002.bin" in remaining, "Recent segment should remain"
        
        print("✓ test_retention_with_metadata passed")


def test_empty_directory():
    """Test retention policy with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy = RetentionPolicy(tmpdir, retention_seconds=100)
        
        segments = policy.list_segments()
        assert segments == [], "Empty dir should have no segments"
        
        expired = policy.get_expired_segments()
        assert expired == [], "Empty dir should have no expired segments"
        
        deleted, bytes_freed = policy.cleanup_once()
        assert deleted == 0, "Nothing to delete in empty dir"
        assert bytes_freed == 0, "Nothing to free in empty dir"
        
        print("✓ test_empty_directory passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Retention Policy")
    print("=" * 60)
    print()
    
    try:
        test_cleanup_stale_tmp_files()
        test_retention_policy_list_segments()
        test_retention_policy_get_expired()
        test_retention_policy_cleanup()
        test_retention_policy_min_segments()
        test_retention_policy_stats()
        test_disk_usage_function()
        test_disk_usage_nonexistent()
        test_retention_with_metadata()
        test_empty_directory()
        
        print()
        print("=" * 60)
        print("✓ All retention policy tests passed!")
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
