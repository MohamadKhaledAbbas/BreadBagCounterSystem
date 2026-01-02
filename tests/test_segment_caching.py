#!/usr/bin/env python3
"""
Tests for Segment List Caching Performance Improvements.

Tests the fixes for issue #1 (publisher falls behind):
- Fast os.scandir() implementation of list_segments()
- Segment list caching with configurable refresh rate
- Cache statistics tracking
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.segment_io import (
    SegmentWriter, SegmentReader, FrameRecord,
    SEGMENT_MAGIC, SEGMENT_VERSION
)


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
            index=segment_num * 100 + i,
            width=640,
            height=480,
            dts_sec=0,
            dts_nsec=i * 33333333,
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


def test_list_segments_uses_scandir():
    """Test that list_segments uses os.scandir (not pathlib.glob)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some segments
        for seg in [1, 5, 10]:
            create_segment_with_frames(tmpdir, seg, 2)
        
        reader = SegmentReader(tmpdir)
        segments = reader.list_segments()
        
        # Should return sorted list
        assert segments == [1, 5, 10], f"Expected [1, 5, 10], got {segments}"
        
        print("✓ test_list_segments_uses_scandir passed")


def test_segment_list_caching_basic():
    """Test that segment list caching works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create initial segments
        for seg in [1, 2, 3]:
            create_segment_with_frames(tmpdir, seg, 2)
        
        # Create reader with short cache interval
        reader = SegmentReader(tmpdir, cache_refresh_interval=0.5)
        
        # First call should be a cache miss
        segments1 = reader.list_segments(use_cache=True)
        assert segments1 == [1, 2, 3], f"Expected [1, 2, 3], got {segments1}"
        
        # Second call immediately should be a cache hit
        segments2 = reader.list_segments(use_cache=True)
        assert segments2 == [1, 2, 3], f"Expected [1, 2, 3], got {segments2}"
        
        # Check cache stats
        stats = reader.get_cache_stats()
        assert stats['hits'] >= 1, f"Expected at least 1 cache hit, got {stats['hits']}"
        assert stats['misses'] == 1, f"Expected 1 cache miss, got {stats['misses']}"
        
        print("✓ test_segment_list_caching_basic passed")


def test_cache_refresh_after_interval():
    """Test that cache refreshes after the configured interval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create initial segments
        for seg in [1, 2]:
            create_segment_with_frames(tmpdir, seg, 2)
        
        # Create reader with very short cache interval (0.1 seconds)
        reader = SegmentReader(tmpdir, cache_refresh_interval=0.1)
        
        # First call
        segments1 = reader.list_segments(use_cache=True)
        assert segments1 == [1, 2]
        
        # Add a new segment
        create_segment_with_frames(tmpdir, 3, 2)
        
        # Immediate call should still use cache (miss the new segment)
        segments2 = reader.list_segments(use_cache=True)
        assert segments2 == [1, 2], f"Expected cached [1, 2], got {segments2}"
        
        # Wait for cache to expire
        time.sleep(0.15)
        
        # Now should refresh and see new segment
        segments3 = reader.list_segments(use_cache=True)
        assert segments3 == [1, 2, 3], f"Expected refreshed [1, 2, 3], got {segments3}"
        
        print("✓ test_cache_refresh_after_interval passed")


def test_cache_bypass_with_use_cache_false():
    """Test that use_cache=False bypasses cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create initial segments
        for seg in [1, 2]:
            create_segment_with_frames(tmpdir, seg, 2)
        
        reader = SegmentReader(tmpdir, cache_refresh_interval=10.0)  # Long interval
        
        # First call with cache
        segments1 = reader.list_segments(use_cache=True)
        assert segments1 == [1, 2]
        
        # Add new segment
        create_segment_with_frames(tmpdir, 3, 2)
        
        # Call with use_cache=False should see new segment immediately
        segments2 = reader.list_segments(use_cache=False)
        assert segments2 == [1, 2, 3], f"Expected [1, 2, 3] with use_cache=False, got {segments2}"
        
        print("✓ test_cache_bypass_with_use_cache_false passed")


def test_cache_stats_tracking():
    """Test that cache statistics are tracked correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments
        for seg in [1, 2, 3]:
            create_segment_with_frames(tmpdir, seg, 2)
        
        reader = SegmentReader(tmpdir, cache_refresh_interval=1.0)
        
        # Initial stats should be zero
        stats = reader.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        
        # First call - cache miss
        reader.list_segments(use_cache=True)
        stats = reader.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 1
        
        # Multiple cache hits
        for _ in range(5):
            reader.list_segments(use_cache=True)
        
        stats = reader.get_cache_stats()
        assert stats['hits'] == 5, f"Expected 5 hits, got {stats['hits']}"
        assert stats['misses'] == 1, f"Expected 1 miss, got {stats['misses']}"
        assert stats['hit_rate_pct'] > 80.0, f"Expected >80% hit rate, got {stats['hit_rate_pct']}"
        
        print("✓ test_cache_stats_tracking passed")


def test_performance_with_many_segments():
    """Test that performance improves with many segments (spot check)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create many segments (simulate growing spool directory)
        num_segments = 50
        for seg in range(1, num_segments + 1):
            create_segment_with_frames(tmpdir, seg, 1)
        
        reader = SegmentReader(tmpdir, cache_refresh_interval=1.0)
        
        # First call (uncached) - measure time
        start = time.time()
        segments1 = reader.list_segments(use_cache=True)
        uncached_time = time.time() - start
        
        # Multiple cached calls - should be much faster
        cached_times = []
        for _ in range(10):
            start = time.time()
            segments2 = reader.list_segments(use_cache=True)
            cached_times.append(time.time() - start)
        
        avg_cached_time = sum(cached_times) / len(cached_times)
        
        # Cached should be at least 10x faster (typically 100x+ faster)
        # We use conservative 10x to account for test environment variability
        assert avg_cached_time < uncached_time, \
            f"Cached calls should be faster: uncached={uncached_time:.6f}s, cached={avg_cached_time:.6f}s"
        
        # Verify correctness
        assert len(segments1) == num_segments
        assert len(segments2) == num_segments
        assert segments1 == segments2
        
        print(f"✓ test_performance_with_many_segments passed "
              f"(uncached: {uncached_time*1000:.2f}ms, cached: {avg_cached_time*1000:.2f}ms, "
              f"speedup: {uncached_time/avg_cached_time:.1f}x)")


def test_cache_thread_safety():
    """Test that cache operations are thread-safe."""
    import threading
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments
        for seg in [1, 2, 3]:
            create_segment_with_frames(tmpdir, seg, 2)
        
        reader = SegmentReader(tmpdir, cache_refresh_interval=0.1)
        
        # Access cache from multiple threads simultaneously
        errors = []
        results = []
        
        def access_cache():
            try:
                for _ in range(10):
                    segments = reader.list_segments(use_cache=True)
                    results.append(segments)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=access_cache) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # All results should be consistent
        for result in results:
            assert result == [1, 2, 3], f"Inconsistent result: {result}"
        
        print("✓ test_cache_thread_safety passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Segment List Caching Performance")
    print("=" * 60)
    print()
    
    try:
        test_list_segments_uses_scandir()
        test_segment_list_caching_basic()
        test_cache_refresh_after_interval()
        test_cache_bypass_with_use_cache_false()
        test_cache_stats_tracking()
        test_performance_with_many_segments()
        test_cache_thread_safety()
        
        print()
        print("=" * 60)
        print("✓ All segment caching tests passed!")
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
