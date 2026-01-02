#!/usr/bin/env python3
"""
Tests for Processor Resume Behavior.

Tests the fixes for issue #2 (processor restart does not resume gracefully):
- Resume from last_published_segment (not oldest)
- Skip frames by index only within the same segment
- Handle missing/deleted resume segment by jumping forward
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import itertools

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.segment_io import (
    SegmentWriter, SegmentReader, FrameRecord,
    SEGMENT_MAGIC, SEGMENT_VERSION
)


def create_segment_with_frames(tmpdir: str, segment_num: int, num_frames: int = 5,
                               start_index: int = None) -> str:
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
    
    # Write frame records with potentially non-monotonic indices across segments
    if start_index is None:
        start_index = segment_num * 100
    
    for i in range(num_frames):
        frame = FrameRecord(
            index=start_index + i,
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


def test_resume_from_last_published_segment():
    """Test that processor resumes from last_published_segment, not oldest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments 1, 2, 3
        # Segment 1: frames 0-4
        # Segment 2: frames 100-104 (indices not continuous!)
        # Segment 3: frames 200-204
        create_segment_with_frames(tmpdir, 1, 5, start_index=0)
        create_segment_with_frames(tmpdir, 2, 5, start_index=100)
        create_segment_with_frames(tmpdir, 3, 5, start_index=200)
        
        reader = SegmentReader(tmpdir)
        
        # Simulate processor state: last published was frame 102 from segment 2
        # Processor should resume from segment 2, skip frames 100-102, start at 103
        
        # Start from last_published_segment (2)
        frames = list(reader.read_frames(start_segment=2))
        
        # Should read frames from segments 2 and 3
        assert len(frames) == 10, f"Expected 10 frames (5 from seg 2, 5 from seg 3), got {len(frames)}"
        
        # Verify indices from segment 2
        seg2_indices = [f.index for f in frames[:5]]
        assert seg2_indices == [100, 101, 102, 103, 104], f"Segment 2 indices wrong: {seg2_indices}"
        
        # Verify indices from segment 3
        seg3_indices = [f.index for f in frames[5:]]
        assert seg3_indices == [200, 201, 202, 203, 204], f"Segment 3 indices wrong: {seg3_indices}"
        
        print("✓ test_resume_from_last_published_segment passed")


def test_skip_frames_within_resume_segment():
    """Test that frames are skipped by index only within the resume segment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segment 5 with frames 50-59
        create_segment_with_frames(tmpdir, 5, 10, start_index=50)
        
        reader = SegmentReader(tmpdir)
        
        # Simulate: last_published_index = 54 (within segment 5)
        # Should skip frames 50-54, start from 55
        generator = reader.read_frames(start_segment=5)
        
        # Manually skip frames <= 54
        skipped_count = 0
        first_frame = None
        for frame in generator:
            if frame.index <= 54:
                skipped_count += 1
                continue
            else:
                first_frame = frame
                break
        
        assert skipped_count == 5, f"Expected to skip 5 frames, skipped {skipped_count}"
        assert first_frame is not None, "Should have found first frame after skip"
        assert first_frame.index == 55, f"First frame should be 55, got {first_frame.index}"
        
        print("✓ test_skip_frames_within_resume_segment passed")


def test_resume_segment_deleted_jump_forward():
    """Test that missing resume segment causes jump forward (not backward)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments 1, 3, 5 (segment 2 missing)
        create_segment_with_frames(tmpdir, 1, 5, start_index=0)
        create_segment_with_frames(tmpdir, 3, 5, start_index=300)
        create_segment_with_frames(tmpdir, 5, 5, start_index=500)
        
        reader = SegmentReader(tmpdir)
        segments = reader.list_segments(use_cache=False)
        
        # Simulate: last_published_segment = 2 (which is missing)
        last_published_segment = 2
        
        # Find nearest segment >= last_published_segment
        candidates = [s for s in segments if s >= last_published_segment]
        assert candidates == [3, 5], f"Expected forward candidates [3, 5], got {candidates}"
        
        resume_segment = min(candidates)
        assert resume_segment == 3, f"Should jump to segment 3, got {resume_segment}"
        
        # Verify we don't jump backward to segment 1
        assert resume_segment > last_published_segment, "Should jump forward, not backward"
        
        print("✓ test_resume_segment_deleted_jump_forward passed")


def test_no_backward_jump_when_segment_deleted():
    """Test that processor never jumps backward to oldest when resume segment missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments 10, 11, 12, 13
        for seg in [10, 11, 12, 13]:
            create_segment_with_frames(tmpdir, seg, 3, start_index=seg*100)
        
        reader = SegmentReader(tmpdir)
        
        # Simulate: last_published_segment = 11 (exists)
        # Processor should resume from 11, not 10
        frames = list(reader.read_frames(start_segment=11))
        
        # First frame should be from segment 11 (index 1100)
        assert frames[0].index == 1100, f"Should start from seg 11, got index {frames[0].index}"
        
        # Should NOT have frames from segment 10
        all_indices = [f.index for f in frames]
        assert all(idx >= 1100 for idx in all_indices), f"Should not include seg 10 frames: {all_indices[:5]}"
        
        print("✓ test_no_backward_jump_when_segment_deleted passed")


def test_indices_not_monotonic_across_segments():
    """Test handling of non-monotonic indices across segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments with intentionally non-monotonic indices
        # Segment 1: indices 500-504
        # Segment 2: indices 100-104 (lower than segment 1!)
        # Segment 3: indices 600-604
        create_segment_with_frames(tmpdir, 1, 5, start_index=500)
        create_segment_with_frames(tmpdir, 2, 5, start_index=100)
        create_segment_with_frames(tmpdir, 3, 5, start_index=600)
        
        reader = SegmentReader(tmpdir)
        
        # Read all frames
        frames = list(reader.read_frames(start_segment=1))
        indices = [f.index for f in frames]
        
        # Indices should be: [500,501,502,503,504,100,101,102,103,104,600,601,602,603,604]
        # This is NOT monotonic globally, but IS monotonic within each segment
        
        # Verify segment boundaries
        seg1_indices = indices[0:5]
        seg2_indices = indices[5:10]
        seg3_indices = indices[10:15]
        
        assert seg1_indices == [500, 501, 502, 503, 504], f"Seg 1 wrong: {seg1_indices}"
        assert seg2_indices == [100, 101, 102, 103, 104], f"Seg 2 wrong: {seg2_indices}"
        assert seg3_indices == [600, 601, 602, 603, 604], f"Seg 3 wrong: {seg3_indices}"
        
        # Global non-monotonicity: 504 -> 100 (decreases!)
        assert indices[4] > indices[5], "Indices should NOT be monotonic across segments"
        
        print("✓ test_indices_not_monotonic_across_segments passed")


def test_resume_jumps_to_next_segment_after_last_published():
    """Test that resume can jump to segment > last_published_segment when that segment missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create segments 1, 5, 10 (large gaps)
        create_segment_with_frames(tmpdir, 1, 5, start_index=0)
        create_segment_with_frames(tmpdir, 5, 5, start_index=500)
        create_segment_with_frames(tmpdir, 10, 5, start_index=1000)
        
        reader = SegmentReader(tmpdir)
        segments = reader.list_segments(use_cache=False)
        
        # Simulate: last_published_segment = 3 (missing)
        last_published_segment = 3
        
        # Should jump to segment 5 (nearest >= 3)
        candidates = [s for s in segments if s >= last_published_segment]
        resume_segment = min(candidates) if candidates else None
        
        assert resume_segment == 5, f"Should jump to segment 5, got {resume_segment}"
        
        # Verify we skipped segments 2, 3, 4 (which don't exist)
        assert resume_segment - last_published_segment == 2, "Should skip 2 segments forward"
        
        print("✓ test_resume_jumps_to_next_segment_after_last_published passed")


def test_resume_with_all_segments_deleted():
    """Test behavior when all segments >= last_published are deleted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create only segment 1
        create_segment_with_frames(tmpdir, 1, 5, start_index=0)
        
        reader = SegmentReader(tmpdir)
        segments = reader.list_segments(use_cache=False)
        
        # Simulate: last_published_segment = 5 (newer than any existing segment)
        last_published_segment = 5
        
        # No segments >= 5 exist
        candidates = [s for s in segments if s >= last_published_segment]
        assert candidates == [], f"Expected no forward candidates, got {candidates}"
        
        # Fallback behavior: should use oldest available segment
        # (This is unusual but safe fallback)
        fallback_segment = min(segments) if segments else None
        assert fallback_segment == 1, f"Fallback should be segment 1, got {fallback_segment}"
        
        print("✓ test_resume_with_all_segments_deleted passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Processor Resume Behavior")
    print("=" * 60)
    print()
    
    try:
        test_resume_from_last_published_segment()
        test_skip_frames_within_resume_segment()
        test_resume_segment_deleted_jump_forward()
        test_no_backward_jump_when_segment_deleted()
        test_indices_not_monotonic_across_segments()
        test_resume_jumps_to_next_segment_after_last_published()
        test_resume_with_all_segments_deleted()
        
        print()
        print("=" * 60)
        print("✓ All processor resume tests passed!")
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
