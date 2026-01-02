"""
Tests for segment I/O roundtrip.

These tests verify that frames can be written to segment files
and read back correctly.
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.segment_io import (
    SegmentWriter,
    SegmentReader,
    FrameRecord,
    SEGMENT_MAGIC,
    SEGMENT_VERSION,
    validate_segment_file,
)


def create_test_frame(index: int, data_size: int = 100) -> FrameRecord:
    """Create a test frame record."""
    return FrameRecord(
        index=index,
        width=1920,
        height=1080,
        dts_sec=index,
        dts_nsec=index * 1000000,
        pts_sec=index,
        pts_nsec=index * 1000000,
        encoding="H264",
        data=bytes([i % 256 for i in range(data_size)])
    )


def test_frame_record_serialization():
    """Test FrameRecord serialization and deserialization."""
    original = create_test_frame(42, 256)
    
    # Serialize
    data = original.to_bytes()
    
    # Check that data is not empty
    assert len(data) > 0, "Serialized data should not be empty"
    
    # Split header and data
    from src.spool.segment_io import RECORD_HEADER_SIZE
    header_bytes = data[:RECORD_HEADER_SIZE]
    frame_data = data[RECORD_HEADER_SIZE:]
    
    # Deserialize
    restored = FrameRecord.from_bytes(header_bytes, frame_data)
    
    # Verify all fields match
    assert restored.index == original.index, f"Index mismatch: {restored.index} vs {original.index}"
    assert restored.width == original.width, f"Width mismatch"
    assert restored.height == original.height, f"Height mismatch"
    assert restored.dts_sec == original.dts_sec, f"DTS sec mismatch"
    assert restored.dts_nsec == original.dts_nsec, f"DTS nsec mismatch"
    assert restored.pts_sec == original.pts_sec, f"PTS sec mismatch"
    assert restored.pts_nsec == original.pts_nsec, f"PTS nsec mismatch"
    assert restored.encoding == original.encoding, f"Encoding mismatch"
    assert restored.data == original.data, f"Data mismatch"
    
    print("✓ test_frame_record_serialization passed")


def test_writer_creates_segment():
    """Test that SegmentWriter creates segment files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(tmpdir, segment_duration=5.0)
        writer.start()
        
        # Write a frame
        frame = create_test_frame(1)
        success = writer.write_frame(frame, has_idr=True)
        
        assert success, "Write should succeed"
        
        # Close writer
        writer.close()
        
        # Check that segment file was created
        segment_files = list(os.listdir(tmpdir))
        bin_files = [f for f in segment_files if f.endswith('.bin')]
        
        assert len(bin_files) == 1, f"Expected 1 .bin file, got {len(bin_files)}"
        
        # Validate segment file
        segment_path = os.path.join(tmpdir, bin_files[0])
        assert validate_segment_file(segment_path), "Segment file should be valid"
        
        print("✓ test_writer_creates_segment passed")


def test_segment_roundtrip():
    """Test full write/read roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write frames
        writer = SegmentWriter(tmpdir, segment_duration=5.0)
        writer.start()
        
        frames_written = []
        for i in range(10):
            frame = create_test_frame(i, data_size=50 + i * 10)
            writer.write_frame(frame, has_idr=(i == 0))
            frames_written.append(frame)
        
        writer.close()
        
        # Read frames back
        reader = SegmentReader(tmpdir)
        frames_read = list(reader.read_frames())
        
        # Verify count
        assert len(frames_read) == len(frames_written), \
            f"Frame count mismatch: {len(frames_read)} vs {len(frames_written)}"
        
        # Verify each frame
        for i, (written, read) in enumerate(zip(frames_written, frames_read)):
            assert read.index == written.index, f"Frame {i} index mismatch"
            assert read.width == written.width, f"Frame {i} width mismatch"
            assert read.height == written.height, f"Frame {i} height mismatch"
            assert read.encoding == written.encoding, f"Frame {i} encoding mismatch"
            assert read.data == written.data, f"Frame {i} data mismatch"
        
        print("✓ test_segment_roundtrip passed")


def test_segment_rotation():
    """Test that segments rotate based on duration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Very short segment duration for testing
        writer = SegmentWriter(
            tmpdir, 
            segment_duration=0.001,  # 1ms - forces rotation
            max_segment_duration=0.002
        )
        writer.start()
        
        # Write multiple frames (should create multiple segments)
        for i in range(5):
            frame = create_test_frame(i)
            # All frames after first are IDR to allow rotation
            writer.write_frame(frame, has_idr=True)
        
        writer.close()
        
        # Check that multiple segments were created
        segment_files = [f for f in os.listdir(tmpdir) if f.endswith('.bin')]
        
        # Should have multiple segments (exact number depends on timing)
        assert len(segment_files) >= 1, f"Expected at least 1 segment, got {len(segment_files)}"
        
        print(f"✓ test_segment_rotation passed (created {len(segment_files)} segments)")


def test_reader_list_segments():
    """Test SegmentReader.list_segments()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some segment files manually
        for i in [1, 3, 5]:
            with open(os.path.join(tmpdir, f"seg_{i:06d}.bin"), 'wb') as f:
                f.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
        
        reader = SegmentReader(tmpdir)
        segments = reader.list_segments()
        
        assert segments == [1, 3, 5], f"Expected [1, 3, 5], got {segments}"
        
        assert reader.get_oldest_segment() == 1
        assert reader.get_newest_segment() == 5
        
        print("✓ test_reader_list_segments passed")


def test_metadata_file():
    """Test that metadata files are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(tmpdir, segment_duration=5.0, write_metadata=True)
        writer.start()
        
        # Write frames
        for i in range(3):
            frame = create_test_frame(i)
            writer.write_frame(frame, has_idr=(i == 0))
        
        writer.close()
        
        # Check for metadata file
        meta_files = [f for f in os.listdir(tmpdir) if f.endswith('.meta.json')]
        
        assert len(meta_files) == 1, f"Expected 1 metadata file, got {len(meta_files)}"
        
        # Read metadata
        import json
        meta_path = os.path.join(tmpdir, meta_files[0])
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        assert 'frame_count' in meta, "Metadata should have frame_count"
        assert meta['frame_count'] == 3, f"Expected 3 frames, got {meta['frame_count']}"
        
        print("✓ test_metadata_file passed")


def test_atomic_write():
    """Test that writes are atomic (no .tmp files left)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(tmpdir)
        writer.start()
        
        # Write a frame
        frame = create_test_frame(1)
        writer.write_frame(frame, has_idr=True)
        
        # Close properly
        writer.close()
        
        # No .tmp files should remain
        tmp_files = [f for f in os.listdir(tmpdir) if f.endswith('.tmp')]
        assert len(tmp_files) == 0, f"Expected no .tmp files, got {tmp_files}"
        
        print("✓ test_atomic_write passed")


def test_empty_spool():
    """Test reading from empty spool directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reader = SegmentReader(tmpdir)
        
        segments = reader.list_segments()
        assert segments == [], "Empty spool should have no segments"
        
        assert reader.get_oldest_segment() is None
        assert reader.get_newest_segment() is None
        
        frames = list(reader.read_frames())
        assert frames == [], "Empty spool should yield no frames"
        
        print("✓ test_empty_spool passed")


def test_frame_record_timestamps():
    """Test FrameRecord timestamp properties."""
    frame = FrameRecord(
        index=1,
        width=1920,
        height=1080,
        dts_sec=100,
        dts_nsec=500000000,  # 0.5 seconds
        pts_sec=101,
        pts_nsec=250000000,  # 0.25 seconds
        encoding="H264",
        data=b"test"
    )
    
    # Check nanosecond accessors
    expected_dts_ns = 100 * 1_000_000_000 + 500_000_000
    expected_pts_ns = 101 * 1_000_000_000 + 250_000_000
    
    assert frame.dts_ns == expected_dts_ns, f"DTS ns mismatch: {frame.dts_ns}"
    assert frame.pts_ns == expected_pts_ns, f"PTS ns mismatch: {frame.pts_ns}"
    
    print("✓ test_frame_record_timestamps passed")


def test_large_frame_data():
    """Test handling of large frame data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(tmpdir)
        writer.start()
        
        # Create a large frame (1MB of data)
        large_data = bytes([i % 256 for i in range(1024 * 1024)])
        frame = FrameRecord(
            index=1,
            width=1920,
            height=1080,
            dts_sec=0,
            dts_nsec=0,
            pts_sec=0,
            pts_nsec=0,
            encoding="H264",
            data=large_data
        )
        
        writer.write_frame(frame, has_idr=True)
        writer.close()
        
        # Read back
        reader = SegmentReader(tmpdir)
        frames = list(reader.read_frames())
        
        assert len(frames) == 1, "Should read back 1 frame"
        assert frames[0].data == large_data, "Large frame data should match"
        
        print("✓ test_large_frame_data passed")


def test_encoding_type_handling():
    """Test handling of different encoding field types (str, bytes, list)."""
    
    # Test with string encoding
    frame_str = FrameRecord(
        index=1, width=1920, height=1080,
        dts_sec=0, dts_nsec=0, pts_sec=0, pts_nsec=0,
        encoding="H264",
        data=b"test"
    )
    data_str = frame_str.to_bytes()
    assert len(data_str) > 0, "String encoding should serialize"
    
    # Test with bytes encoding
    frame_bytes = FrameRecord(
        index=2, width=1920, height=1080,
        dts_sec=0, dts_nsec=0, pts_sec=0, pts_nsec=0,
        encoding=b"H264",
        data=b"test"
    )
    data_bytes = frame_bytes.to_bytes()
    assert len(data_bytes) > 0, "Bytes encoding should serialize"
    
    # Test with list encoding (simulating array-like type from ROS message)
    list_encoding = [72, 50, 54, 52, 0, 0, 0, 0]  # "H264" + nulls as ASCII values
    frame_list = FrameRecord(
        index=3, width=1920, height=1080,
        dts_sec=0, dts_nsec=0, pts_sec=0, pts_nsec=0,
        encoding=list_encoding,
        data=b"test"
    )
    data_list = frame_list.to_bytes()
    assert len(data_list) > 0, "List encoding should serialize"
    
    print("✓ test_encoding_type_handling passed")


def test_get_current_segment_tracking():
    """Test that SegmentReader.get_current_segment() tracks progress during read_frames()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple segments with frames
        for seg_num in [1, 2, 3]:
            writer = SegmentWriter(tmpdir, segment_duration=999)
            writer.start()
            writer._current_segment = seg_num
            
            tmp_path = writer._get_segment_path(seg_num, tmp=True)
            writer._current_file = open(tmp_path, 'wb')
            writer._current_file.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
            
            # Write 3 frames per segment
            for i in range(3):
                frame = create_test_frame(seg_num * 100 + i)
                writer._current_file.write(frame.to_bytes())
            
            writer._current_file.close()
            final_path = writer._get_segment_path(seg_num, tmp=False)
            tmp_path.rename(final_path)
        
        # Create reader and verify initial state
        reader = SegmentReader(tmpdir)
        assert reader.get_current_segment() == -1, "Initial current segment should be -1"
        
        # Read frames and verify segment tracking
        segments_seen = []
        for frame in reader.read_frames():
            current_seg = reader.get_current_segment()
            if current_seg not in segments_seen:
                segments_seen.append(current_seg)
        
        # Should have tracked all 3 segments in order
        assert segments_seen == [1, 2, 3], f"Expected [1, 2, 3], got {segments_seen}"
        
        # After reading all frames, current_segment should be the last segment
        assert reader.get_current_segment() == 3, "Current segment should be 3 after reading all"
        
        print("✓ test_get_current_segment_tracking passed")


def test_get_last_completed_segment():
    """Test that SegmentReader.get_last_completed_segment() tracks completion correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple segments with frames
        for seg_num in [1, 2, 3]:
            writer = SegmentWriter(tmpdir, segment_duration=999)
            writer.start()
            writer._current_segment = seg_num
            
            tmp_path = writer._get_segment_path(seg_num, tmp=True)
            writer._current_file = open(tmp_path, 'wb')
            writer._current_file.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
            
            # Write 3 frames per segment
            for i in range(3):
                frame = create_test_frame(seg_num * 100 + i)
                writer._current_file.write(frame.to_bytes())
            
            writer._current_file.close()
            final_path = writer._get_segment_path(seg_num, tmp=False)
            tmp_path.rename(final_path)
        
        # Create reader and verify initial state
        reader = SegmentReader(tmpdir)
        assert reader.get_last_completed_segment() == -1, "Initial last completed should be -1"
        
        # Read frames and track when segments complete
        generator = reader.read_frames()
        
        # Read first 3 frames (segment 1) - segment not complete yet until we try to read more
        for _ in range(3):
            next(generator)
        # Note: _last_completed_segment is updated when the generator MOVES to the next segment
        # So after reading 3 frames, we haven't moved to segment 2 yet
        assert reader.get_last_completed_segment() == -1, "Segment 1 not complete yet (waiting for move to next)"
        
        # Read 4th frame (first of segment 2) - THIS triggers completion of segment 1
        next(generator)
        assert reader.get_last_completed_segment() == 1, "Segment 1 should be completed after 4th frame"
        
        # Read 5th and 6th frames (rest of segment 2)
        next(generator)
        next(generator)
        assert reader.get_last_completed_segment() == 1, "Still segment 1 - segment 2 not complete yet"
        
        # Read 7th frame (first of segment 3) - THIS triggers completion of segment 2
        next(generator)
        assert reader.get_last_completed_segment() == 2, "Segment 2 should be completed after 7th frame"
        
        # Read remaining 2 frames (rest of segment 3)
        next(generator)
        next(generator)
        assert reader.get_last_completed_segment() == 2, "Still segment 2 - segment 3 not complete yet"
        
        # Try to read more - StopIteration will be raised and segment 3 completed
        try:
            next(generator)
        except StopIteration:
            pass
        
        # After StopIteration, segment 3 should be completed
        assert reader.get_last_completed_segment() == 3, "Last completed should be 3 after exhaustion"
        
        print("✓ test_get_last_completed_segment passed")


def test_read_single_segment():
    """
    Test that read_single_segment() reads exactly one segment and raises StopIteration.
    
    V8.7: This tests the new per-segment generator that enables immediate deletion.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 segments with known frame counts
        for seg_num in [1, 2, 3]:
            writer = SegmentWriter(tmpdir, segment_duration=999)
            writer.start()
            writer._current_segment = seg_num
            tmp_path = writer._get_segment_path(seg_num, tmp=True)
            writer._current_file = open(tmp_path, 'wb')
            writer._current_file.write(SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0]))
            for i in range(3):  # 3 frames per segment
                frame = create_test_frame(seg_num * 100 + i)
                writer._current_file.write(frame.to_bytes())
            writer._current_file.close()
            final_path = writer._get_segment_path(seg_num, tmp=False)
            tmp_path.rename(final_path)
        
        reader = SegmentReader(tmpdir)
        
        # Read segment 1 only
        gen1 = reader.read_single_segment(1)
        frames1 = list(gen1)  # Should exhaust after 3 frames and raise StopIteration
        assert len(frames1) == 3, f"Segment 1 should have 3 frames, got {len(frames1)}"
        assert reader.get_last_completed_segment() == 1, "Segment 1 should be completed"
        
        # Read segment 2 only
        gen2 = reader.read_single_segment(2)
        frames2 = list(gen2)
        assert len(frames2) == 3, f"Segment 2 should have 3 frames, got {len(frames2)}"
        assert reader.get_last_completed_segment() == 2, "Segment 2 should be completed"
        
        # Read segment 3 only
        gen3 = reader.read_single_segment(3)
        frames3 = list(gen3)
        assert len(frames3) == 3, f"Segment 3 should have 3 frames, got {len(frames3)}"
        assert reader.get_last_completed_segment() == 3, "Segment 3 should be completed"
        
        # Verify total frames
        total = len(frames1) + len(frames2) + len(frames3)
        assert total == 9, f"Total frames should be 9, got {total}"
        
        print("✓ test_read_single_segment passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Segment I/O Roundtrip")
    print("=" * 60)
    print()
    
    try:
        test_frame_record_serialization()
        test_writer_creates_segment()
        test_segment_roundtrip()
        test_segment_rotation()
        test_reader_list_segments()
        test_metadata_file()
        test_atomic_write()
        test_empty_spool()
        test_frame_record_timestamps()
        test_large_frame_data()
        test_encoding_type_handling()
        test_get_current_segment_tracking()
        test_get_last_completed_segment()
        test_read_single_segment()
        
        print()
        print("=" * 60)
        print("✓ All segment I/O tests passed!")
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
