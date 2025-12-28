"""
Segment I/O Module for H.264 Frame Spooling.

Provides writer and reader classes for binary segment files that store
H.264 frames with metadata in a compact format suitable for 24/7 operation.

Segment File Format (Version 1):
================================
Header:
  - Magic bytes: "SPOOL1" (6 bytes)
  - Version: uint8 (1 byte)
  - Flags: uint8 (1 byte, reserved)

Each Record:
  - Magic: "FR" (2 bytes, record marker)
  - Index: uint32 (frame index from source)
  - Width: uint32
  - Height: uint32
  - DTS seconds: int64
  - DTS nanoseconds: uint32
  - PTS seconds: int64
  - PTS nanoseconds: uint32
  - Encoding: 12 bytes (null-padded string, e.g., "H264")
  - Data length: uint32
  - Data: raw bytes

Atomic Writes:
==============
Files are written with .tmp extension and atomically renamed to .bin
when complete. This prevents corruption during crashes and allows
safe retention policy operation.
"""

import os
import struct
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Dict, Any
from pathlib import Path
import threading

from src.utils.AppLogging import logger


# Segment file constants
SEGMENT_MAGIC = b"SPOOL1"
SEGMENT_VERSION = 1
SEGMENT_HEADER_SIZE = 8  # 6 (magic) + 1 (version) + 1 (flags)

# Record constants
RECORD_MAGIC = b"FR"
RECORD_HEADER_SIZE = 54  # 2 + 4 + 4 + 4 + 8 + 4 + 8 + 4 + 12 + 4 = 54 bytes

# Record header struct format
# < = little-endian
# 2s = record magic (2 bytes)
# I = index (uint32)
# I = width (uint32)
# I = height (uint32)
# q = dts_sec (int64)
# I = dts_nsec (uint32)
# q = pts_sec (int64)
# I = pts_nsec (uint32)
# 12s = encoding (12 bytes, null-padded)
# I = data_len (uint32)
RECORD_STRUCT = struct.Struct("<2sIIIqIqI12sI")


@dataclass
class FrameRecord:
    """
    Represents a single frame record in a segment file.
    
    Attributes:
        index: Frame index from the original source
        width: Frame width in pixels
        height: Frame height in pixels
        dts_sec: Decode timestamp seconds
        dts_nsec: Decode timestamp nanoseconds
        pts_sec: Presentation timestamp seconds
        pts_nsec: Presentation timestamp nanoseconds
        encoding: Encoding type string (e.g., "H264", "H265")
        data: Raw encoded frame data
    """
    index: int
    width: int
    height: int
    dts_sec: int
    dts_nsec: int
    pts_sec: int
    pts_nsec: int
    encoding: str
    data: bytes
    
    def to_bytes(self) -> bytes:
        """Serialize the frame record to bytes."""
        encoding_bytes = self.encoding.encode('utf-8')[:12].ljust(12, b'\x00')
        header = RECORD_STRUCT.pack(
            RECORD_MAGIC,
            self.index,
            self.width,
            self.height,
            self.dts_sec,
            self.dts_nsec,
            self.pts_sec,
            self.pts_nsec,
            encoding_bytes,
            len(self.data)
        )
        return header + self.data
    
    @classmethod
    def from_bytes(cls, header_bytes: bytes, data: bytes) -> 'FrameRecord':
        """Deserialize a frame record from bytes."""
        (
            magic, index, width, height,
            dts_sec, dts_nsec, pts_sec, pts_nsec,
            encoding_bytes, data_len
        ) = RECORD_STRUCT.unpack(header_bytes)
        
        if magic != RECORD_MAGIC:
            raise ValueError(f"Invalid record magic: {magic!r}")
        
        encoding = encoding_bytes.rstrip(b'\x00').decode('utf-8')
        
        return cls(
            index=index,
            width=width,
            height=height,
            dts_sec=dts_sec,
            dts_nsec=dts_nsec,
            pts_sec=pts_sec,
            pts_nsec=pts_nsec,
            encoding=encoding,
            data=data
        )
    
    @property
    def dts_ns(self) -> int:
        """Get DTS as total nanoseconds."""
        return self.dts_sec * 1_000_000_000 + self.dts_nsec
    
    @property
    def pts_ns(self) -> int:
        """Get PTS as total nanoseconds."""
        return self.pts_sec * 1_000_000_000 + self.pts_nsec


@dataclass
class SegmentMetadata:
    """
    Metadata for a segment file (stored in .meta.json).
    
    Attributes:
        segment_number: Segment sequence number
        start_time: Unix timestamp when segment started
        end_time: Unix timestamp when segment ended
        frame_count: Number of frames in segment
        bytes_written: Total bytes written to segment
        first_frame_index: Index of first frame
        last_frame_index: Index of last frame
        has_idr: Whether segment starts with IDR frame
    """
    segment_number: int
    start_time: float
    end_time: float = 0.0
    frame_count: int = 0
    bytes_written: int = 0
    first_frame_index: int = 0
    last_frame_index: int = 0
    has_idr: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'segment_number': self.segment_number,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'frame_count': self.frame_count,
            'bytes_written': self.bytes_written,
            'first_frame_index': self.first_frame_index,
            'last_frame_index': self.last_frame_index,
            'has_idr': self.has_idr,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SegmentMetadata':
        """Create from dictionary."""
        return cls(
            segment_number=data['segment_number'],
            start_time=data['start_time'],
            end_time=data.get('end_time', 0.0),
            frame_count=data.get('frame_count', 0),
            bytes_written=data.get('bytes_written', 0),
            first_frame_index=data.get('first_frame_index', 0),
            last_frame_index=data.get('last_frame_index', 0),
            has_idr=data.get('has_idr', False),
        )


class SegmentWriter:
    """
    Writes H.264 frames to segment files with atomic completion.
    
    Features:
    - Atomic writes using .tmp -> .bin rename
    - Segment rotation based on duration
    - IDR-aligned rotation when possible
    - Metadata file generation
    - Thread-safe operations
    
    Usage:
        writer = SegmentWriter('/path/to/spool', segment_duration=5.0)
        writer.start()
        
        for frame in frames:
            writer.write_frame(frame_record)
        
        writer.close()
    """
    
    def __init__(
        self,
        spool_dir: str,
        segment_duration: float = 5.0,
        max_segment_duration: float = 10.0,
        write_metadata: bool = True
    ):
        """
        Initialize the segment writer.
        
        Args:
            spool_dir: Directory to write segment files
            segment_duration: Target segment duration in seconds
            max_segment_duration: Maximum segment duration (hard limit)
            write_metadata: Whether to write .meta.json files
        """
        self.spool_dir = Path(spool_dir)
        self.segment_duration = segment_duration
        self.max_segment_duration = max_segment_duration
        self.write_metadata = write_metadata
        
        self._lock = threading.Lock()
        self._current_file: Optional[Any] = None
        self._current_segment: int = 0
        self._current_metadata: Optional[SegmentMetadata] = None
        self._segment_start_time: float = 0.0
        self._waiting_for_idr: bool = False
        self._cached_sps: Optional[bytes] = None
        self._cached_pps: Optional[bytes] = None
        
        # Statistics
        self.total_bytes_written: int = 0
        self.total_frames_written: int = 0
        self.segments_completed: int = 0
    
    def start(self):
        """Initialize the writer and create spool directory."""
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._find_next_segment_number()
        logger.info(f"[SegmentWriter] Started. Spool dir: {self.spool_dir}, "
                   f"starting at segment {self._current_segment}")
    
    def _find_next_segment_number(self):
        """Find the next available segment number."""
        max_num = 0
        for f in self.spool_dir.glob("seg_*.bin"):
            try:
                num = int(f.stem.split('_')[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                continue
        self._current_segment = max_num + 1
    
    def _get_segment_path(self, segment_num: int, tmp: bool = False) -> Path:
        """Get path for a segment file."""
        ext = ".tmp" if tmp else ".bin"
        return self.spool_dir / f"seg_{segment_num:06d}{ext}"
    
    def _get_metadata_path(self, segment_num: int) -> Path:
        """Get path for segment metadata file."""
        return self.spool_dir / f"seg_{segment_num:06d}.meta.json"
    
    def _open_new_segment(self):
        """Open a new segment file."""
        tmp_path = self._get_segment_path(self._current_segment, tmp=True)
        self._current_file = open(tmp_path, 'wb')
        
        # Write segment header
        header = SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0])
        self._current_file.write(header)
        
        self._segment_start_time = time.time()
        self._current_metadata = SegmentMetadata(
            segment_number=self._current_segment,
            start_time=self._segment_start_time
        )
        
        logger.info(f"[SegmentWriter] Opened segment {self._current_segment}")
    
    def _close_current_segment(self):
        """Close and finalize the current segment."""
        if self._current_file is None:
            return
        
        self._current_file.flush()
        os.fsync(self._current_file.fileno())
        self._current_file.close()
        
        # Atomic rename from .tmp to .bin
        tmp_path = self._get_segment_path(self._current_segment, tmp=True)
        final_path = self._get_segment_path(self._current_segment, tmp=False)
        tmp_path.rename(final_path)
        
        # Write metadata
        if self.write_metadata and self._current_metadata:
            self._current_metadata.end_time = time.time()
            meta_path = self._get_metadata_path(self._current_segment)
            with open(meta_path, 'w') as f:
                json.dump(self._current_metadata.to_dict(), f, indent=2)
        
        logger.info(f"[SegmentWriter] Closed segment {self._current_segment}: "
                   f"{self._current_metadata.frame_count} frames, "
                   f"{self._current_metadata.bytes_written} bytes")
        
        self.segments_completed += 1
        self._current_segment += 1
        self._current_file = None
        self._current_metadata = None
        self._waiting_for_idr = False
    
    def _should_rotate(self, has_idr: bool) -> bool:
        """Check if segment should be rotated."""
        if self._current_file is None:
            return False
        
        elapsed = time.time() - self._segment_start_time
        
        # Hard limit: always rotate
        if elapsed >= self.max_segment_duration:
            return True
        
        # Soft limit: rotate on IDR
        if elapsed >= self.segment_duration:
            if has_idr:
                return True
            # Mark that we're waiting for IDR
            self._waiting_for_idr = True
        
        return False
    
    def write_frame(self, record: FrameRecord, has_idr: bool = False) -> bool:
        """
        Write a frame record to the current segment.
        
        Args:
            record: The frame record to write
            has_idr: Whether this frame contains an IDR
            
        Returns:
            True if successful, False on error
        """
        with self._lock:
            try:
                # Check for rotation
                if self._should_rotate(has_idr):
                    self._close_current_segment()
                
                # Open new segment if needed
                if self._current_file is None:
                    self._open_new_segment()
                    
                    # Prepend cached SPS/PPS if we have them and this isn't an IDR
                    if not has_idr and self._cached_sps and self._cached_pps:
                        # Note: We don't write SPS/PPS as separate records,
                        # they should be included in the frame data itself
                        pass
                
                # Write the record
                record_bytes = record.to_bytes()
                self._current_file.write(record_bytes)
                
                # Update statistics
                self.total_bytes_written += len(record_bytes)
                self.total_frames_written += 1
                
                if self._current_metadata:
                    self._current_metadata.frame_count += 1
                    self._current_metadata.bytes_written += len(record_bytes)
                    if self._current_metadata.frame_count == 1:
                        self._current_metadata.first_frame_index = record.index
                        self._current_metadata.has_idr = has_idr
                    self._current_metadata.last_frame_index = record.index
                
                return True
                
            except Exception as e:
                logger.error(f"[SegmentWriter] Error writing frame: {e}")
                return False
    
    def update_sps_pps(self, sps: Optional[bytes], pps: Optional[bytes]):
        """Update cached SPS/PPS for segment boundary insertion."""
        with self._lock:
            if sps:
                self._cached_sps = sps
            if pps:
                self._cached_pps = pps
    
    def flush(self):
        """Flush the current file to disk."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
    
    def close(self):
        """Close the writer and finalize any open segment."""
        with self._lock:
            if self._current_file:
                self._close_current_segment()
        logger.info(f"[SegmentWriter] Closed. Total: {self.total_frames_written} frames, "
                   f"{self.total_bytes_written} bytes, {self.segments_completed} segments")


class SegmentReader:
    """
    Reads H.264 frames from segment files.
    
    Supports:
    - Sequential reading of segment files in order
    - Iterator interface for frame-by-frame access
    - Automatic segment progression
    - Metadata reading
    
    Usage:
        reader = SegmentReader('/path/to/spool')
        for record in reader.read_frames():
            process(record)
    """
    
    def __init__(self, spool_dir: str):
        """
        Initialize the segment reader.
        
        Args:
            spool_dir: Directory containing segment files
        """
        self.spool_dir = Path(spool_dir)
        self._current_file: Optional[Any] = None
        self._current_segment: int = 0
    
    def list_segments(self) -> List[int]:
        """
        List all available segment numbers (completed .bin files only).
        
        Returns:
            Sorted list of segment numbers
        """
        segments = []
        for f in self.spool_dir.glob("seg_*.bin"):
            try:
                num = int(f.stem.split('_')[1])
                segments.append(num)
            except (IndexError, ValueError):
                continue
        return sorted(segments)
    
    def get_oldest_segment(self) -> Optional[int]:
        """Get the oldest available segment number."""
        segments = self.list_segments()
        return segments[0] if segments else None
    
    def get_newest_segment(self) -> Optional[int]:
        """Get the newest available segment number."""
        segments = self.list_segments()
        return segments[-1] if segments else None
    
    def read_segment_metadata(self, segment_num: int) -> Optional[SegmentMetadata]:
        """Read metadata for a specific segment."""
        meta_path = self.spool_dir / f"seg_{segment_num:06d}.meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, 'r') as f:
                return SegmentMetadata.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"[SegmentReader] Error reading metadata for segment {segment_num}: {e}")
            return None
    
    def read_segment(self, segment_num: int) -> Iterator[FrameRecord]:
        """
        Read all frames from a specific segment.
        
        Args:
            segment_num: Segment number to read
            
        Yields:
            FrameRecord objects
        """
        path = self.spool_dir / f"seg_{segment_num:06d}.bin"
        if not path.exists():
            logger.warning(f"[SegmentReader] Segment {segment_num} not found")
            return
        
        try:
            with open(path, 'rb') as f:
                # Read and verify header
                header = f.read(SEGMENT_HEADER_SIZE)
                if len(header) < SEGMENT_HEADER_SIZE:
                    logger.error(f"[SegmentReader] Truncated header in segment {segment_num}")
                    return
                
                if header[:6] != SEGMENT_MAGIC:
                    logger.error(f"[SegmentReader] Invalid magic in segment {segment_num}")
                    return
                
                version = header[6]
                if version != SEGMENT_VERSION:
                    logger.warning(f"[SegmentReader] Unknown version {version} in segment {segment_num}")
                
                # Read records
                while True:
                    record_header = f.read(RECORD_HEADER_SIZE)
                    if len(record_header) < RECORD_HEADER_SIZE:
                        break  # End of file
                    
                    # Extract data length from header (at offset 50)
                    # Offset: 2+4+4+4+8+4+8+4+12 = 50
                    data_len = struct.unpack_from("<I", record_header, 50)[0]
                    data = f.read(data_len)
                    
                    if len(data) < data_len:
                        logger.error(f"[SegmentReader] Truncated record in segment {segment_num}")
                        break
                    
                    try:
                        record = FrameRecord.from_bytes(record_header, data)
                        yield record
                    except Exception as e:
                        logger.error(f"[SegmentReader] Error parsing record: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"[SegmentReader] Error reading segment {segment_num}: {e}")
    
    def read_frames(self, start_segment: Optional[int] = None) -> Iterator[FrameRecord]:
        """
        Read frames from all segments in order.
        
        Args:
            start_segment: Optional segment number to start from
            
        Yields:
            FrameRecord objects in chronological order
        """
        segments = self.list_segments()
        
        if start_segment is not None:
            segments = [s for s in segments if s >= start_segment]
        
        for seg_num in segments:
            logger.debug(f"[SegmentReader] Reading segment {seg_num}")
            yield from self.read_segment(seg_num)


def validate_segment_file(path: str) -> bool:
    """
    Validate a segment file's integrity.
    
    Args:
        path: Path to segment file
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(SEGMENT_HEADER_SIZE)
            if len(header) < SEGMENT_HEADER_SIZE:
                return False
            if header[:6] != SEGMENT_MAGIC:
                return False
            return True
    except Exception:
        return False
