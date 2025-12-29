"""
Spool module for H.264 frame storage and retrieval.

This module provides disk-based spooling of encoded H.264 frames
for the Accuracy Mode feature, enabling pull-based replay with
backpressure control.
"""

from src.spool.h264_nal import (
    NALUnitType,
    find_start_codes,
    parse_nal_units,
    detect_frame_type,
    extract_sps_pps,
    is_idr_frame,
)

from src.spool.segment_io import (
    SegmentWriter,
    SegmentReader,
    FrameRecord,
    SEGMENT_MAGIC,
    SEGMENT_VERSION,
)

from src.spool.retention import (
    RetentionPolicy,
    cleanup_stale_tmp_files,
)

__all__ = [
    # NAL parsing
    'NALUnitType',
    'find_start_codes',
    'parse_nal_units',
    'detect_frame_type',
    'extract_sps_pps',
    'is_idr_frame',
    # Segment I/O
    'SegmentWriter',
    'SegmentReader',
    'FrameRecord',
    'SEGMENT_MAGIC',
    'SEGMENT_VERSION',
    # Retention
    'RetentionPolicy',
    'cleanup_stale_tmp_files',
]
