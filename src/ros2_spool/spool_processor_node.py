#!/usr/bin/env python3
"""
Spool Processor Node for ACK-Free Video Processing.

This node reads H.264 frames from the spool and publishes them to the decoder
at a controlled rate without waiting for acknowledgments.

ACK-Free Architecture:
---------------------
1. Reads frames continuously from spool
2. Publishes at a controlled rate (target_fps)
3. Never blocks on consumer feedback
4. Relies on retention guards to protect unprocessed data

This aligns with industry-standard streaming architectures (Kafka, GStreamer, DeepStream).

Usage:
    python -m src.ros2_spool.spool_processor_node

Configuration (via database config table):
    spool_dir: Directory for spool files (default: /home/sunrise/BreadCounting/data/spool)
    spool_target_fps: Target FPS for publishing (default: 40.0)
"""

import os
import sys
import time
import signal
import threading
import itertools
from typing import Optional, Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.config.settings import AppConfig

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK
from src.spool.segment_io import SegmentReader, FrameRecord
from src.spool.h264_nal import extract_sps_pps, is_idr_frame, detect_frame_type
from src.spool.spool_utils import (
    calculate_crc32,
    save_processor_state,
    load_processor_state,
    ProcessorState,
    format_structured_log,
    throttled_log
)
from src.spool.retention import RetentionPolicy  # V8: For segment deletion after processing
from src.logging.Database import DatabaseManager
from src import constants

# Import message definitions (minimal - only what we need for ACK-free mode)
from src.ros2_spool.messages import (
    generate_session_id
)

# ROS2 imports (only on RDK platform)
if IS_RDK:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from img_msgs.msg import H26XFrame
    from builtin_interfaces.msg import Time
else:
    # Stub for non-RDK development
    class Node:
        def __init__(self, name): pass
        def get_logger(self): return logger
        def create_subscription(self, *args, **kwargs): pass
        def create_publisher(self, *args, **kwargs): return MockPublisher()
        def destroy_node(self): pass
    
    class MockPublisher:
        def publish(self, msg): pass


class ProcessorRunState(Enum):
    """Runtime state of the processor (ACK-free mode).
    
    Note: Named ProcessorRunState to avoid conflict with ProcessorState dataclass
    from spool_utils which is used for persistence.
    """
    IDLE = "idle"
    PUBLISHING = "publishing"  # Continuously publishing frames
    SPOOL_EMPTY = "spool_empty"
    STOPPED = "stopped"


# Default configuration values
DEFAULT_SPOOL_DIR = "/home/sunrise/BreadCounting/data/spool"
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_STATS_INTERVAL = 10.0
DEFAULT_SPS_PPS_PREPEND = True  # Prepend cached SPS/PPS to first frame of segment
DEFAULT_TARGET_FPS = 30.0  # V8.1: Increased from 20 to 30 FPS to keep up with recorder
DEFAULT_STATE_FILE = "processor_state.json"  # Relative to spool_dir
DEFAULT_SPOOL_LAG_WARN_THRESHOLD = 5  # Segments
DEFAULT_SPOOL_LAG_ERROR_THRESHOLD = 10  # Segments
DEFAULT_WATCHDOG_TIMEOUT = 30.0  # Seconds without publishing before alert
DEFAULT_ENABLE_ADAPTIVE_PACING = True  # V8: Enable adaptive pacing by default
DEFAULT_ADAPTIVE_FPS_MIN = 20.0  # V8.1: Increased from 15 to 20 FPS minimum
DEFAULT_ENABLE_CRC32_LOGGING = False  # Add CRC32 checksums to logs
ADAPTIVE_FPS_REDUCTION_FACTOR = 0.9  # V8.1: Less aggressive reduction (was 0.8)
# V8: Segment deletion and pacing control
DEFAULT_DELETE_PROCESSED_SEGMENTS = True  # Delete segments after processing to save disk space
DEFAULT_MIN_FRAME_INTERVAL_MS = 20.0  # V8.1: Reduced from 30ms to 10ms - 30ms was too slow

# Add new constants at the top (around line 94-111)
DEFAULT_SPOOL_LAG_HEALTHY_THRESHOLD = 5  # Less than this = healthy, relax
DEFAULT_SPOOL_LAG_NORMAL_THRESHOLD = 15  # Between 5-15 = normal pace
# Above 15 = high lag, speed up

DEFAULT_ADAPTIVE_FPS_RELAXED = 20.0  # Healthy state - save resources
DEFAULT_ADAPTIVE_FPS_MAX = 50.0  # High lag state - catch up (15ms min interval)

@dataclass
class ProcessorConfig:
    """Configuration for the spool processor (ACK-free mode only)."""
    spool_dir_path: str = DEFAULT_SPOOL_DIR
    poll_interval: float = DEFAULT_POLL_INTERVAL
    stats_interval: float = DEFAULT_STATS_INTERVAL
    prepend_sps_pps: bool = DEFAULT_SPS_PPS_PREPEND
    target_fps: float = DEFAULT_TARGET_FPS
    # V7: Robustness and observability
    state_file: str = DEFAULT_STATE_FILE
    spool_lag_warn_threshold: int = DEFAULT_SPOOL_LAG_WARN_THRESHOLD
    spool_lag_error_threshold: int = DEFAULT_SPOOL_LAG_ERROR_THRESHOLD
    watchdog_timeout: float = DEFAULT_WATCHDOG_TIMEOUT
    enable_adaptive_pacing: bool = DEFAULT_ENABLE_ADAPTIVE_PACING
    adaptive_fps_min: float = DEFAULT_ADAPTIVE_FPS_MIN
    enable_crc32_logging: bool = DEFAULT_ENABLE_CRC32_LOGGING
    # V8: Segment deletion and pacing control
    delete_processed_segments: bool = DEFAULT_DELETE_PROCESSED_SEGMENTS
    min_frame_interval_ms: float = DEFAULT_MIN_FRAME_INTERVAL_MS
    # V8.2: Segment list caching for performance
    segment_list_cache_interval: float = 1.0  # Refresh segment list cache every 1 second


def load_default_config() -> ProcessorConfig:
    """Load spool processor configuration from database config table."""
    return ProcessorConfig(
        spool_dir_path=DEFAULT_SPOOL_DIR,
        target_fps=DEFAULT_TARGET_FPS,
    )

class SpoolProcessorNode(Node):
    """
    Spool Processor Node - ACK-Free Video Streaming.
    
    Reads H.264 frames from the spool and publishes them to the decoder
    at a controlled rate without waiting for acknowledgments.
    
    ACK-Free Architecture (Production):
    ----------------------------------
    1. Reads frames continuously from spool
    2. Publishes at target_fps rate
    3. Never blocks on consumer feedback
    4. Relies on retention guards for data safety
    
    This aligns with industry-standard streaming architectures.
    
    V7 Features:
    -----------
    - Gap/duplicate detection with anomaly counters
    - Spool lag monitoring with adaptive pacing
    - SPS/PPS robustness at segment boundaries
    - Watchdog for stalled publishing detection
    - Persisted state for restart continuity
    - Retention guard for segment existence
    - Structured logging for machine parsing
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__('spool_processor')
        
        # Load configuration from database if not provided
        self.config = config or load_default_config()
        
        # Generate unique session ID for this run
        self._session_id = generate_session_id()
        
        # Log mode selection
        logger.info(f"[SpoolProcessor] Mode: ACK-FREE (Production)")
        
        logger.info(f"[SpoolProcessor] Initializing with config: "
                   f"spool_dir={self.config.spool_dir_path}, "
                   f"target_fps={self.config.target_fps}, "
                   f"session_id={self._session_id}")
        
        # Initialize components
        spool_dir = Path(self.config.spool_dir_path)
        spool_dir.mkdir(parents=True, exist_ok=True)

        self._reader = SegmentReader(
            self.config.spool_dir_path,
            cache_refresh_interval=self.config.segment_list_cache_interval
        )
        self._frame_generator: Optional[Generator] = None
        self._current_frame: Optional[FrameRecord] = None
        self._current_frame_index: int = 0
        self._current_segment: int = -1  # Track current segment for SPS/PPS handling
        
        # Sequence counter for published frames
        self._seq_counter: int = 0
        self._seq_lock = threading.Lock()
        
        # SPS/PPS caching for segment boundary handling
        self._cached_sps: Optional[bytes] = None
        self._cached_pps: Optional[bytes] = None
        self._segment_needs_sps_pps: bool = True  # First frame of segment needs SPS/PPS
        
        # State management
        self._state = ProcessorRunState.IDLE
        self._state_lock = threading.Lock()
        
        # Processing thread
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._frames_processed = 0
        self._frames_skipped = 0
        self._segments_processed = 0
        self._sps_pps_prepends = 0
        self._last_stats_time = time.time()
        self._last_detailed_stats_time = time.time()
        self._stats_lock = threading.Lock()
        
        # V7: Robustness counters
        self._last_published_index: int = -1  # For gap/dup detection
        self._last_published_segment: int = -1  # Track which segment last frame came from
        self._anomalies_gap: int = 0  # Gap detections
        self._anomalies_dup: int = 0  # Duplicate detections
        self._last_publish_time: float = 0.0  # For watchdog
        self._sps_pps_missing_critical: int = 0  # SPS/PPS unavailable at boundary
        self._current_target_fps: float = self.config.target_fps  # Adaptive pacing
        self._throttle_log_dict = {}  # For throttled logging
        
        # State file path
        self._state_file_path = os.path.join(self.config.spool_dir_path, self.config.state_file)
        self._allow_next_gap = False
        
        # V8: Initialize retention policy for segment deletion after processing
        if self.config.delete_processed_segments:
            self._retention_policy = RetentionPolicy(
                spool_dir=self.config.spool_dir_path,
                retention_seconds=300.0,  # 5 minutes fallback retention
                cleanup_interval=30.0,  # Cleanup check interval
                min_segments_to_keep=2,  # Always keep at least 2 segments
                retention_safety_enabled=True,  # Protect unprocessed segments
                max_spool_size_bytes=2_147_483_648,  # 2GB hard limit
                delete_processed_segments=True  # Delete segments immediately after processing
            )
            logger.info(format_structured_log(
                "[SpoolProcessor] Segment deletion enabled",
                delete_after_processing=True,
                min_segments_to_keep=2,
                max_spool_size_mb=2048
            ))
            ))
        else:
            self._retention_policy = None
        
        # ROS2 publishers and subscribers
        if IS_RDK:
            # QoS for encoded frames - must match decoder's subscription QoS
            frame_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10  # Buffering for reliability
            )

            # Publisher for encoded frames (to decoder input)
            # This is the ONLY topic - publishes H.264 frames to hobot_codec
            self._frame_pub = self.create_publisher(
                H26XFrame,
                '/spool_image_ch_0',
                frame_qos
            )
            
            logger.info("[SpoolProcessor] ROS2 topics configured (ACK-FREE MODE): "
                       "/spool_image_ch_0 (pub) - Simple, robust architecture")
    
    def start(self):
        """Start the processor."""
        if self._running:
            return
        
        logger.info("[SpoolProcessor] Starting...")
        self._running = True
        
        # V7: Load persisted state for restart continuity
        loaded_state = load_processor_state(self._state_file_path)
        if loaded_state and loaded_state.last_published_index >= 0:
            self._last_published_index = loaded_state.last_published_index
            self._last_published_segment = loaded_state.last_published_segment
            logger.info(format_structured_log(
                "[SpoolProcessor] Loaded persisted state",
                last_index=loaded_state.last_published_index,
                last_segment=loaded_state.last_published_segment,
                prev_session=loaded_state.session_id[:8]
            ))
        
        # Initialize frame generator
        self._init_frame_generator()
        
        # V7.4: Save initial state to let retention policy know we're starting
        # This creates the state file early, reducing the window where retention
        # might delete segments we're about to read
        if self._current_segment >= 0:
            state = ProcessorState(
                last_published_index=self._last_published_index,
                last_published_segment=self._current_segment,
                session_id=self._session_id,
                timestamp=time.time()
            )
            if save_processor_state(self._state_file_path, state):
                logger.info(format_structured_log(
                    "[SpoolProcessor] Initial state saved for retention safety",
                    current_segment=self._current_segment,
                    last_published_index=self._last_published_index
                ))
        
        # Start processing thread
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            daemon=True,
            name="SpoolProcessor"
        )
        self._processor_thread.start()
        
        logger.info("[SpoolProcessor] Started")
    
    def stop(self):
        """Stop the processor gracefully."""
        if not self._running:
            return
        
        logger.info("[SpoolProcessor] Stopping...")
        self._running = False
        
        with self._state_lock:
            self._state = ProcessorRunState.STOPPED
        
        # Wait for processor thread
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
            if self._processor_thread.is_alive():
                logger.error("[SpoolProcessor] ⚠ Processor thread did not stop gracefully within timeout - incomplete stop")
        
        # V7: Save state before exit
        if self._last_published_index >= 0:
            state = ProcessorState(
                last_published_index=self._last_published_index,
                last_published_segment=self._current_segment,
                session_id=self._session_id,
                timestamp=time.time()
            )
            if save_processor_state(self._state_file_path, state):
                logger.info(format_structured_log(
                    "[SpoolProcessor] State saved",
                    last_index=self._last_published_index,
                    last_segment=self._current_segment
                ))
        
        # Log final stats with structured format
        with self._stats_lock:
            # V8: Get segments deleted from retention policy
            segments_deleted = 0
            if self._retention_policy is not None:
                segments_deleted = self._retention_policy.segments_deleted_by_processing
            
            final_msg = format_structured_log(
                "[SpoolProcessor] Final stats",
                session=self._session_id[:8],
                seq=self._seq_counter,
                processed=self._frames_processed,
                skipped=self._frames_skipped,
                segments=self._segments_processed,
                segments_deleted=segments_deleted,
                sps_pps_prepends=self._sps_pps_prepends,
                anomalies_gap=self._anomalies_gap,
                anomalies_dup=self._anomalies_dup
            )
            logger.info(final_msg)
        
        logger.info("[SpoolProcessor] Stopped")
    
    def _init_frame_generator(self):
        """
        Initialize the frame generator from correct resume position.
        
        V8: Fixed to use last_published_segment for resume (not oldest segment).
        This ensures correct restart behavior when segments are deleted and
        frame indices are not monotonic across segments.
        
        Resume logic:
        1. If no persisted state: start from oldest segment
        2. If persisted state exists:
           a. Try to start from last_published_segment
           b. If that segment is missing, find nearest segment >= it
           c. Within resume segment, skip frames with index <= last_published_index
        """
        # Determine starting segment
        start_segment = None
        
        if self._last_published_segment >= 0:
            # V8: Resume from last published segment (or nearest available >= it)
            segments = self._reader.list_segments(use_cache=False)  # Critical operation - no cache
            
            if self._last_published_segment in segments:
                # Exact match - resume from this segment
                start_segment = self._last_published_segment
                logger.info(format_structured_log(
                    "[SpoolProcessor] Resuming from last published segment",
                    last_published_segment=self._last_published_segment,
                    last_published_index=self._last_published_index
                ))
            else:
                # Segment was deleted - find nearest available segment >= last_published
                candidates = [s for s in segments if s >= self._last_published_segment]
                if candidates:
                    start_segment = min(candidates)
                    logger.warning(format_structured_log(
                        "[SpoolProcessor] Resume segment missing, jumping forward",
                        last_published_segment=self._last_published_segment,
                        resume_segment=start_segment,
                        skipped_segments=start_segment - self._last_published_segment
                    ))
                else:
                    # No segments >= last published - fall back to oldest (shouldn't happen normally)
                    start_segment = min(segments) if segments else None
                    if start_segment:
                        logger.warning(format_structured_log(
                            "[SpoolProcessor] No forward segments for resume, using oldest (unusual)",
                            last_published_segment=self._last_published_segment,
                            oldest_available=start_segment
                        ))
        else:
            # No persisted state - start from oldest segment
            start_segment = self._reader.get_oldest_segment()
        
        if start_segment is not None:
            logger.info(f"[SpoolProcessor] Starting from segment {start_segment}")
            self._frame_generator = self._reader.read_frames(start_segment=start_segment)
            self._segment_needs_sps_pps = True
            self._current_segment = start_segment
            
            # V8: If resuming, skip already-published frames WITHIN the resume segment
            # Frame indices are only comparable within the same segment
            if self._last_published_segment >= 0 and start_segment == self._last_published_segment:
                target_index = self._last_published_index + 1
                skipped = 0
                logger.info(format_structured_log(
                    "[SpoolProcessor] Seeking to resume position within segment",
                    resume_segment=start_segment,
                    last_published_index=self._last_published_index,
                    target_index=target_index
                ))
                
                try:
                    while True:
                        frame = next(self._frame_generator)
                        
                        # Extract and cache SPS/PPS while seeking
                        sps, pps = extract_sps_pps(frame.data)
                        if sps:
                            self._cached_sps = sps
                        if pps:
                            self._cached_pps = pps
                        
                        if frame.index <= self._last_published_index:
                            skipped += 1
                            continue
                        else:
                            # Found target or beyond - recreate generator starting here
                            logger.info(format_structured_log(
                                "[SpoolProcessor] Resume position reached",
                                skipped_frames=skipped,
                                next_index=frame.index
                            ))
                            
                            # Use itertools.chain to combine first frame with remaining frames
                            self._frame_generator = itertools.chain([frame], self._frame_generator)
                            break
                            
                except StopIteration:
                    # Reached end of resume segment - generator will naturally continue to next segment
                    logger.info(format_structured_log(
                        "[SpoolProcessor] Reached end of resume segment while seeking",
                        skipped_frames=skipped,
                        target_was=target_index,
                        resume_segment=start_segment
                    ))
                    # Generator is already exhausted, will naturally move to next segment
            elif self._last_published_segment >= 0 and start_segment > self._last_published_segment:
                # Jumped forward to a different segment - don't skip by index
                # (indices are not comparable across segments)
                logger.info(format_structured_log(
                    "[SpoolProcessor] Starting from segment ahead of last published, no index skipping",
                    last_published_segment=self._last_published_segment,
                    start_segment=start_segment
                ))
                # Pre-scan for SPS/PPS if needed
                if self._cached_sps is None or self._cached_pps is None:
                    self._prescan_for_sps_pps()
            else:
                # Starting fresh (no resume) - pre-scan for SPS/PPS
                if self._cached_sps is None or self._cached_pps is None:
                    self._prescan_for_sps_pps()
        else:
            logger.warning("[SpoolProcessor] No segments available")
            self._frame_generator = iter([])
    
    def _prescan_for_sps_pps(self):
        """
        Pre-scan frames to find and cache SPS/PPS NAL units.
        
        V7.1: Simplified to avoid frame skipping. SPS/PPS are now cached during
        normal frame iteration in _get_next_frame(). This eliminates gaps caused
        by complex frame buffering logic that was consuming frames without properly
        re-injecting them.
        """
        logger.info("[SpoolProcessor] SPS/PPS will be cached during normal frame iteration")
        # Note: SPS/PPS caching happens in _get_next_frame() during normal processing
        # This avoids the frame skipping issue that occurred with buffered_generator
    
    def _get_next_frame(self) -> Optional[FrameRecord]:
        """
        Get the next frame from the spool.
        
        V7: Includes retention guard - checks if current segment still exists.
        V7.5: When current segment is missing, jumps forward to nearest available
              segment >= current_segment (not oldest) to prevent rewinds.
        """
        if self._frame_generator is None:
            self._init_frame_generator()
        
        # V7: Retention guard - check if current segment was deleted
        if self._current_segment >= 0:
            segments = self._reader.list_segments()
            if self._current_segment not in segments:
                error_msg = format_structured_log(
                    "🔴 CRITICAL: Current segment disappeared",
                    segment=self._current_segment,
                    available_segments=len(segments)
                )
                throttled_log(
                    logger.error,
                    f"[SpoolProcessor] {error_msg}",
                    key="segment_disappeared",
                    throttle_dict=self._throttle_log_dict,
                    min_interval=10.0
                )
                # V7.5: Jump to nearest available segment >= current, not oldest
                # This prevents rewinds when retention deletes the current segment
                if segments:
                    candidates = [s for s in segments if s >= self._current_segment]
                    if candidates:
                        target_segment = min(candidates)
                        logger.warning(format_structured_log(
                            "[SpoolProcessor] ⚠ Jumping forward after segment deletion",
                            missing_segment=self._current_segment,
                            target_segment=target_segment
                        ))
                    else:
                        # No segments >= current available, use the minimum available
                        # (This is the fallback, but at least we log it)
                        target_segment = min(segments)
                        logger.warning(format_structured_log(
                            "[SpoolProcessor] ⚠ No forward segments available, using oldest",
                            missing_segment=self._current_segment,
                            target_segment=target_segment
                        ))
                    
                    self._frame_generator = self._reader.read_frames(start_segment=target_segment)
                    self._current_segment = target_segment
                    self._segment_needs_sps_pps = True
                else:
                    # No segments at all
                    self._frame_generator = iter([])
                    return None
        
        try:
            frame = next(self._frame_generator)
            # Extract and cache SPS/PPS if present in this frame
            sps, pps = extract_sps_pps(frame.data)
            if sps:
                self._cached_sps = sps
                logger.debug(f"[SpoolProcessor] Cached SPS from frame {frame.index}")
            if pps:
                self._cached_pps = pps
                logger.debug(f"[SpoolProcessor] Cached PPS from frame {frame.index}")
            if sps or pps:
                self._allow_next_gap = True
            return frame
        except StopIteration:
            # Segment exhausted, try to move to next sequential segment
            old_segment = self._current_segment
            with self._stats_lock:
                if self._current_segment >= 0:
                    self._segments_processed += 1
            
            # V8: Notify retention policy that segment is fully processed
            # This will trigger immediate segment deletion if enabled
            if self._retention_policy is not None and old_segment >= 0:
                self._retention_policy.set_last_processed_segment(old_segment)
                logger.info(format_structured_log(
                    "[SpoolProcessor] 🗑️ Segment processed and queued for deletion",
                    segment=old_segment
                ))
            
            self._segment_needs_sps_pps = True  # New segment will need SPS/PPS
            
            # V7.2: Move to NEXT sequential segment, not oldest
            # This ensures proper sequential processing even when old segments are deleted
            next_segment = old_segment + 1 if old_segment >= 0 else None
            
            # Check if next segment exists, or find the nearest available segment after current
            available_segments = self._reader.list_segments()
            target_segment = None
            
            if next_segment is not None and next_segment in available_segments:
                # Next sequential segment exists
                target_segment = next_segment
            elif available_segments:
                # Next segment missing - find nearest segment >= next_segment
                candidates = [s for s in available_segments if s >= next_segment] if next_segment is not None else available_segments
                if candidates:
                    target_segment = min(candidates)
                    if old_segment >= 0:
                        skipped = target_segment - next_segment
                        if skipped > 0:
                            logger.warning(format_structured_log(
                                "[SpoolProcessor] ⚠ Segments missing, skipping forward",
                                old_segment=old_segment,
                                next_expected=next_segment,
                                actual=target_segment,
                                skipped=skipped
                            ))
            
            if target_segment is not None:
                logger.info(f"[SpoolProcessor] Segment transition: {old_segment} → {target_segment}")
                self._frame_generator = self._reader.read_frames(start_segment=target_segment)
                self._current_segment = target_segment
                
                # V7.4: Save state when transitioning segments for cross-process safety
                # This allows retention policy (in recorder process) to know current position
                if self._last_published_index >= 0:
                    state = ProcessorState(
                        last_published_index=self._last_published_index,
                        last_published_segment=self._current_segment,
                        session_id=self._session_id,
                        timestamp=time.time()
                    )
                    save_processor_state(self._state_file_path, state)
                
                try:
                    frame = next(self._frame_generator)
                    # Extract and cache SPS/PPS if present
                    sps, pps = extract_sps_pps(frame.data)
                    if sps:
                        self._cached_sps = sps
                    if pps:
                        self._cached_pps = pps
                    return frame
                except StopIteration:
                    # V7.5: Empty segment - skip forward to next available segment > target
                    # This prevents infinite loops reopening the same empty segment
                    throttled_log(
                        logger.warning,
                        format_structured_log(
                            "[SpoolProcessor] ⚠ Empty segment detected, skipping forward",
                            empty_segment=target_segment
                        ),
                        key="empty_segment",
                        throttle_dict=self._throttle_log_dict,
                        min_interval=5.0
                    )
                    
                    # Mark this segment as processed
                    with self._stats_lock:
                        self._segments_processed += 1
                    
                    # Find next segment > target_segment
                    forward_candidates = [s for s in available_segments if s > target_segment]
                    if forward_candidates:
                        next_target = min(forward_candidates)
                        logger.info(format_structured_log(
                            "[SpoolProcessor] Advancing past empty segment",
                            empty_segment=target_segment,
                            next_segment=next_target
                        ))
                        self._frame_generator = self._reader.read_frames(start_segment=next_target)
                        self._current_segment = next_target
                        self._segment_needs_sps_pps = True
                        
                        # Try to get frame from next segment
                        try:
                            frame = next(self._frame_generator)
                            sps, pps = extract_sps_pps(frame.data)
                            if sps:
                                self._cached_sps = sps
                            if pps:
                                self._cached_pps = pps
                            return frame
                        except StopIteration:
                            # Multiple empty segments - return None and let the main loop retry
                            return None
                    else:
                        # No more segments available after the empty one
                        return None
            else:
                logger.debug("[SpoolProcessor] No more segments available")
                self._frame_generator = iter([])
                return None
    
    def _maybe_prepend_sps_pps(self, data: bytes) -> bytes:
        """
        Prepend cached SPS/PPS to frame data if needed.
        
        V7: Always prepends at segment boundaries if cache exists, regardless of IDR detection.
        If cache is missing, attempts prescan and logs critical warning.
        
        This ensures the decoder can initialize properly at segment boundaries,
        even if the first frame of a segment isn't an IDR frame with SPS/PPS.
        
        Args:
            data: Original frame data
            
        Returns:
            Frame data with SPS/PPS prepended if needed, otherwise original data
        """
        if not self.config.prepend_sps_pps:
            return data
        
        if not self._segment_needs_sps_pps:
            return data
        
        # V7: At segment boundary, ALWAYS prepend cached SPS/PPS if available
        if self._cached_sps and self._cached_pps:
            prepended = bytearray()
            prepended.extend(self._cached_sps)
            prepended.extend(self._cached_pps)
            with self._stats_lock:
                self._sps_pps_prepends += 1
            self._segment_needs_sps_pps = False
            logger.debug(format_structured_log(
                "[SpoolProcessor] Prepended SPS/PPS at segment boundary",
                segment=self._current_segment,
                prepended_bytes=len(prepended)
            ))
            return bytes(prepended) + data
        
        # V7: Cache missing at boundary - check if frame has SPS/PPS
        frame_has_idr, frame_has_sps, frame_has_pps = False, False, False
        try:
            frame_has_idr, frame_has_sps, frame_has_pps = detect_frame_type(data)
        except Exception:
            pass
        
        # If frame has SPS/PPS, extract and cache them
        if frame_has_sps and frame_has_pps:
            sps, pps = extract_sps_pps(data)
            if sps:
                self._cached_sps = sps
            if pps:
                self._cached_pps = pps
            self._segment_needs_sps_pps = False
            logger.debug("[SpoolProcessor] Extracted and cached SPS/PPS from frame at segment boundary")
            return data
        
        # V7: Critical - no cached SPS/PPS and frame doesn't have them
        with self._stats_lock:
            self._sps_pps_missing_critical += 1
        
        critical_msg = format_structured_log(
            "🔴 CRITICAL: SPS/PPS unavailable at segment boundary",
            segment=self._current_segment,
            has_idr=frame_has_idr,
            has_sps=frame_has_sps,
            has_pps=frame_has_pps,
            missing_count=self._sps_pps_missing_critical
        )
        throttled_log(
            logger.error,
            f"[SpoolProcessor] {critical_msg}",
            key="sps_pps_missing",
            throttle_dict=self._throttle_log_dict,
            min_interval=5.0
        )
        
        self._segment_needs_sps_pps = False
        return data
    
    def _publish_frame(self, record: FrameRecord) -> bool:
        """
        Publish a frame to the decoder input topic.
        
        V7: Includes gap/dup detection and optional CRC32 checksum logging.
        
        Returns:
            bool: True if publishing succeeded, False otherwise
        """
        if not IS_RDK:
            return True

        try:
            # V7: Detect gaps and duplicates WITHIN the same segment only
            # Frame indices are not continuous across segment boundaries
            if self._last_published_index >= 0 and hasattr(self, '_last_published_segment'):
                # Only check for gaps/dups if we're in the same segment
                if self._last_published_segment == self._current_segment:
                    expected_index = self._last_published_index + 1
                    if record.index > expected_index:
                        if self._allow_next_gap:
                            self._allow_next_gap = False # Expected no need to log warning when PPS or SPS
                        else:
                            gap_size = record.index - expected_index
                            with self._stats_lock:
                                self._anomalies_gap += 1
                            gap_msg = format_structured_log(
                                "⚠ GAP DETECTED",
                                expected=expected_index,
                                actual=record.index,
                                gap_size=gap_size,
                                segment=self._current_segment,
                                total_gaps=self._anomalies_gap
                            )
                            logger.warning(f"[SpoolProcessor] {gap_msg}")

                    elif record.index < expected_index:
                        # Duplicate or out-of-order within segment
                        with self._stats_lock:
                            self._anomalies_dup += 1
                        dup_msg = format_structured_log(
                            "⚠ DUPLICATE/OUT-OF-ORDER DETECTED",
                            expected=expected_index,
                            actual=record.index,
                            segment=self._current_segment,
                            total_dups=self._anomalies_dup
                        )
                        logger.warning(f"[SpoolProcessor] {dup_msg}")
            
            # Get next sequence number
            with self._seq_lock:
                seq = self._seq_counter
                self._seq_counter += 1
            
            # Prepare frame data with SPS/PPS prepending if needed
            frame_data = self._maybe_prepend_sps_pps(record.data)
            
            # Publish the encoded frame immediately (no delay needed)
            # The FIFO queue in Ros2FrameServer will handle proper correlation
            frame_msg = H26XFrame()
            frame_msg.index = record.index
            frame_msg.width = record.width
            frame_msg.height = record.height
            frame_msg.dts = Time()
            frame_msg.dts.sec = record.dts_sec
            frame_msg.dts.nanosec = record.dts_nsec
            frame_msg.pts = Time()
            frame_msg.pts.sec = record.pts_sec
            frame_msg.pts.nanosec = record.pts_nsec
            
            # Convert encoding string to list of 12 unsigned integers (as expected by H26XFrame)
            # The encoding field in H26XFrame is a sequence of 12 bytes (uint8 array)
            if isinstance(record.encoding, str):
                encoding_bytes = record.encoding.encode('utf-8')[:12]
            elif isinstance(record.encoding, bytes):
                encoding_bytes = record.encoding[:12]
            else:
                encoding_bytes = bytes(record.encoding)[:12]
            # Pad to exactly 12 bytes
            encoding_padded = list(encoding_bytes) + [0] * (12 - len(encoding_bytes))
            frame_msg.encoding = encoding_padded
            
            frame_msg.data = list(frame_data)
            
            self._frame_pub.publish(frame_msg)
            
            # V7: Update last published index and segment (for gap/dup detection and state persistence)
            self._last_published_index = record.index
            self._last_published_segment = self._current_segment
            self._last_publish_time = time.time()
            
            # V7: Optional CRC32 logging
            crc32 = None
            if self.config.enable_crc32_logging:
                crc32 = calculate_crc32(frame_data)
            
            # Structured logging - use debug for regular frames, info for milestones
            if seq % 100 == 0:
                log_fields = {
                    "frames_published": seq,
                    "frame_index": record.index,
                    "session": self._session_id[:8],
                    "segment": self._current_segment,
                    "data_len": len(frame_data)
                }
                if crc32 is not None:
                    log_fields["crc32"] = crc32
                milestone_msg = format_structured_log(
                    "[SpoolProcessor] 📤 Milestone",
                    **log_fields
                )
                logger.info(milestone_msg)
            else:
                log_fields = {
                    "frame_index": record.index,
                    "seq": seq,
                    "session": self._session_id[:8],
                    "segment": self._current_segment,
                    "data_len": len(frame_data)
                }
                if crc32 is not None:
                    log_fields["crc32"] = crc32
                pub_msg = format_structured_log(
                    "[SpoolProcessor] 📤 Frame published",
                    **log_fields
                )
                logger.debug(pub_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error publishing frame: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    


    def _processor_loop(self):
        """
        Main processing loop - ACK-free continuous publishing.
        """
        logger.info("[SpoolProcessor] Processing loop started")
        
        # Call the ACK-free loop directly
        self._processor_loop_ack_free()
        
        logger.info("[SpoolProcessor] Processing loop stopped")
    
    def _processor_loop_ack_free(self):
        """
        V6/V7 ACK-Free Processing Loop (Production-Grade).
        
        This loop processes frames continuously without waiting for ACKs:
        1. Read frame from spool
        2. Publish immediately
        3. Pace to target_fps (adaptive)
        4. Never block
        
        V7 Additions:
        - Spool lag computation and warnings
        - Watchdog for stalled publishing (using monotonic time)
        - Adaptive pacing on high lag (optional)
        
        Benefits:
        - No deadlocks (impossible)
        - Maximum throughput
        - Stable latency
        - Production-safe
        
        The consumer processes frames at its own pace. Retention guards
        protect unprocessed data from deletion.
        """
        logger.info(format_structured_log(
            "[SpoolProcessor] ACK-FREE mode active",
            target_fps=self.config.target_fps,
            adaptive_pacing=self.config.enable_adaptive_pacing,
            min_frame_interval_ms=self.config.min_frame_interval_ms,
            delete_processed_segments=self.config.delete_processed_segments
        ))
        
        # Use monotonic time for pacing and watchdog
        frame_interval = 1.0 / self._current_target_fps if self._current_target_fps > 0 else 0.025
        last_publish_monotonic = time.monotonic()
        last_watchdog_check = time.monotonic()
        
        # V8: Pre-calculate minimum interval (avoid division in tight loop)
        min_interval_sec = self.config.min_frame_interval_ms / 1000.0
        
        with self._state_lock:
            self._state = ProcessorRunState.PUBLISHING
        
        while self._running:
            try:
                current_monotonic = time.monotonic()
                
                # V7: Compute spool lag
                segments = self._reader.list_segments()
                newest_segment = max(segments) if segments else None
                spool_lag = 0
                if newest_segment is not None and self._current_segment >= 0:
                    spool_lag = newest_segment - self._current_segment

                # V7: Check lag thresholds and adaptive pacing (3-tier system)
                if self.config.enable_adaptive_pacing:
                    if spool_lag < DEFAULT_SPOOL_LAG_HEALTHY_THRESHOLD:
                        # HEALTHY: < 5 segments - RELAX and save resources
                        target_fps = DEFAULT_ADAPTIVE_FPS_RELAXED  # 20 FPS
                        mode_emoji = "😌"
                        mode_text = "RELAXED - System healthy, conserving resources"

                    elif spool_lag <= DEFAULT_SPOOL_LAG_NORMAL_THRESHOLD:
                        # NORMAL: 5-15 segments - maintain default pace
                        target_fps = DEFAULT_TARGET_FPS  # 30 FPS
                        mode_emoji = "✅"
                        mode_text = "NORMAL - Maintaining default pace"

                    else:
                        # HIGH LAG: > 15 segments - SPEED UP to catch up
                        target_fps = DEFAULT_ADAPTIVE_FPS_MAX  # 66 FPS (15ms intervals)
                        mode_emoji = "🚀"
                        mode_text = "CATCHING UP - High lag detected"

                    # Only update if significant change
                    if abs(self._current_target_fps - target_fps) > 0.1:
                        old_fps = self._current_target_fps
                        self._current_target_fps = target_fps
                        frame_interval = 1.0 / self._current_target_fps if self._current_target_fps > 0 else 0.025

                        # Choose appropriate log level based on mode
                        log_func = logger.info if spool_lag < DEFAULT_SPOOL_LAG_NORMAL_THRESHOLD else logger.warning

                        log_func(format_structured_log(
                            f"[SpoolProcessor] {mode_emoji} Adaptive pacing: {mode_text}",
                            spool_lag=spool_lag,
                            old_fps=f"{old_fps:.1f}",
                            new_fps=f"{self._current_target_fps:.1f}",
                            new_interval_ms=f"{frame_interval * 1000:.1f}"
                        ))
                
                # V7: Watchdog - check for stalled publishing (using monotonic time)
                if current_monotonic - last_watchdog_check > 10.0:  # Check every 10 seconds
                    if self._last_publish_time > 0:
                        # Convert last publish time to monotonic-relative
                        stalled_time = time.time() - self._last_publish_time
                        if stalled_time > self.config.watchdog_timeout:
                            watchdog_msg = format_structured_log(
                                "🔴 WATCHDOG: No frames published recently",
                                stalled_seconds=f"{stalled_time:.1f}",
                                threshold=self.config.watchdog_timeout
                            )
                            throttled_log(
                                logger.error,
                                f"[SpoolProcessor] {watchdog_msg}",
                                key="watchdog",
                                throttle_dict=self._throttle_log_dict,
                                min_interval=10.0
                            )
                    last_watchdog_check = current_monotonic
                
                # Get next frame
                frame = self._get_next_frame()
                
                if frame is None:
                    # Spool is empty, wait and retry
                    with self._state_lock:
                        self._state = ProcessorRunState.SPOOL_EMPTY
                    logger.debug("[SpoolProcessor] Spool empty, waiting for new frames...")
                    time.sleep(self.config.poll_interval)
                    with self._state_lock:
                        self._state = ProcessorRunState.PUBLISHING
                    continue
                
                self._current_frame = frame
                self._current_frame_index = frame.index
                
                # V7.1: Adaptive frame rate pacing with robust interval guards
                publish_start = time.monotonic()
                
                # Publish frame (no ACK wait)
                success = self._publish_frame(frame)
                
                # Calculate adaptive sleep based on actual processing time
                publish_end = time.monotonic()
                processing_time = publish_end - last_publish_monotonic
                
                # Guard against negative or zero intervals
                if frame_interval <= 0:
                    logger.error(
                        f"[SpoolProcessor] Invalid frame_interval: {frame_interval}, "
                        f"resetting to 25ms (40fps)"
                    )
                    frame_interval = 0.025
                    self._current_target_fps = 40.0
                
                # V8: Ensure minimum frame interval to avoid CPU heat
                # This guarantees at least min_frame_interval_ms between frames
                # (min_interval_sec is pre-calculated outside the loop)
                target_sleep = max(min_interval_sec, frame_interval - processing_time)
                
                if target_sleep > 0:
                    time.sleep(target_sleep)
                
                last_publish_monotonic = time.monotonic()
                
                if success:
                    with self._stats_lock:
                        self._frames_processed += 1
                else:
                    with self._stats_lock:
                        self._frames_skipped += 1
                    logger.warning(f"[SpoolProcessor] Frame {frame.index} publish failed")
                
                # Log stats periodically
                self._maybe_log_stats()
                
            except Exception as e:
                logger.error(f"[SpoolProcessor] Error in ACK-free loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(0.1)
    


    def _maybe_log_stats(self):
        """Log statistics periodically with structured format (V7)."""
        current_time = time.time()
        
        # Regular stats every 10 seconds
        if current_time - self._last_stats_time >= self.config.stats_interval:
            with self._stats_lock:
                # Compute spool lag (uses cached segment list)
                segments = self._reader.list_segments()
                newest_segment = max(segments) if segments else None
                spool_lag = 0
                if newest_segment is not None and self._current_segment >= 0:
                    spool_lag = newest_segment - self._current_segment
                
                # V8: Get segments deleted from retention policy
                segments_deleted = 0
                if self._retention_policy is not None:
                    segments_deleted = self._retention_policy.segments_deleted_by_processing
                
                # V8.2: Get cache statistics for performance monitoring
                cache_stats = self._reader.get_cache_stats()
                
                # Structured stats logging
                stats_msg = format_structured_log(
                    "[SpoolProcessor] Stats",
                    session=self._session_id[:8],
                    seq=self._seq_counter,
                    frames_processed=self._frames_processed,
                    frames_skipped=self._frames_skipped,
                    anomalies_gap=self._anomalies_gap,
                    anomalies_dup=self._anomalies_dup,
                    segments_processed=self._segments_processed,
                    segments_deleted=segments_deleted,
                    sps_pps_prepends=self._sps_pps_prepends,
                    sps_pps_missing=self._sps_pps_missing_critical,
                    state=self._state.value,
                    cache_hit_rate=f"{cache_stats['hit_rate_pct']:.1f}%"
                )
                logger.info(stats_msg)
                
                # Log spool status with lag
                spool_msg = format_structured_log(
                    "[SpoolProcessor] Spool",
                    total_segments=len(segments),
                    current_segment=self._current_segment,
                    current_frame=self._current_frame_index,
                    spool_lag=spool_lag,
                )
                logger.info(spool_msg)
                
                # V8.2: Log cache performance (only when cache misses occur to avoid spam)
                if cache_stats['misses'] > 0:
                    cache_msg = format_structured_log(
                        "[SpoolProcessor] Cache stats",
                        hits=cache_stats['hits'],
                        misses=cache_stats['misses'],
                        hit_rate=f"{cache_stats['hit_rate_pct']:.1f}%",
                        refresh_interval=f"{cache_stats['refresh_interval_sec']:.1f}s"
                    )
                    logger.debug(cache_msg)
                
                # Warn on spool lag thresholds
                if spool_lag >= self.config.spool_lag_error_threshold:
                    lag_msg = format_structured_log(
                        "🔴 ERROR: High spool lag",
                        spool_lag=spool_lag,
                        threshold=self.config.spool_lag_error_threshold
                    )
                    logger.error(f"[SpoolProcessor] {lag_msg}")
                elif spool_lag >= self.config.spool_lag_warn_threshold:
                    lag_msg = format_structured_log(
                        "⚠ WARNING: Elevated spool lag",
                        spool_lag=spool_lag,
                        threshold=self.config.spool_lag_warn_threshold
                    )
                    logger.warning(f"[SpoolProcessor] {lag_msg}")
            
            self._last_stats_time = current_time
        
        # Detailed stats every 2 minutes (120 seconds)
        if current_time - self._last_detailed_stats_time >= 120.0:
            with self._stats_lock:
                # Get spool information for lag detection
                segments = self._reader.list_segments()
                oldest_segment = self._reader.get_oldest_segment()
                newest_segment = max(segments) if segments else None
                
                # Calculate spool lag: difference between newest and current segment
                spool_lag = 0
                if newest_segment is not None and self._current_segment >= 0:
                    spool_lag = newest_segment - self._current_segment
                
                # V7.2: Log recorder vs processor lag with RECORDER_LAG keyword (every 2 minutes)
                if newest_segment is not None and self._current_segment >= 0:
                    # Estimate time lag (assuming ~5s per segment average)
                    time_lag_estimate = spool_lag * 5
                    if spool_lag > 0:
                        logger.info(format_structured_log(
                            "[SpoolProcessor] RECORDER_LAG: Recorder ahead of processor",
                            recorder_segment=newest_segment,
                            processor_segment=self._current_segment,
                            lag_segments=spool_lag,
                            lag_time_estimate=f"~{time_lag_estimate}s"
                        ))
                    elif spool_lag < 0:
                        logger.info(format_structured_log(
                            "[SpoolProcessor] RECORDER_LAG: Processor ahead of recorder (catching up)",
                            recorder_segment=newest_segment,
                            processor_segment=self._current_segment,
                            lag_segments=spool_lag
                        ))
                    else:
                        logger.info(format_structured_log(
                            "[SpoolProcessor] RECORDER_LAG: Processor synchronized with recorder",
                            recorder_segment=newest_segment,
                            processor_segment=self._current_segment,
                            lag_segments=0
                        ))
                
                logger.info("=" * 80)
                logger.info(f"[SpoolProcessor] 📊 Detailed Statistics (2-minute summary)")
                logger.info(f"  Session: {self._session_id}")
                logger.info(f"  ACK Statistics:")
                logger.info(f"  Frame Processing:")
                logger.info(f"    - Processed: {self._frames_processed}")
                logger.info(f"    - Skipped: {self._frames_skipped}")
                logger.info(f"  Spool Status:")
                logger.info(f"    - Total segments: {len(segments)}")
                logger.info(f"    - Current segment: {self._current_segment}")
                logger.info(f"    - Oldest segment: {oldest_segment}")
                logger.info(f"    - Newest segment: {newest_segment}")
                logger.info(f"    - Spool lag: {spool_lag} segments")
                
                # Warn if spool is falling behind (lag > 10 segments = ~50 seconds at 5s/segment)
                if spool_lag > 10:
                    logger.warning(f"  ⚠ SPOOL LAG WARNING: Processor is {spool_lag} segments behind!")
                    logger.warning(f"     Recording is ahead by ~{spool_lag * 5}s. Processing too slow!")
                    logger.warning(f"     Consider: reducing ACK timeout, increasing processing speed, or checking consumer.")
                elif spool_lag > 5:
                    logger.warning(f"  ⚠ SPOOL LAG NOTICE: Processor is {spool_lag} segments behind (borderline)")
                else:
                    logger.info(f"  ✓ Spool lag is healthy ({spool_lag} segments)")
                
                logger.info("=" * 80)
            
            self._last_detailed_stats_time = current_time
    
    def get_state(self) -> ProcessorRunState:
        """Get current processor state."""
        with self._state_lock:
            return self._state


def main():
    """Main entry point for the spool processor node."""
    logger.info("=" * 60)
    logger.info("  Spool Processor Node - Accuracy Mode")
    logger.info("=" * 60)
    
    if not IS_RDK:
        logger.error("[SpoolProcessor] This node requires RDK platform with ROS2")
        logger.info("[SpoolProcessor] Running in stub mode for testing")
    
    # Initialize ROS2
    if IS_RDK:
        rclpy.init()
    
    # Create and start node
    node = SpoolProcessorNode()
    node.start()
    
    # Setup signal handlers for clean shutdown
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"[SpoolProcessor] Received signal {signum}, shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Spin ROS2
    if IS_RDK:
        try:
            while not shutdown_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            node.stop()
            node.destroy_node()
            rclpy.shutdown()
    else:
        # For testing without ROS2
        logger.info("[SpoolProcessor] Running in test mode. Press Ctrl+C to exit.")
        try:
            shutdown_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            node.stop()
    
    logger.info("[SpoolProcessor] Shutdown complete")


if __name__ == '__main__':
    main()
