#!/usr/bin/env python3
"""
Spool Processor Node for Accuracy Mode.

This node reads H.264 frames from the spool and publishes them to the decoder.

V6 Update: ACK-Free Mode (Production-Grade)
-------------------------------------------
Per-frame ACKs are fundamentally incompatible with real-time video processing.
They cause: ACK reordering, blocking, DDS QoS issues, and false confidence.

When `spool_ack_free_mode=true`, the processor:
1. Reads frames continuously without waiting for ACK
2. Publishes at a controlled rate (target_fps)
3. Never blocks on consumer feedback
4. Relies on retention guards to protect unprocessed data

This aligns with industry-standard streaming architectures (Kafka, GStreamer, DeepStream).

Legacy ACK Mode (Deprecated):
----------------------------
When `spool_ack_free_mode=false` (default for backward compatibility):
1. Waits for ACK from BagCounterApp before sending next frame
2. Implements strict backpressure
3. NOT recommended for production

Usage:
    python -m src.ros2_spool.spool_processor_node

Configuration (via database config table):
    spool_dir: Directory for spool files (default: /home/sunrise/BreadCounting/data/spool)
    spool_ack_free_mode: Enable ACK-free mode (default: true, recommended)
    spool_target_fps: Target FPS for ACK-free mode (default: 25.0)
    spool_ack_timeout: Timeout waiting for ACK in seconds (default: 10.0) - only for legacy mode
    spool_retry_count: Number of retries before advancing (default: 2) - only for legacy mode
"""

import os
import sys
import time
import signal
import threading
from typing import Optional, Generator
from dataclasses import dataclass
from enum import Enum

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
from src.logging.Database import DatabaseManager
from src import constants

# Import message definitions
from src.ros2_spool.messages import (
    generate_session_id,
    get_current_time_ros,
    FrameMetadata,
    ProcessingAck,
    ProcessingReady,
    processing_ack_from_ros_string,
    processing_ready_from_ros_string,
    frame_metadata_to_ros_string
)

# ROS2 imports (only on RDK platform)
if IS_RDK:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
    from img_msgs.msg import H26XFrame
    from std_msgs.msg import UInt32, String
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


class ProcessorState(Enum):
    """State of the processor."""
    IDLE = "idle"
    PUBLISHING = "publishing"  # V6: ACK-free mode - continuously publishing
    WAITING_FOR_ACK = "waiting_for_ack"  # Legacy mode only
    WAITING_FOR_READY = "waiting_for_ready"  # Waiting for consumer ready signal
    SPOOL_EMPTY = "spool_empty"
    STOPPED = "stopped"


# Default configuration values
DEFAULT_SPOOL_DIR = "/home/sunrise/BreadCounting/data/spool"
DEFAULT_ACK_TIMEOUT = 10.0  # Legacy mode only
DEFAULT_RETRY_COUNT = 2  # Legacy mode only
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_STATS_INTERVAL = 10.0
DEFAULT_STARTUP_GRACE_PERIOD = 10.0  # Seconds to wait for consumer to start
DEFAULT_SPS_PPS_PREPEND = True  # Prepend cached SPS/PPS to first frame of segment
DEFAULT_ACK_FREE_MODE = True  # V6: ACK-free mode enabled by default (production-grade)
DEFAULT_TARGET_FPS = 40.0  # V6: Target FPS for ACK-free mode
DEFAULT_STATE_FILE = "processor_state.json"  # Relative to spool_dir
DEFAULT_SPOOL_LAG_WARN_THRESHOLD = 5  # Segments
DEFAULT_SPOOL_LAG_ERROR_THRESHOLD = 10  # Segments
DEFAULT_WATCHDOG_TIMEOUT = 30.0  # Seconds without publishing before alert
DEFAULT_ENABLE_ADAPTIVE_PACING = False  # Reduce FPS on high lag
DEFAULT_ADAPTIVE_FPS_MIN = 15.0  # Minimum FPS during adaptive pacing
DEFAULT_ENABLE_CRC32_LOGGING = False  # Add CRC32 checksums to logs


@dataclass
class ProcessorConfig:
    """Configuration for the spool processor."""
    spool_dir: str = DEFAULT_SPOOL_DIR
    ack_timeout: float = DEFAULT_ACK_TIMEOUT  # Legacy mode only
    retry_count: int = DEFAULT_RETRY_COUNT  # Legacy mode only
    poll_interval: float = DEFAULT_POLL_INTERVAL
    stats_interval: float = DEFAULT_STATS_INTERVAL
    startup_grace_period: float = DEFAULT_STARTUP_GRACE_PERIOD
    prepend_sps_pps: bool = DEFAULT_SPS_PPS_PREPEND
    # V6: ACK-free mode configuration
    ack_free_mode: bool = DEFAULT_ACK_FREE_MODE
    target_fps: float = DEFAULT_TARGET_FPS
    # V7: Robustness and observability
    state_file: str = DEFAULT_STATE_FILE
    spool_lag_warn_threshold: int = DEFAULT_SPOOL_LAG_WARN_THRESHOLD
    spool_lag_error_threshold: int = DEFAULT_SPOOL_LAG_ERROR_THRESHOLD
    watchdog_timeout: float = DEFAULT_WATCHDOG_TIMEOUT
    enable_adaptive_pacing: bool = DEFAULT_ENABLE_ADAPTIVE_PACING
    adaptive_fps_min: float = DEFAULT_ADAPTIVE_FPS_MIN
    enable_crc32_logging: bool = DEFAULT_ENABLE_CRC32_LOGGING


def load_default_config() -> ProcessorConfig:
    """Load spool processor configuration from database config table."""
    return ProcessorConfig(
        spool_dir=DEFAULT_SPOOL_DIR,
        ack_timeout=DEFAULT_ACK_TIMEOUT,
        retry_count=DEFAULT_RETRY_COUNT,
        ack_free_mode=DEFAULT_ACK_FREE_MODE,
        target_fps=DEFAULT_TARGET_FPS,
    )

class SpoolProcessorNode(Node):
    """
    ROS2 Node that processes spooled H.264 frames.
    
    V6 ACK-Free Mode (Production-Grade, Recommended):
    ------------------------------------------------
    When `ack_free_mode=True` (default):
    1. Reads frames continuously from spool
    2. Publishes at target_fps rate
    3. Never blocks on consumer feedback
    4. Relies on retention guards for data safety
    
    This mode aligns with industry-standard streaming architectures.
    
    Legacy ACK Mode (Deprecated):
    ----------------------------
    When `ack_free_mode=False`:
    1. Waits for ACK from BagCounterApp before sending next frame
    2. Implements strict backpressure
    3. NOT recommended - causes blocking and deadlocks
    
    Production Reliability Features:
    - V6: ACK-free continuous processing (no blocking)
    - Startup synchronization: Waits for consumer before processing
    - SPS/PPS caching: Prepends to segment boundaries for decoder init
    - Graceful degradation: Continues processing even under load
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__('spool_processor')
        
        # Load configuration from database if not provided
        self.config = config or load_default_config()
        
        # Generate unique session ID for this run
        self._session_id = generate_session_id()
        
        # Log mode selection
        mode_str = "ACK-FREE (V6 Production)" if self.config.ack_free_mode else "LEGACY ACK (Deprecated)"
        logger.info(f"[SpoolProcessor] Mode: {mode_str}")
        
        logger.info(f"[SpoolProcessor] Initializing with config: "
                   f"spool_dir={self.config.spool_dir}, "
                   f"ack_free_mode={self.config.ack_free_mode}, "
                   f"target_fps={self.config.target_fps}, "
                   f"ack_timeout={self.config.ack_timeout}s (legacy only), "
                   f"retry_count={self.config.retry_count} (legacy only), "
                   f"startup_grace={self.config.startup_grace_period}s, "
                   f"session_id={self._session_id}")
        
        # Initialize components
        self._reader = SegmentReader(self.config.spool_dir)
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
        self._state = ProcessorState.WAITING_FOR_READY
        self._state_lock = threading.Lock()
        self._ack_received = threading.Event()
        self._ready_received = threading.Event()
        self._last_ack: Optional[ProcessingAck] = None
        self._consumer_session_id: Optional[str] = None  # Session ID from consumer's READY
        self._last_ack_time: float = 0.0  # Track last successful ACK for watchdog
        
        # Processing thread
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._frames_processed = 0
        self._frames_retried = 0
        self._frames_skipped = 0
        self._ack_timeouts = 0
        self._ack_rejected_stale = 0  # ACKs rejected due to wrong session
        self._ack_accepted = 0  # Total ACKs accepted
        self._segments_processed = 0
        self._sps_pps_prepends = 0
        self._last_stats_time = time.time()
        self._last_detailed_stats_time = time.time()  # For 2-minute detailed stats
        self._stats_lock = threading.Lock()
        
        # V7: Robustness counters
        self._last_published_index: int = -1  # For gap/dup detection
        self._anomalies_gap: int = 0  # Gap detections
        self._anomalies_dup: int = 0  # Duplicate detections
        self._last_publish_time: float = 0.0  # For watchdog
        self._sps_pps_missing_critical: int = 0  # SPS/PPS unavailable at boundary
        self._current_target_fps: float = self.config.target_fps  # Adaptive pacing
        self._throttle_log_dict = {}  # For throttled logging
        
        # State file path
        self._state_file_path = os.path.join(self.config.spool_dir, self.config.state_file)
        
        # ROS2 publishers and subscribers
        if IS_RDK:
            # QoS for READY topic - TRANSIENT_LOCAL for late joiners
            ready_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
            )
            
            # QoS for ACK and metadata - RELIABLE with good depth
            control_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=20  # Higher depth for control messages
            )
            
            # QoS for encoded frames - must match decoder's subscription QoS
            frame_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10  # Buffering for reliability
            )

            # Publisher for encoded frames (to decoder input)
            self._frame_pub = self.create_publisher(
                H26XFrame,
                '/spool_image_ch_0',
                frame_qos
            )
            
            # Publisher for frame metadata (replaces /spool/current_frame_index)
            self._metadata_pub = self.create_publisher(
                String,
                '/spool/current_frame_metadata',
                control_qos
            )
            
            # Subscriber for processing READY
            self._ready_sub = self.create_subscription(
                String,
                '/processing_ready',
                self._ready_callback,
                ready_qos
            )
            
            # Subscriber for processing ACK
            self._ack_sub = self.create_subscription(
                String,
                '/processing_ack',
                self._ack_callback,
                control_qos
            )
            
            # Optional: Pull request topic (for external control)
            self._request_sub = self.create_subscription(
                UInt32,
                '/spool/request_next',
                self._request_callback,
                control_qos
            )
            
            logger.info("[SpoolProcessor] ROS2 topics configured: "
                       "/spool_image_ch_0 (pub, RELIABLE), "
                       "/spool/current_frame_metadata (pub, RELIABLE), "
                       "/processing_ready (sub, TRANSIENT_LOCAL), "
                       "/processing_ack (sub, RELIABLE)")
    
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
            logger.info(format_structured_log(
                "[SpoolProcessor] Loaded persisted state",
                last_index=loaded_state.last_published_index,
                last_segment=loaded_state.last_published_segment,
                prev_session=loaded_state.session_id[:8]
            ))
        
        # Initialize frame generator
        self._init_frame_generator()
        
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
            self._state = ProcessorState.STOPPED
        
        # Wake up any waiting threads
        self._ack_received.set()
        self._ready_received.set()
        
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
            final_msg = format_structured_log(
                "[SpoolProcessor] Final stats",
                session=self._session_id[:8],
                seq=self._seq_counter,
                processed=self._frames_processed,
                retried=self._frames_retried,
                skipped=self._frames_skipped,
                timeouts=self._ack_timeouts,
                ack_rejected=self._ack_rejected_stale,
                segments=self._segments_processed,
                sps_pps_prepends=self._sps_pps_prepends,
                anomalies_gap=self._anomalies_gap,
                anomalies_dup=self._anomalies_dup
            )
            logger.info(final_msg)
        
        logger.info("[SpoolProcessor] Stopped")
    
    def _init_frame_generator(self):
        """
        Initialize the frame generator from oldest segment.
        
        V7: Supports seeking to last published frame + 1 for restart continuity.
        """
        oldest = self._reader.get_oldest_segment()
        if oldest is not None:
            logger.info(f"[SpoolProcessor] Starting from segment {oldest}")
            self._frame_generator = self._reader.read_frames(start_segment=oldest)
            # Mark that we need SPS/PPS for the first frame of this segment
            self._segment_needs_sps_pps = True
            self._current_segment = oldest
            
            # V7: If we have persisted state, skip already-published frames
            if self._last_published_index >= 0:
                target_index = self._last_published_index + 1
                skipped = 0
                logger.info(format_structured_log(
                    "[SpoolProcessor] Seeking to resume position",
                    last_published=self._last_published_index,
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
                        
                        if frame.index < target_index:
                            skipped += 1
                            continue
                        else:
                            # Found target or beyond - recreate generator starting here
                            logger.info(format_structured_log(
                                "[SpoolProcessor] Resume position reached",
                                skipped_frames=skipped,
                                next_index=frame.index
                            ))
                            
                            # Create a generator that yields this frame first, then continues
                            def resume_generator(first_frame):
                                yield first_frame
                                while True:
                                    try:
                                        yield next(self._frame_generator)
                                    except StopIteration:
                                        break
                            
                            self._frame_generator = resume_generator(frame)
                            break
                            
                except StopIteration:
                    logger.warning(format_structured_log(
                        "[SpoolProcessor] Reached end of spool while seeking",
                        skipped_frames=skipped,
                        target_was=target_index
                    ))
                    self._frame_generator = iter([])
            else:
                # Pre-scan for SPS/PPS if we don't have cached values yet
                # This is critical for decoder initialization on startup
                if self._cached_sps is None or self._cached_pps is None:
                    self._prescan_for_sps_pps()
        else:
            logger.warning("[SpoolProcessor] No segments available")
            self._frame_generator = iter([])
    
    def _prescan_for_sps_pps(self):
        """
        Pre-scan frames to find and cache SPS/PPS NAL units.
        
        This is called on startup to ensure we have SPS/PPS available
        for the decoder before sending any frames. Critical for decoder
        initialization.
        """
        logger.info("[SpoolProcessor] Pre-scanning for SPS/PPS NAL units...")
        
        # Read up to 100 frames looking for SPS/PPS
        frames_scanned = 0
        temp_frames = []
        max_scan = 100
        
        try:
            while frames_scanned < max_scan:
                try:
                    frame = next(self._frame_generator)
                    temp_frames.append(frame)
                    frames_scanned += 1
                    
                    # Try to extract SPS/PPS from this frame
                    sps, pps = extract_sps_pps(frame.data)
                    if sps:
                        self._cached_sps = sps
                        logger.info(f"[SpoolProcessor] Found and cached SPS from frame {frame.index} during pre-scan")
                    if pps:
                        self._cached_pps = pps
                        logger.info(f"[SpoolProcessor] Found and cached PPS from frame {frame.index} during pre-scan")
                    
                    # If we found both, we're done
                    if self._cached_sps and self._cached_pps:
                        logger.info(f"[SpoolProcessor] Pre-scan complete: found SPS/PPS after scanning {frames_scanned} frames")
                        break
                        
                except StopIteration:
                    logger.warning(f"[SpoolProcessor] Pre-scan reached end of spool after {frames_scanned} frames")
                    break
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error during SPS/PPS pre-scan: {e}")
        
        # Recreate generator with buffered frames at the front
        # This avoids the "generator already executing" issue by creating a fresh generator
        # from the original source and prepending the buffered frames
        oldest = self._current_segment
        if oldest is not None:
            # Get a fresh generator from the reader
            fresh_generator = self._reader.read_frames(start_segment=oldest)
            
            # Create iterator that yields buffered frames first, then continues with fresh generator
            def buffered_generator():
                # First yield all buffered frames
                for frame in temp_frames:
                    yield frame
                # Then yield from fresh generator, skipping frames we already buffered
                frames_to_skip = len(temp_frames)
                skipped = 0
                try:
                    while True:
                        frame = next(fresh_generator)
                        if skipped < frames_to_skip:
                            skipped += 1
                            continue
                        yield frame
                except StopIteration:
                    pass
            
            self._frame_generator = buffered_generator()
        else:
            # Fallback: just use buffered frames if no segment available
            self._frame_generator = iter(temp_frames)
        
        if self._cached_sps and self._cached_pps:
            logger.info("[SpoolProcessor] Pre-scan successful: SPS/PPS cached and ready for decoder")
        else:
            missing = []
            if not self._cached_sps:
                missing.append("SPS")
            if not self._cached_pps:
                missing.append("PPS")
            logger.warning(f"[SpoolProcessor] Pre-scan incomplete: missing {', '.join(missing)} - decoder may fail to initialize")
    
    def _get_next_frame(self) -> Optional[FrameRecord]:
        """
        Get the next frame from the spool.
        
        V7: Includes retention guard - checks if current segment still exists.
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
                # Reinitialize from oldest available segment
                self._init_frame_generator()
        
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
            return frame
        except StopIteration:
            # Segment exhausted, try to reinitialize from new segments
            with self._stats_lock:
                if self._current_segment >= 0:
                    self._segments_processed += 1
            self._segment_needs_sps_pps = True  # New segment will need SPS/PPS
            
            # Don't call _init_frame_generator which may do prescan again
            # Instead, directly get the next segment and create a fresh generator
            oldest = self._reader.get_oldest_segment()
            if oldest is not None:
                logger.info(f"[SpoolProcessor] Starting from segment {oldest}")
                self._frame_generator = self._reader.read_frames(start_segment=oldest)
                self._current_segment = oldest
                
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
    
    def _publish_frame(self, record: FrameRecord) -> tuple[bool, int, int, int]:
        """
        Publish a frame to the decoder input topic with metadata.
        
        V7: Includes gap/dup detection and optional CRC32 checksum logging.
        
        Returns:
            Tuple of (success: bool, seq: int, sent_time_sec: int, sent_time_nsec: int)
        """
        if not IS_RDK:
            return True, 0, 0, 0
        
        try:
            # V7: Detect gaps and duplicates
            if self._last_published_index >= 0:
                expected_index = self._last_published_index + 1
                if record.index > expected_index:
                    # Gap detected
                    gap_size = record.index - expected_index
                    with self._stats_lock:
                        self._anomalies_gap += 1
                    gap_msg = format_structured_log(
                        "⚠ GAP DETECTED",
                        expected=expected_index,
                        actual=record.index,
                        gap_size=gap_size,
                        total_gaps=self._anomalies_gap
                    )
                    logger.warning(f"[SpoolProcessor] {gap_msg}")
                elif record.index < expected_index:
                    # Duplicate or out-of-order
                    with self._stats_lock:
                        self._anomalies_dup += 1
                    dup_msg = format_structured_log(
                        "⚠ DUPLICATE/OUT-OF-ORDER DETECTED",
                        expected=expected_index,
                        actual=record.index,
                        total_dups=self._anomalies_dup
                    )
                    logger.warning(f"[SpoolProcessor] {dup_msg}")
            
            # Get next sequence number
            with self._seq_lock:
                seq = self._seq_counter
                self._seq_counter += 1
            
            # Get send timestamp
            sent_time_sec, sent_time_nsec = get_current_time_ros()
            
            # Publish frame metadata for ACK correlation
            metadata = FrameMetadata(
                frame_index=record.index,
                session_id=self._session_id,
                seq=seq,
                sent_time_sec=sent_time_sec,
                sent_time_nsec=sent_time_nsec,
                segment_num=self._current_segment
            )
            metadata_msg = String()
            metadata_msg.data = frame_metadata_to_ros_string(metadata)
            self._metadata_pub.publish(metadata_msg)
            
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
            
            # V7: Update last published index (for gap/dup detection and state persistence)
            self._last_published_index = record.index
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
            
            return True, seq, sent_time_sec, sent_time_nsec
            
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error publishing frame: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, 0, 0, 0
    
    def _wait_for_ack(self, expected_seq: int, expected_frame_index: int, timeout: float) -> bool:
        """
        Wait for ACK for a specific frame sequence.
        
        Args:
            expected_seq: Expected sequence number in ACK
            expected_frame_index: Expected frame index
            timeout: Maximum time to wait
            
        Returns:
            True if valid ACK received, False on timeout
        """
        self._ack_received.clear()
        start_time = time.time()
        logger.debug(f"[SpoolProcessor] Waiting for ACK: seq={expected_seq}, "
                    f"frame_index={expected_frame_index}, timeout={timeout}s")
        
        while self._running:
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                logger.warning(f"[SpoolProcessor] ⏱ ACK timeout: seq={expected_seq}, "
                             f"frame_index={expected_frame_index}, timeout={timeout}s")
                return False
            
            if self._ack_received.wait(timeout=min(remaining, 1.0)):
                elapsed = time.time() - start_time
                
                if self._last_ack is None:
                    logger.warning("[SpoolProcessor] ⚠ ACK event set but no ACK data available")
                    continue
                
                ack = self._last_ack
                
                # Validate session ID
                if ack.session_id != self._session_id:
                    with self._stats_lock:
                        self._ack_rejected_stale += 1
                    logger.warning(f"[SpoolProcessor] ⚠ ACK rejected: wrong session_id. "
                                 f"Expected {self._session_id[:8]}, got {ack.session_id[:8]}. "
                                 f"Stale ACK count: {self._ack_rejected_stale}")
                    self._ack_received.clear()
                    continue
                
                # Validate sequence or frame index
                if ack.seq == expected_seq and ack.frame_index == expected_frame_index:
                    # Perfect match
                    with self._stats_lock:
                        self._ack_accepted += 1
                    logger.debug(f"[SpoolProcessor] ✓ ACK matched: seq={ack.seq}, "
                              f"frame_index={ack.frame_index}, elapsed={elapsed:.3f}s")
                    self._last_ack_time = time.time()
                    return True
                elif ack.frame_index == expected_frame_index:
                    # Frame index matches but seq doesn't - still accept
                    with self._stats_lock:
                        self._ack_accepted += 1
                    logger.debug(f"[SpoolProcessor] ✓ ACK matched (frame_index): seq={ack.seq} "
                              f"(expected {expected_seq}), frame_index={ack.frame_index}, elapsed={elapsed:.3f}s")
                    self._last_ack_time = time.time()
                    return True
                else:
                    # Neither matches - continue waiting
                    logger.debug(f"[SpoolProcessor] ACK mismatch: got seq={ack.seq}, frame_index={ack.frame_index}; "
                               f"expected seq={expected_seq}, frame_index={expected_frame_index}")
                    self._ack_received.clear()
                    continue
        
        return False
    
    def _ready_callback(self, msg):
        """Callback for processing READY messages."""
        try:
            ready = processing_ready_from_ros_string(msg.data)
            self._consumer_session_id = ready.session_id
            self._ready_received.set()
            logger.info(f"[SpoolProcessor] ✓ READY received from consumer: session_id={ready.session_id[:8]}")
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error parsing READY message: {e}")
    
    def _ack_callback(self, msg):
        """Callback for processing ACK messages."""
        try:
            ack = processing_ack_from_ros_string(msg.data)
            self._last_ack = ack
            self._ack_received.set()
            logger.debug(f"[SpoolProcessor] ACK callback: seq={ack.seq}, frame_index={ack.frame_index}, "
                        f"session_id={ack.session_id[:8]}")
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error parsing ACK message: {e}")
    
    def _request_callback(self, msg):
        """Callback for external pull requests (optional feature)."""
        # This allows external control of frame advancement
        logger.debug(f"[SpoolProcessor] Received request {msg.data}")
    
    def _wait_for_consumer_ready(self) -> bool:
        """
        Wait for consumer (BagCounterApp) to be ready before processing.
        
        The consumer publishes a READY message with the session_id it's prepared
        to serve. This ensures proper startup handshake.
        
        Returns:
            True if consumer is ready, False if timeout
        """
        with self._state_lock:
            self._state = ProcessorState.WAITING_FOR_READY
        
        logger.info(f"[SpoolProcessor] 🔄 Session started: session_id={self._session_id}")
        logger.info(f"[SpoolProcessor] Waiting for consumer READY (timeout: {self.config.startup_grace_period}s)...")
        
        # Wait for READY signal from consumer
        if self._ready_received.wait(timeout=self.config.startup_grace_period):
            # Check if consumer is ready for our session or any session
            if self._consumer_session_id:
                logger.info(f"[SpoolProcessor] ✓ Consumer READY: consumer_session={self._consumer_session_id[:8]}, "
                          f"processor_session={self._session_id[:8]}")
            return True
        
        # Timeout - proceed anyway with warning
        logger.warning(f"[SpoolProcessor] ⚠ Consumer READY timeout ({self.config.startup_grace_period}s) - "
                      "proceeding without explicit READY signal. Consumer may not be synchronized.")
        return False
    
    def _processor_loop(self):
        """
        Main processing loop - dispatches to ACK-free or legacy mode.
        """
        logger.info("[SpoolProcessor] Processing loop started")
        
        # Startup synchronization: Wait for consumer READY
        self._wait_for_consumer_ready()
        
        if self.config.ack_free_mode:
            self._processor_loop_ack_free()
        else:
            self._processor_loop_legacy()
        
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
        - Watchdog for stalled publishing
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
            adaptive_pacing=self.config.enable_adaptive_pacing
        ))
        
        frame_interval = 1.0 / self._current_target_fps
        last_publish_time = 0.0
        
        with self._state_lock:
            self._state = ProcessorState.PUBLISHING
        
        while self._running:
            try:
                # V7: Compute spool lag
                segments = self._reader.list_segments()
                newest_segment = max(segments) if segments else None
                spool_lag = 0
                if newest_segment is not None and self._current_segment >= 0:
                    spool_lag = newest_segment - self._current_segment
                
                # V7: Check lag thresholds and adaptive pacing
                if self.config.enable_adaptive_pacing and spool_lag > self.config.spool_lag_error_threshold:
                    # High lag - reduce FPS temporarily
                    old_fps = self._current_target_fps
                    self._current_target_fps = max(
                        self.config.adaptive_fps_min,
                        self._current_target_fps * 0.8
                    )
                    if old_fps != self._current_target_fps:
                        logger.warning(format_structured_log(
                            "[SpoolProcessor] 🐢 Adaptive pacing: Reducing FPS due to high lag",
                            spool_lag=spool_lag,
                            old_fps=old_fps,
                            new_fps=self._current_target_fps
                        ))
                        frame_interval = 1.0 / self._current_target_fps
                elif self.config.enable_adaptive_pacing and spool_lag < self.config.spool_lag_warn_threshold:
                    # Lag healthy - restore FPS
                    if self._current_target_fps < self.config.target_fps:
                        self._current_target_fps = self.config.target_fps
                        logger.info(format_structured_log(
                            "[SpoolProcessor] 🚀 Adaptive pacing: Restoring FPS",
                            spool_lag=spool_lag,
                            fps=self._current_target_fps
                        ))
                        frame_interval = 1.0 / self._current_target_fps
                
                # V7: Watchdog - check for stalled publishing
                current_time = time.time()
                if self._last_publish_time > 0 and (current_time - self._last_publish_time) > self.config.watchdog_timeout:
                    watchdog_msg = format_structured_log(
                        "🔴 WATCHDOG: No frames published recently",
                        stalled_seconds=current_time - self._last_publish_time,
                        threshold=self.config.watchdog_timeout
                    )
                    throttled_log(
                        logger.error,
                        f"[SpoolProcessor] {watchdog_msg}",
                        key="watchdog",
                        throttle_dict=self._throttle_log_dict,
                        min_interval=10.0
                    )
                
                # Get next frame
                frame = self._get_next_frame()
                
                if frame is None:
                    # Spool is empty, wait and retry
                    with self._state_lock:
                        self._state = ProcessorState.SPOOL_EMPTY
                    logger.debug("[SpoolProcessor] Spool empty, waiting for new frames...")
                    time.sleep(self.config.poll_interval)
                    with self._state_lock:
                        self._state = ProcessorState.PUBLISHING
                    continue
                
                self._current_frame = frame
                self._current_frame_index = frame.index
                
                # Pace to target FPS (non-blocking pacing)
                current_time = time.time()
                elapsed = current_time - last_publish_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
                # Publish frame (no ACK wait)
                success, seq, _, _ = self._publish_frame(frame)
                last_publish_time = time.time()
                
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
    
    def _processor_loop_legacy(self):
        """
        Legacy ACK-Based Processing Loop (Deprecated).
        
        This loop waits for ACK before sending next frame:
        1. Get next frame from spool
        2. Publish frame
        3. Wait for ACK (with retry)
        4. Repeat
        
        WARNING: This mode can cause blocking, deadlocks, and throughput issues.
        Use ACK-free mode for production.
        """
        logger.warning("[SpoolProcessor] LEGACY ACK mode active (deprecated) - consider enabling ACK-free mode")
        
        while self._running:
            try:
                # Get next frame
                frame = self._get_next_frame()
                
                if frame is None:
                    # Spool is empty, wait and retry
                    with self._state_lock:
                        self._state = ProcessorState.SPOOL_EMPTY
                    logger.debug("[SpoolProcessor] Spool empty, waiting for new frames...")
                    time.sleep(self.config.poll_interval)
                    continue
                
                self._current_frame = frame
                self._current_frame_index = frame.index
                
                # Process frame with retry logic (waits for ACK)
                success = self._process_frame_with_retry(frame)
                
                if success:
                    with self._stats_lock:
                        self._frames_processed += 1
                else:
                    with self._stats_lock:
                        self._frames_skipped += 1
                    logger.warning(f"[SpoolProcessor] Frame {frame.index} skipped after retries")
                
                # Log stats periodically
                self._maybe_log_stats()
                
            except Exception as e:
                logger.error(f"[SpoolProcessor] Error in legacy loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(1.0)
    
    def _process_frame_with_retry(self, frame: FrameRecord) -> bool:
        """
        Process a frame with retry logic (Legacy ACK mode only).
        
        Args:
            frame: Frame record to process
            
        Returns:
            True if successfully processed (ACK received), False otherwise
        """
        retries = 0
        seq = 0
        sent_time_sec = 0
        sent_time_nsec = 0
        
        while retries <= self.config.retry_count and self._running:
            with self._state_lock:
                self._state = ProcessorState.IDLE
            
            # Publish frame (gets new seq number each time)
            success, seq, sent_time_sec, sent_time_nsec = self._publish_frame(frame)
            if not success:
                logger.warning(f"[SpoolProcessor] 🔴 Failed to publish frame {frame.index}")
                retries += 1
                continue
            
            with self._state_lock:
                self._state = ProcessorState.WAITING_FOR_ACK
            
            # Wait for ACK with session and seq validation
            if self._wait_for_ack(seq, frame.index, self.config.ack_timeout):
                return True
            
            # Timeout - retry
            with self._stats_lock:
                self._ack_timeouts += 1
            
            if retries < self.config.retry_count:
                with self._stats_lock:
                    self._frames_retried += 1
                logger.warning(f"[SpoolProcessor] ⏱ ACK timeout for frame {frame.index}, seq={seq}, "
                              f"retry {retries + 1}/{self.config.retry_count}")
            
            retries += 1
        
        logger.error(f"[SpoolProcessor] 🔴 Frame {frame.index} failed after {retries} attempts")
        return False
    
    def _maybe_log_stats(self):
        """Log statistics periodically with structured format (V7)."""
        current_time = time.time()
        
        # Regular stats every 10 seconds
        if current_time - self._last_stats_time >= self.config.stats_interval:
            with self._stats_lock:
                # Calculate time since last successful ACK (watchdog info - legacy mode)
                ack_staleness = current_time - self._last_ack_time if self._last_ack_time > 0 else 0.0
                
                # Compute spool lag
                segments = self._reader.list_segments()
                newest_segment = max(segments) if segments else None
                spool_lag = 0
                if newest_segment is not None and self._current_segment >= 0:
                    spool_lag = newest_segment - self._current_segment
                
                # Structured stats logging
                stats_msg = format_structured_log(
                    "[SpoolProcessor] Stats",
                    session=self._session_id[:8],
                    seq=self._seq_counter,
                    frames_processed=self._frames_processed,
                    frames_retried=self._frames_retried,
                    frames_skipped=self._frames_skipped,
                    anomalies_gap=self._anomalies_gap,
                    anomalies_dup=self._anomalies_dup,
                    ack_timeouts=self._ack_timeouts,
                    ack_rejected=self._ack_rejected_stale,
                    segments_processed=self._segments_processed,
                    sps_pps_prepends=self._sps_pps_prepends,
                    sps_pps_missing=self._sps_pps_missing_critical,
                    state=self._state.value
                )
                logger.info(stats_msg)
                
                # Log spool status with lag
                spool_msg = format_structured_log(
                    "[SpoolProcessor] Spool",
                    total_segments=len(segments),
                    current_segment=self._current_segment,
                    current_frame=self._current_frame_index,
                    spool_lag=spool_lag,
                    last_ack_age=f"{ack_staleness:.1f}"
                )
                logger.info(spool_msg)
                
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
                
                # Watchdog warning if ACKs are stale (legacy mode)
                if not self.config.ack_free_mode and ack_staleness > self.config.ack_timeout * 2 and self._last_ack_time > 0:
                    watchdog_msg = format_structured_log(
                        "⚠ WATCHDOG: No ACK received",
                        stalled_seconds=f"{ack_staleness:.1f}",
                        threshold=self.config.ack_timeout * 2
                    )
                    logger.warning(f"[SpoolProcessor] {watchdog_msg}")
            
            self._last_stats_time = current_time
        
        # Detailed stats every 2 minutes (120 seconds)
        if current_time - self._last_detailed_stats_time >= 120.0:
            with self._stats_lock:
                ack_accepted = self._ack_accepted
                ack_rejected = self._ack_rejected_stale
                total_acks = ack_accepted + ack_rejected
                ack_accept_rate = (ack_accepted / total_acks * 100) if total_acks > 0 else 0.0
                ack_reject_rate = (ack_rejected / total_acks * 100) if total_acks > 0 else 0.0
                
                # Get spool information for lag detection
                segments = self._reader.list_segments()
                oldest_segment = self._reader.get_oldest_segment()
                newest_segment = max(segments) if segments else None
                
                # Calculate spool lag: difference between newest and current segment
                spool_lag = 0
                if newest_segment is not None and self._current_segment >= 0:
                    spool_lag = newest_segment - self._current_segment
                
                logger.info("=" * 80)
                logger.info(f"[SpoolProcessor] 📊 Detailed Statistics (2-minute summary)")
                logger.info(f"  Session: {self._session_id}")
                logger.info(f"  ACK Statistics:")
                logger.info(f"    - Accepted: {ack_accepted} ({ack_accept_rate:.1f}%)")
                logger.info(f"    - Rejected (stale): {ack_rejected} ({ack_reject_rate:.1f}%)")
                logger.info(f"    - Total: {total_acks}")
                logger.info(f"  Frame Processing:")
                logger.info(f"    - Processed: {self._frames_processed}")
                logger.info(f"    - Retried: {self._frames_retried}")
                logger.info(f"    - Skipped: {self._frames_skipped}")
                logger.info(f"    - Timeouts: {self._ack_timeouts}")
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
    
    def get_state(self) -> ProcessorState:
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
