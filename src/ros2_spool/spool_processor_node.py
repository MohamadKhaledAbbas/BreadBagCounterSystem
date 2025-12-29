#!/usr/bin/env python3
"""
Spool Processor Node for Accuracy Mode.

This node reads H.264 frames from the spool and publishes them at a
controlled pace, waiting for ACK from BagCounterApp before sending
the next frame. This implements strict backpressure to prevent drops.

Architecture:
1. Spool Reader: Reads frames from oldest closed segments
2. Pull Mechanism: Topic-based request/response (no custom .srv required)
   - Request topic: /spool/request_next (std_msgs/UInt32 with request_id)
   - Response topic: /spool/next_frame (img_msgs/H26XFrame)
3. Pump: Publishes frames to /spool_image_ch_0 for decoder
4. ACK Handler: Waits for /processing_ack before next frame

Usage:
    python -m src.ros2_spool.spool_processor_node

Configuration (via database config table):
    spool_dir: Directory for spool files (default: /home/sunrise/BreadCounting/data/spool)
    spool_ack_timeout: Timeout waiting for ACK in seconds (default: 10.0)
    spool_retry_count: Number of retries before advancing (default: 2)
"""

import os
import sys
import time
import signal
import threading
from typing import Optional, Generator
from dataclasses import dataclass
from enum import Enum
from collections import deque

from src.config.settings import AppConfig

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK
from src.spool.segment_io import SegmentReader, FrameRecord
from src.spool.h264_nal import extract_sps_pps, is_idr_frame, detect_frame_type
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
    WAITING_FOR_ACK = "waiting_for_ack"
    WAITING_FOR_READY = "waiting_for_ready"  # Waiting for consumer ready signal
    SPOOL_EMPTY = "spool_empty"
    STOPPED = "stopped"


@dataclass
class InflightFrame:
    """
    Tracking information for a frame in the inflight window.
    
    Attributes:
        seq: Sequence number assigned when published
        frame_index: Frame index from the spool
        segment_num: Segment number
        publish_time: Timestamp when frame was published
        retry_count: Number of retries so far
        acked: Whether this frame has been acknowledged
        frame_record: The original FrameRecord (for retry)
    """
    seq: int
    frame_index: int
    segment_num: int
    publish_time: float
    retry_count: int
    acked: bool
    frame_record: 'FrameRecord'  # Forward reference


# Default configuration values
DEFAULT_SPOOL_DIR = "/home/sunrise/BreadCounting/data/spool"
DEFAULT_ACK_TIMEOUT = 10.0  # Reduced from 30.0 - should be much faster with FIFO correlation
DEFAULT_RETRY_COUNT = 2
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_STATS_INTERVAL = 10.0
DEFAULT_STARTUP_GRACE_PERIOD = 10.0  # Seconds to wait for consumer to start
DEFAULT_SPS_PPS_PREPEND = True  # Prepend cached SPS/PPS to first frame of segment
DEFAULT_INFLIGHT_WINDOW = 1  # Default to 1 for backward compatibility (strict one-at-a-time)


@dataclass
class ProcessorConfig:
    """Configuration for the spool processor."""
    spool_dir: str = DEFAULT_SPOOL_DIR
    ack_timeout: float = DEFAULT_ACK_TIMEOUT
    retry_count: int = DEFAULT_RETRY_COUNT
    poll_interval: float = DEFAULT_POLL_INTERVAL
    stats_interval: float = DEFAULT_STATS_INTERVAL
    startup_grace_period: float = DEFAULT_STARTUP_GRACE_PERIOD
    prepend_sps_pps: bool = DEFAULT_SPS_PPS_PREPEND
    inflight_window: int = DEFAULT_INFLIGHT_WINDOW


def load_config_from_db(db_path: str = AppConfig.db_path) -> ProcessorConfig:
    """Load spool processor configuration from database config table."""
    try:
        db = DatabaseManager(db_path)
        
        spool_dir = db.get_config_value(constants.spool_dir)
        ack_timeout = db.get_config_value(constants.spool_ack_timeout)
        retry_count = db.get_config_value(constants.spool_retry_count)
        inflight_window = db.get_config_value(constants.spool_inflight_window)
        
        db.close()
        
        return ProcessorConfig(
            spool_dir=spool_dir if spool_dir else DEFAULT_SPOOL_DIR,
            ack_timeout=float(ack_timeout) if ack_timeout else DEFAULT_ACK_TIMEOUT,
            retry_count=int(retry_count) if retry_count else DEFAULT_RETRY_COUNT,
            inflight_window=int(inflight_window) if inflight_window else DEFAULT_INFLIGHT_WINDOW,
        )
    except Exception as e:
        logger.warning(f"[SpoolProcessor] Failed to load config from DB: {e}, using defaults")
        return ProcessorConfig()


class SpoolProcessorNode(Node):
    """
    ROS2 Node that processes spooled H.264 frames with configurable backpressure.
    
    The processor supports windowed ACK processing (configurable via inflight_window):
    - inflight_window=1: Strict one-at-a-time processing (backward compatible)
    - inflight_window>1: Multiple frames can be in-flight simultaneously
    
    Processing flow:
    1. Read frames from spool
    2. Publish up to inflight_window frames
    3. Track ACKs (can arrive out-of-order)
    4. Retire frames from head of window when acknowledged
    5. Handle timeouts and retries per frame
    
    This design implements pull-based processing with configurable parallelism
    to balance throughput and backpressure.
    
    Production Reliability Features:
    - Startup synchronization: Waits for consumer before processing
    - Watchdog: Auto-recovery from stuck ACK states
    - SPS/PPS caching: Prepends to segment boundaries for decoder init
    - Graceful degradation: Advances after max retries to prevent deadlock
    - Out-of-order ACK handling: Supports parallel processing in consumer
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__('spool_processor')
        
        # Load configuration from database if not provided
        self.config = config or load_config_from_db()
        
        # Generate unique session ID for this run
        self._session_id = generate_session_id()
        
        logger.info(f"[SpoolProcessor] Initializing with config: "
                   f"spool_dir={self.config.spool_dir}, "
                   f"ack_timeout={self.config.ack_timeout}s, "
                   f"retry_count={self.config.retry_count}, "
                   f"inflight_window={self.config.inflight_window}, "
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
        
        # Inflight window tracking (for windowed ACK / backpressure)
        self._inflight_frames: deque[InflightFrame] = deque()
        self._inflight_lock = threading.Lock()  # Protects _inflight_frames
        
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
        self._out_of_order_acks = 0  # ACKs received out of order
        self._segments_processed = 0
        self._sps_pps_prepends = 0
        self._last_stats_time = time.time()
        self._last_detailed_stats_time = time.time()  # For 2-minute detailed stats
        self._stats_lock = threading.Lock()
        
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
                logger.warning("[SpoolProcessor] Processor thread did not stop in time")
        
        # Log final stats
        with self._stats_lock:
            inflight_count = self._get_inflight_count()
            logger.info(f"[SpoolProcessor] Final stats: "
                       f"session={self._session_id[:8]}, "
                       f"seq={self._seq_counter}, "
                       f"processed={self._frames_processed}, "
                       f"retried={self._frames_retried}, "
                       f"skipped={self._frames_skipped}, "
                       f"timeouts={self._ack_timeouts}, "
                       f"ack_rejected={self._ack_rejected_stale}, "
                       f"out_of_order_acks={self._out_of_order_acks}, "
                       f"inflight={inflight_count}, "
                       f"segments={self._segments_processed}, "
                       f"sps_pps_prepends={self._sps_pps_prepends}")
        
        logger.info("[SpoolProcessor] Stopped")
    
    def _init_frame_generator(self):
        """Initialize the frame generator from oldest segment."""
        oldest = self._reader.get_oldest_segment()
        if oldest is not None:
            logger.info(f"[SpoolProcessor] Starting from segment {oldest}")
            self._frame_generator = self._reader.read_frames(start_segment=oldest)
            # Mark that we need SPS/PPS for the first frame of this segment
            self._segment_needs_sps_pps = True
            self._current_segment = oldest
            
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
        """Get the next frame from the spool."""
        if self._frame_generator is None:
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
        
        # Check if frame already has SPS/PPS
        # detect_frame_type is imported at module level
        frame_has_idr, frame_has_sps, frame_has_pps = False, False, False
        try:
            frame_has_idr, frame_has_sps, frame_has_pps = detect_frame_type(data)
        except Exception:
            pass
        
        # If frame already has SPS/PPS, no need to prepend
        if frame_has_sps and frame_has_pps:
            self._segment_needs_sps_pps = False
            return data
        
        # Prepend cached SPS/PPS if available
        prepended = bytearray()
        if self._cached_sps:
            prepended.extend(self._cached_sps)
            logger.debug("[SpoolProcessor] Prepending cached SPS to frame")
        if self._cached_pps:
            prepended.extend(self._cached_pps)
            logger.debug("[SpoolProcessor] Prepending cached PPS to frame")
        
        if prepended:
            with self._stats_lock:
                self._sps_pps_prepends += 1
            self._segment_needs_sps_pps = False
            return bytes(prepended) + data
        
        # No cached SPS/PPS available - frame must be self-contained or decoder will fail
        if not frame_has_idr:
            logger.warning("[SpoolProcessor] First frame of segment has no SPS/PPS NAL units and no cached SPS/PPS available - decoder may fail to initialize")
        
        self._segment_needs_sps_pps = False
        return data
    
    def _publish_frame(self, record: FrameRecord) -> tuple[bool, int, int, int]:
        """
        Publish a frame to the decoder input topic with metadata.
        
        Returns:
            Tuple of (success: bool, seq: int, sent_time_sec: int, sent_time_nsec: int)
        """
        if not IS_RDK:
            return True, 0, 0, 0
        
        try:
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
            
            # Structured logging - use debug for regular frames, info for milestones
            if seq % 100 == 0:
                logger.info(f"[SpoolProcessor] 📤 Milestone: published {seq} frames, "
                          f"current: index={record.index}, session={self._session_id[:8]}, "
                          f"segment={self._current_segment}, data_len={len(frame_data)}")
            else:
                logger.debug(f"[SpoolProcessor] 📤 Frame published: index={record.index}, seq={seq}, "
                           f"session={self._session_id[:8]}, segment={self._current_segment}, data_len={len(frame_data)}")
            
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
        """
        Callback for processing ACK messages.
        
        Handles windowed ACKs:
        - Validates session ID
        - Marks corresponding inflight frame as acked
        - Triggers window retirement if needed
        """
        try:
            ack = processing_ack_from_ros_string(msg.data)
            
            # Validate session ID first
            if ack.session_id != self._session_id:
                with self._stats_lock:
                    self._ack_rejected_stale += 1
                logger.warning(f"[SpoolProcessor] ⚠ ACK rejected: wrong session_id. "
                             f"Expected {self._session_id[:8]}, got {ack.session_id[:8]}. "
                             f"Stale ACK count: {self._ack_rejected_stale}")
                return
            
            # Find and mark the corresponding inflight frame as acked
            # Note: Linear search is acceptable for small windows (1-10 frames typical)
            # For larger windows, consider adding a seq->frame mapping dict
            with self._inflight_lock:
                found = False
                for i, inflight in enumerate(self._inflight_frames):
                    if inflight.seq == ack.seq and inflight.frame_index == ack.frame_index:
                        if not inflight.acked:
                            inflight.acked = True
                            found = True
                            with self._stats_lock:
                                self._ack_accepted += 1
                                # Track if this is out-of-order (not at head of queue)
                                if i > 0:
                                    self._out_of_order_acks += 1
                            logger.debug(f"[SpoolProcessor] ✓ ACK marked: seq={ack.seq}, "
                                       f"frame_index={ack.frame_index}, position={i}, "
                                       f"out_of_order={i > 0}")
                            break
                        else:
                            # Duplicate ACK
                            logger.debug(f"[SpoolProcessor] Duplicate ACK: seq={ack.seq}, "
                                       f"frame_index={ack.frame_index}")
                            return
                
                if not found:
                    logger.debug(f"[SpoolProcessor] ACK for unknown frame: seq={ack.seq}, "
                               f"frame_index={ack.frame_index} (may have already been retired)")
            
            # Signal that we received an ACK (for legacy wait_for_ack compatibility)
            self._last_ack = ack
            self._ack_received.set()
            self._last_ack_time = time.time()
            
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error parsing ACK message: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
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
    
    def _retire_acked_frames(self):
        """
        Retire (remove) frames from the head of the inflight window that have been acked.
        
        This maintains ordering: we only remove frames from the head while they are
        contiguously acked. Out-of-order ACKs are marked but not retired until all
        earlier frames are acked.
        """
        with self._inflight_lock:
            while self._inflight_frames:
                head = self._inflight_frames[0]
                if head.acked:
                    self._inflight_frames.popleft()
                    # Count as processed when successfully retired
                    with self._stats_lock:
                        self._frames_processed += 1
                    logger.debug(f"[SpoolProcessor] Retired frame: seq={head.seq}, "
                               f"frame_index={head.frame_index}")
                else:
                    # Head frame not acked yet, stop retiring
                    break
    
    def _check_and_retry_timeouts(self):
        """
        Check for timed-out frames in the inflight window and retry or skip them.
        
        For each frame that has exceeded ack_timeout:
        - If retry_count not exceeded: increment retry counter and republish
        - If retry_count exceeded: mark as acked (to allow retirement) and skip
        
        This ensures the pipeline doesn't get stuck on a single failed frame.
        """
        current_time = time.time()
        
        # Note: We create a list copy to safely iterate while holding the lock
        # and potentially calling _publish_frame. For small windows (1-10 frames),
        # this copy is inexpensive and avoids complex iteration patterns.
        with self._inflight_lock:
            for inflight in list(self._inflight_frames):
                # Skip already acked frames
                if inflight.acked:
                    continue
                
                # Check if frame has timed out
                age = current_time - inflight.publish_time
                if age > self.config.ack_timeout:
                    # Frame has timed out
                    if inflight.retry_count < self.config.retry_count:
                        # Retry: republish the frame
                        inflight.retry_count += 1
                        with self._stats_lock:
                            self._frames_retried += 1
                            self._ack_timeouts += 1
                        
                        logger.warning(f"[SpoolProcessor] ⏱ ACK timeout for frame {inflight.frame_index}, "
                                     f"seq={inflight.seq}, retry {inflight.retry_count}/{self.config.retry_count}")
                        
                        # Republish the frame with a new sequence number
                        success, new_seq, _, _ = self._publish_frame(inflight.frame_record)
                        if success:
                            inflight.seq = new_seq
                            inflight.publish_time = current_time
                            logger.debug(f"[SpoolProcessor] Retried frame {inflight.frame_index} with new seq={new_seq}")
                        else:
                            logger.error(f"[SpoolProcessor] Failed to retry frame {inflight.frame_index}")
                    else:
                        # Max retries exceeded, mark as acked to allow retirement
                        inflight.acked = True
                        with self._stats_lock:
                            self._frames_skipped += 1
                            self._ack_timeouts += 1
                        logger.error(f"[SpoolProcessor] 🔴 Frame {inflight.frame_index} skipped after "
                                   f"{inflight.retry_count} retries")
    
    def _can_publish_frame(self) -> bool:
        """
        Check if we can publish a new frame (window not full).
        
        Returns:
            True if inflight window has space for another frame
        """
        with self._inflight_lock:
            return len(self._inflight_frames) < self.config.inflight_window
    
    def _get_inflight_count(self) -> int:
        """Get current number of inflight frames."""
        with self._inflight_lock:
            return len(self._inflight_frames)
    
    def _get_oldest_inflight_age(self) -> float:
        """
        Get age of the oldest inflight frame (for monitoring).
        
        Returns:
            Age in seconds, or 0.0 if no frames inflight
        """
        with self._inflight_lock:
            if not self._inflight_frames:
                return 0.0
            oldest = self._inflight_frames[0]
            return time.time() - oldest.publish_time
    
    def _processor_loop(self):
        """
        Main processing loop with windowed backpressure.
        
        This loop manages the inflight window:
        1. Wait for consumer startup (startup sync)
        2. While window not full:
           - Get next frame from spool
           - Publish frame and add to inflight window
        3. Retire acked frames from head of window
        4. Check for timeouts and retry/skip as needed
        5. Repeat
        
        Production reliability features:
        - Startup synchronization: Wait for consumer before first frame
        - Windowed ACK: Configurable parallelism (1 = strict serial)
        - Out-of-order ACK handling: Marks frames but retires in order
        - Timeout/retry per frame: Doesn't block other frames
        - Watchdog: Detect and recover from stuck states
        - Graceful degradation: Skip frames after max retries
        """
        logger.info("[SpoolProcessor] Processing loop started")
        
        # Startup synchronization: Wait for consumer READY
        self._wait_for_consumer_ready()
        
        # Track when spool was last empty (to avoid tight loop on empty spool)
        last_spool_empty_time = 0.0
        spool_was_empty = False
        
        while self._running:
            try:
                # Retire any acked frames from the head of the window
                self._retire_acked_frames()
                
                # Check for timeouts and retry/skip
                self._check_and_retry_timeouts()
                
                # Try to fill the window with new frames
                current_time = time.time()
                if self._can_publish_frame():
                    # If spool was recently empty, throttle checks to avoid hammering empty spool
                    # Otherwise, rapidly fill the window to maximize throughput
                    if spool_was_empty and (current_time - last_spool_empty_time < self.config.poll_interval):
                        # Spool was empty recently, wait before checking again
                        pass
                    else:
                        # Get next frame (no throttling when filling window)
                        frame = self._get_next_frame()
                        
                        if frame is None:
                            # Spool is empty
                            spool_was_empty = True
                            last_spool_empty_time = current_time
                            with self._state_lock:
                                self._state = ProcessorState.SPOOL_EMPTY
                            logger.debug("[SpoolProcessor] Spool empty, waiting for new frames...")
                        else:
                            # Got a frame - spool is not empty, reset flag
                            spool_was_empty = False
                            
                            # Publish the frame and add to inflight window
                            self._current_frame = frame
                            self._current_frame_index = frame.index
                            
                            with self._state_lock:
                                self._state = ProcessorState.IDLE
                            
                            success, seq, sent_time_sec, sent_time_nsec = self._publish_frame(frame)
                            
                            if success:
                                # Add to inflight window
                                inflight = InflightFrame(
                                    seq=seq,
                                    frame_index=frame.index,
                                    segment_num=self._current_segment,
                                    publish_time=time.time(),
                                    retry_count=0,
                                    acked=False,
                                    frame_record=frame
                                )
                                
                                with self._inflight_lock:
                                    self._inflight_frames.append(inflight)
                                
                                logger.debug(f"[SpoolProcessor] Added to inflight: seq={seq}, "
                                           f"frame_index={frame.index}, window_size={self._get_inflight_count()}")
                            else:
                                logger.warning(f"[SpoolProcessor] 🔴 Failed to publish frame {frame.index}")
                                # We'll retry on next iteration
                else:
                    # Window is full, just wait a bit before checking again
                    with self._state_lock:
                        self._state = ProcessorState.WAITING_FOR_ACK
                
                # Log stats periodically
                self._maybe_log_stats()
                
                # Small sleep to avoid tight loop
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"[SpoolProcessor] Error in processing loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(1.0)
        
        logger.info("[SpoolProcessor] Processing loop stopped")
    
    def _maybe_log_stats(self):
        """Log statistics periodically."""
        current_time = time.time()
        
        # Regular stats every 10 seconds
        if current_time - self._last_stats_time >= self.config.stats_interval:
            # Get inflight metrics
            inflight_count = self._get_inflight_count()
            oldest_age = self._get_oldest_inflight_age()
            
            with self._stats_lock:
                # Calculate time since last successful ACK (watchdog info)
                ack_staleness = current_time - self._last_ack_time if self._last_ack_time > 0 else 0.0
                
                logger.info(f"[SpoolProcessor] Stats: "
                           f"session={self._session_id[:8]}, "
                           f"seq={self._seq_counter}, "
                           f"processed={self._frames_processed}, "
                           f"retried={self._frames_retried}, "
                           f"skipped={self._frames_skipped}, "
                           f"timeouts={self._ack_timeouts}, "
                           f"ack_rejected={self._ack_rejected_stale}, "
                           f"out_of_order_acks={self._out_of_order_acks}, "
                           f"inflight={inflight_count}/{self.config.inflight_window}, "
                           f"oldest_inflight_age={oldest_age:.1f}s, "
                           f"segments={self._segments_processed}, "
                           f"sps_pps_prepends={self._sps_pps_prepends}, "
                           f"state={self._state.value}")
                
                # Log spool status
                segments = self._reader.list_segments()
                logger.info(f"[SpoolProcessor] Spool: "
                           f"segments={len(segments)}, "
                           f"current_frame={self._current_frame_index}, "
                           f"last_ack_age={ack_staleness:.1f}s")
                
                # Watchdog warning if ACKs are stale
                if ack_staleness > self.config.ack_timeout * 2 and self._last_ack_time > 0:
                    logger.warning(f"[SpoolProcessor] ⚠ WATCHDOG: No ACK received in {ack_staleness:.1f}s - "
                                  "consumer may be stuck or not processing frames")
            
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
                logger.info(f"  Configuration:")
                logger.info(f"    - Inflight Window: {self.config.inflight_window}")
                logger.info(f"    - ACK Timeout: {self.config.ack_timeout}s")
                logger.info(f"    - Retry Count: {self.config.retry_count}")
                logger.info(f"  ACK Statistics:")
                logger.info(f"    - Accepted: {ack_accepted} ({ack_accept_rate:.1f}%)")
                logger.info(f"    - Rejected (stale): {ack_rejected} ({ack_reject_rate:.1f}%)")
                logger.info(f"    - Out-of-order: {self._out_of_order_acks}")
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
