#!/usr/bin/env python3
"""
Spool Processor Node for Accuracy Mode (Credit-Based Flow Control).

This node reads H.264 frames from the spool and publishes them using a
credit-based, best-effort, bounded in-flight window design. Unlike the
previous blocking ACK design, this implementation allows multiple frames
in-flight for higher throughput while maintaining backpressure.

Architecture:
1. Spool Reader: Reads frames from oldest closed segments
2. Credit-Based Publisher: Publishes frames while in_flight < max_in_flight
3. Non-Blocking ACK: ACK callback frees credit without blocking publish loop
4. Timeout Handling: Expired frames are marked and credit is freed

Usage:
    python -m src.ros2_spool.spool_processor_node

Configuration (via database config table):
    spool_dir: Directory for spool files (default: /home/sunrise/BreadCounting/data/spool)
    spool_ack_timeout: Timeout for in-flight frames in seconds (default: 10.0)
    spool_max_in_flight: Maximum frames in-flight (default: 10)
    spool_publish_idle_sleep_ms: Milliseconds to sleep when idle (default: 5)
    spool_empty_poll_interval: Seconds to wait when spool empty (default: 1.0)
"""

import os
import sys
import time
import signal
import threading
from typing import Optional, Generator, Dict
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
    PUBLISHING = "publishing"  # Actively publishing frames (credit available)
    BACKPRESSURE = "backpressure"  # In-flight window full, waiting for ACKs
    WAITING_FOR_READY = "waiting_for_ready"  # Waiting for consumer ready signal
    SPOOL_EMPTY = "spool_empty"
    STOPPED = "stopped"


# Default configuration values
DEFAULT_SPOOL_DIR = "/home/sunrise/BreadCounting/data/spool"
DEFAULT_ACK_TIMEOUT = 10.0  # Timeout for in-flight frames (marks as expired, frees credit)
DEFAULT_RETRY_COUNT = 2  # Deprecated: retained for backward compatibility
DEFAULT_POLL_INTERVAL = 1.0  # Only used when spool is empty
DEFAULT_STATS_INTERVAL = 10.0
DEFAULT_STARTUP_GRACE_PERIOD = 10.0  # Seconds to wait for consumer to start
DEFAULT_SPS_PPS_PREPEND = True  # Prepend cached SPS/PPS to first frame of segment
DEFAULT_MAX_IN_FLIGHT = 20  # Maximum frames in-flight before backpressure (increased for better throughput)
DEFAULT_PUBLISH_IDLE_SLEEP_MS = 1  # Milliseconds to sleep in publish loop when idle (reduced for higher throughput)
DEFAULT_EMPTY_POLL_INTERVAL = 1.0  # Seconds to wait when spool is empty


@dataclass
class ProcessorConfig:
    """Configuration for the spool processor."""
    spool_dir: str = DEFAULT_SPOOL_DIR
    ack_timeout: float = DEFAULT_ACK_TIMEOUT
    retry_count: int = DEFAULT_RETRY_COUNT  # Deprecated: retained for backward compatibility
    poll_interval: float = DEFAULT_POLL_INTERVAL  # Deprecated: use empty_poll_interval
    stats_interval: float = DEFAULT_STATS_INTERVAL
    startup_grace_period: float = DEFAULT_STARTUP_GRACE_PERIOD
    prepend_sps_pps: bool = DEFAULT_SPS_PPS_PREPEND
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    publish_idle_sleep_ms: int = DEFAULT_PUBLISH_IDLE_SLEEP_MS
    empty_poll_interval: float = DEFAULT_EMPTY_POLL_INTERVAL


@dataclass
class InFlightFrame:
    """Tracks a frame that has been published but not yet ACKed."""
    seq: int
    frame_index: int
    sent_time: float
    segment_num: int
    expired: bool = False


def load_config_from_db(db_path: str = AppConfig.db_path) -> ProcessorConfig:
    """Load spool processor configuration from database config table."""
    try:
        db = DatabaseManager(db_path)
        
        spool_dir = db.get_config_value(constants.spool_dir)
        ack_timeout = db.get_config_value(constants.spool_ack_timeout)
        retry_count = db.get_config_value(constants.spool_retry_count)
        max_in_flight = db.get_config_value(constants.spool_max_in_flight)
        publish_idle_sleep_ms = db.get_config_value(constants.spool_publish_idle_sleep_ms)
        empty_poll_interval = db.get_config_value(constants.spool_empty_poll_interval)
        
        db.close()
        
        return ProcessorConfig(
            spool_dir=spool_dir if spool_dir else DEFAULT_SPOOL_DIR,
            ack_timeout=float(ack_timeout) if ack_timeout else DEFAULT_ACK_TIMEOUT,
            retry_count=int(retry_count) if retry_count else DEFAULT_RETRY_COUNT,
            max_in_flight=int(max_in_flight) if max_in_flight else DEFAULT_MAX_IN_FLIGHT,
            publish_idle_sleep_ms=int(publish_idle_sleep_ms) if publish_idle_sleep_ms else DEFAULT_PUBLISH_IDLE_SLEEP_MS,
            empty_poll_interval=float(empty_poll_interval) if empty_poll_interval else DEFAULT_EMPTY_POLL_INTERVAL,
        )
    except Exception as e:
        logger.warning(f"[SpoolProcessor] Failed to load config from DB: {e}, using defaults")
        return ProcessorConfig()


class SpoolProcessorNode(Node):
    """
    ROS2 Node that processes spooled H.264 frames with credit-based flow control.
    
    The processor implements a bounded in-flight window for best-effort throughput:
    1. Continuously publishes frames while in_flight < max_in_flight
    2. ACK callback frees credit (does not block publish loop)
    3. Backpressure naturally occurs when in-flight window fills
    4. Timeout handling marks expired frames and frees credit
    
    This design allows high throughput (target 20 FPS) while maintaining
    backpressure control.
    
    Production Reliability Features:
    - Startup synchronization: Waits for consumer before processing
    - Credit-based flow control: Bounded in-flight window prevents overload
    - Out-of-order ACK handling: Any in-flight frame can be ACKed
    - Timeout handling: Expired frames free credit to prevent deadlock
    - SPS/PPS caching: Prepends to segment boundaries for decoder init
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
                   f"max_in_flight={self.config.max_in_flight}, "
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
        
        # Credit-based flow control: in-flight tracking
        self._in_flight: Dict[int, InFlightFrame] = {}  # seq -> InFlightFrame
        self._in_flight_order: deque = deque()  # Ordered list of seq numbers for FIFO timeout checking
        self._in_flight_lock = threading.Lock()
        
        # SPS/PPS caching for segment boundary handling
        self._cached_sps: Optional[bytes] = None
        self._cached_pps: Optional[bytes] = None
        self._segment_needs_sps_pps: bool = True  # First frame of segment needs SPS/PPS
        
        # State management
        self._state = ProcessorState.WAITING_FOR_READY
        self._state_lock = threading.Lock()
        self._ready_received = threading.Event()
        self._consumer_session_id: Optional[str] = None  # Session ID from consumer's READY
        self._last_ack_time: float = 0.0  # Track last successful ACK for watchdog
        
        # Processing thread
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._frames_processed = 0
        self._frames_published = 0  # Total frames published (includes retries in old design)
        self._frames_skipped = 0
        self._ack_timeouts = 0  # Frames that expired due to timeout
        self._ack_rejected_stale = 0  # ACKs rejected due to wrong session
        self._ack_accepted = 0  # Total ACKs accepted
        self._segments_processed = 0
        self._sps_pps_prepends = 0
        self._last_stats_time = time.time()
        self._last_detailed_stats_time = time.time()  # For 2-minute detailed stats
        self._publish_rate_window: deque = deque(maxlen=100)  # Track publish times for rate estimation
        self._ack_rate_window: deque = deque(maxlen=100)  # Track ACK times for rate estimation
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
        self._ready_received.set()
        
        # Wait for processor thread
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
            if self._processor_thread.is_alive():
                logger.warning("[SpoolProcessor] Processor thread did not stop in time")
        
        # Log final stats
        with self._stats_lock:
            with self._in_flight_lock:
                in_flight_count = len(self._in_flight)
            logger.info(f"[SpoolProcessor] Final stats: "
                       f"session={self._session_id[:8]}, "
                       f"published={self._frames_published}, "
                       f"acked={self._ack_accepted}, "
                       f"processed={self._frames_processed}, "
                       f"skipped={self._frames_skipped}, "
                       f"timeouts={self._ack_timeouts}, "
                       f"ack_rejected={self._ack_rejected_stale}, "
                       f"in_flight={in_flight_count}, "
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
    
    def _publish_frame(self, record: FrameRecord) -> tuple[bool, int]:
        """
        Publish a frame to the decoder input topic with metadata and track in-flight.
        
        Returns:
            Tuple of (success: bool, seq: int)
        """
        if not IS_RDK:
            return True, 0
        
        try:
            # Get next sequence number
            with self._seq_lock:
                seq = self._seq_counter
                self._seq_counter += 1
            
            # Get send timestamp
            sent_time_sec, sent_time_nsec = get_current_time_ros()
            sent_time = time.time()
            
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
            
            # Publish the encoded frame
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
            if isinstance(record.encoding, str):
                encoding_bytes = record.encoding.encode('utf-8')[:12]
            elif isinstance(record.encoding, bytes):
                encoding_bytes = record.encoding[:12]
            else:
                encoding_bytes = bytes(record.encoding)[:12]
            encoding_padded = list(encoding_bytes) + [0] * (12 - len(encoding_bytes))
            frame_msg.encoding = encoding_padded
            
            frame_msg.data = list(frame_data)
            
            self._frame_pub.publish(frame_msg)
            
            # Track in-flight frame
            with self._in_flight_lock:
                in_flight_frame = InFlightFrame(
                    seq=seq,
                    frame_index=record.index,
                    sent_time=sent_time,
                    segment_num=self._current_segment
                )
                self._in_flight[seq] = in_flight_frame
                self._in_flight_order.append(seq)
            
            # Update statistics
            with self._stats_lock:
                self._frames_published += 1
                self._publish_rate_window.append(sent_time)
            
            # Structured logging - use debug for regular frames, info for milestones
            if seq % 100 == 0:
                with self._in_flight_lock:
                    in_flight_count = len(self._in_flight)
                logger.info(f"[SpoolProcessor] 📤 Milestone: published {seq} frames, "
                          f"current: index={record.index}, session={self._session_id[:8]}, "
                          f"segment={self._current_segment}, in_flight={in_flight_count}/{self.config.max_in_flight}")
            else:
                logger.debug(f"[SpoolProcessor] 📤 Frame published: index={record.index}, seq={seq}, "
                           f"session={self._session_id[:8]}, segment={self._current_segment}")
            
            return True, seq
            
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error publishing frame: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, 0
    
    
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
        Callback for processing ACK messages (credit-based).
        
        This callback frees credit for any in-flight frame that is ACKed.
        It does not block the publish loop - credit is simply released
        when ACK arrives.
        """
        try:
            ack = processing_ack_from_ros_string(msg.data)
            
            # Validate session ID
            if ack.session_id != self._session_id:
                with self._stats_lock:
                    self._ack_rejected_stale += 1
                logger.warning(f"[SpoolProcessor] ⚠ ACK rejected: wrong session_id. "
                             f"Expected {self._session_id[:8]}, got {ack.session_id[:8]}. "
                             f"Stale ACK count: {self._ack_rejected_stale}")
                return
            
            # Find and free the in-flight frame (out-of-order ACKs are OK)
            with self._in_flight_lock:
                if ack.seq in self._in_flight:
                    in_flight_frame = self._in_flight[ack.seq]
                    elapsed = time.time() - in_flight_frame.sent_time
                    
                    # Remove from tracking
                    del self._in_flight[ack.seq]
                    # Note: We don't remove from _in_flight_order deque as it's used for timeout scanning
                    # Timeout scanner will skip already-deleted entries
                    
                    # Update statistics
                    with self._stats_lock:
                        self._ack_accepted += 1
                        self._ack_rate_window.append(time.time())
                        self._frames_processed += 1
                    
                    self._last_ack_time = time.time()
                    
                    logger.debug(f"[SpoolProcessor] ✓ ACK received: seq={ack.seq}, "
                               f"frame_index={ack.frame_index}, elapsed={elapsed:.3f}s, "
                               f"in_flight={len(self._in_flight)}/{self.config.max_in_flight}")
                else:
                    # ACK for frame not in flight (could be duplicate or very late)
                    logger.debug(f"[SpoolProcessor] ACK for non-in-flight frame: seq={ack.seq}, "
                               f"frame_index={ack.frame_index}")
                    
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error in ACK callback: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _check_and_expire_timeouts(self):
        """
        Check for timed-out in-flight frames and free credit.
        
        This method scans the oldest in-flight frames and marks any that
        have exceeded ack_timeout as expired, freeing credit to prevent deadlock.
        """
        current_time = time.time()
        expired_seqs = []
        
        with self._in_flight_lock:
            # Scan from oldest to newest (in_flight_order is FIFO)
            while self._in_flight_order:
                seq = self._in_flight_order[0]
                
                # Check if frame still exists (might have been ACKed)
                if seq not in self._in_flight:
                    self._in_flight_order.popleft()
                    continue
                
                in_flight_frame = self._in_flight[seq]
                age = current_time - in_flight_frame.sent_time
                
                # If oldest frame hasn't timed out, none have (FIFO order)
                if age < self.config.ack_timeout:
                    break
                
                # Frame has timed out - mark as expired and free credit
                if not in_flight_frame.expired:
                    in_flight_frame.expired = True
                    expired_seqs.append(seq)
                    logger.warning(f"[SpoolProcessor] ⏱ Frame timeout: seq={seq}, "
                                 f"frame_index={in_flight_frame.frame_index}, "
                                 f"age={age:.1f}s, freeing credit")
                    
                    # Remove from tracking to free credit
                    del self._in_flight[seq]
                    self._in_flight_order.popleft()
                    
                    # Update statistics
                    with self._stats_lock:
                        self._ack_timeouts += 1
                else:
                    # Already expired, just remove
                    self._in_flight_order.popleft()
    
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
        Main processing loop with credit-based flow control.
        
        This loop implements bounded in-flight window:
        1. Wait for consumer startup (startup sync)
        2. Continuously publish frames while in_flight < max_in_flight
        3. Check for timeouts and free credit for expired frames
        4. Sleep briefly when backpressure or spool empty
        5. No per-frame blocking on ACK - ACK callback frees credit asynchronously
        
        Production reliability features:
        - Startup synchronization: Wait for consumer before first frame
        - Credit-based backpressure: Window fills naturally when consumer slows
        - Timeout handling: Expired frames free credit to prevent deadlock
        - High throughput: Target 20 FPS when consumer can keep up
        """
        logger.info("[SpoolProcessor] Processing loop started (credit-based flow control)")
        
        # Startup synchronization: Wait for consumer READY
        self._wait_for_consumer_ready()
        
        last_timeout_check = time.time()
        timeout_check_interval = 0.5  # Check for timeouts every 500ms
        
        while self._running:
            try:
                # Periodically check for and expire timed-out frames
                current_time = time.time()
                if current_time - last_timeout_check >= timeout_check_interval:
                    self._check_and_expire_timeouts()
                    last_timeout_check = current_time
                
                # Check if we have credit available
                with self._in_flight_lock:
                    in_flight_count = len(self._in_flight)
                    has_credit = in_flight_count < self.config.max_in_flight
                
                if not has_credit:
                    # Backpressure: in-flight window is full
                    with self._state_lock:
                        self._state = ProcessorState.BACKPRESSURE
                    logger.debug(f"[SpoolProcessor] Backpressure: in_flight={in_flight_count}/{self.config.max_in_flight}, waiting for ACKs...")
                    time.sleep(self.config.publish_idle_sleep_ms / 1000.0)
                    continue
                
                # Get next frame from spool
                frame = self._get_next_frame()
                
                if frame is None:
                    # Spool is empty, wait longer before retrying
                    with self._state_lock:
                        self._state = ProcessorState.SPOOL_EMPTY
                    logger.debug("[SpoolProcessor] Spool empty, waiting for new frames...")
                    time.sleep(self.config.empty_poll_interval)
                    continue
                
                # Update state and current frame tracking
                with self._state_lock:
                    self._state = ProcessorState.PUBLISHING
                self._current_frame = frame
                self._current_frame_index = frame.index
                
                # Publish frame (non-blocking, adds to in-flight tracking)
                success, seq = self._publish_frame(frame)
                
                if not success:
                    with self._stats_lock:
                        self._frames_skipped += 1
                    logger.warning(f"[SpoolProcessor] 🔴 Failed to publish frame {frame.index}")
                    time.sleep(0.1)  # Brief pause on error
                    continue
                
                # Log stats periodically
                self._maybe_log_stats()
                
                # No sleep here - continue immediately to publish next frame if credit available
                # Sleep only happens when:
                # 1. Backpressure (no credit) - line 854
                # 2. Spool empty - line 865
                # 3. Publish error - line 881
                
            except Exception as e:
                logger.error(f"[SpoolProcessor] Error in processing loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(1.0)
        
        logger.info("[SpoolProcessor] Processing loop stopped")
    
    def _maybe_log_stats(self):
        """Log statistics periodically with credit-based flow control metrics."""
        current_time = time.time()
        
        # Regular stats every 10 seconds
        if current_time - self._last_stats_time >= self.config.stats_interval:
            with self._stats_lock:
                # Calculate time since last successful ACK (watchdog info)
                ack_staleness = current_time - self._last_ack_time if self._last_ack_time > 0 else 0.0
                
                # Calculate publish rate (frames/sec) from recent window
                publish_rate = 0.0
                if len(self._publish_rate_window) >= 2:
                    time_span = self._publish_rate_window[-1] - self._publish_rate_window[0]
                    if time_span > 0:
                        publish_rate = len(self._publish_rate_window) / time_span
                
                # Calculate ACK rate (frames/sec) from recent window
                ack_rate = 0.0
                if len(self._ack_rate_window) >= 2:
                    time_span = self._ack_rate_window[-1] - self._ack_rate_window[0]
                    if time_span > 0:
                        ack_rate = len(self._ack_rate_window) / time_span
                
                # Get in-flight count
                with self._in_flight_lock:
                    in_flight_count = len(self._in_flight)
                
                logger.info(f"[SpoolProcessor] Stats: "
                           f"session={self._session_id[:8]}, "
                           f"published={self._frames_published}, "
                           f"acked={self._ack_accepted}, "
                           f"processed={self._frames_processed}, "
                           f"skipped={self._frames_skipped}, "
                           f"timeouts={self._ack_timeouts}, "
                           f"ack_rejected={self._ack_rejected_stale}, "
                           f"in_flight={in_flight_count}/{self.config.max_in_flight}, "
                           f"pub_rate={publish_rate:.1f}fps, "
                           f"ack_rate={ack_rate:.1f}fps, "
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
                
                # Warn if in-flight window is consistently full (backpressure)
                if in_flight_count >= self.config.max_in_flight:
                    logger.warning(f"[SpoolProcessor] ⚠ BACKPRESSURE: In-flight window full ({in_flight_count}/{self.config.max_in_flight}) - "
                                  "consumer may be slower than publisher")
            
            self._last_stats_time = current_time
        
        # Detailed stats every 2 minutes (120 seconds)
        if current_time - self._last_detailed_stats_time >= 120.0:
            with self._stats_lock:
                ack_accepted = self._ack_accepted
                ack_rejected = self._ack_rejected_stale
                total_acks = ack_accepted + ack_rejected
                ack_accept_rate = (ack_accepted / total_acks * 100) if total_acks > 0 else 0.0
                ack_reject_rate = (ack_rejected / total_acks * 100) if total_acks > 0 else 0.0
                
                # Calculate publish rate from window
                publish_rate = 0.0
                if len(self._publish_rate_window) >= 2:
                    time_span = self._publish_rate_window[-1] - self._publish_rate_window[0]
                    if time_span > 0:
                        publish_rate = len(self._publish_rate_window) / time_span
                
                # Calculate ACK rate from window
                ack_rate = 0.0
                if len(self._ack_rate_window) >= 2:
                    time_span = self._ack_rate_window[-1] - self._ack_rate_window[0]
                    if time_span > 0:
                        ack_rate = len(self._ack_rate_window) / time_span
                
                # Get in-flight count
                with self._in_flight_lock:
                    in_flight_count = len(self._in_flight)
                
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
                logger.info(f"  Flow Control:")
                logger.info(f"    - In-flight: {in_flight_count}/{self.config.max_in_flight}")
                logger.info(f"    - Publish rate: {publish_rate:.1f} fps")
                logger.info(f"    - ACK rate: {ack_rate:.1f} fps")
                logger.info(f"    - Last ACK age: {current_time - self._last_ack_time:.1f}s" if self._last_ack_time > 0 else "    - Last ACK age: N/A")
                logger.info(f"  ACK Statistics:")
                logger.info(f"    - Accepted: {ack_accepted} ({ack_accept_rate:.1f}%)")
                logger.info(f"    - Rejected (stale): {ack_rejected} ({ack_reject_rate:.1f}%)")
                logger.info(f"    - Total: {total_acks}")
                logger.info(f"  Frame Processing:")
                logger.info(f"    - Published: {self._frames_published}")
                logger.info(f"    - Processed (ACKed): {self._frames_processed}")
                logger.info(f"    - Skipped: {self._frames_skipped}")
                logger.info(f"    - Timeouts: {self._ack_timeouts}")
                logger.info(f"  Spool Status:")
                logger.info(f"    - Total segments: {len(segments)}")
                logger.info(f"    - Current segment: {self._current_segment}")
                logger.info(f"    - Oldest segment: {oldest_segment}")
                logger.info(f"    - Newest segment: {newest_segment}")
                logger.info(f"    - Spool lag: {spool_lag} segments")
                logger.info(f"    - SPS/PPS prepends: {self._sps_pps_prepends}")
                
                # Warn if spool is falling behind (lag > 10 segments = ~50 seconds at 5s/segment)
                if spool_lag > 10:
                    logger.warning(f"  ⚠ SPOOL LAG WARNING: Processor is {spool_lag} segments behind!")
                    logger.warning(f"     Recording is ahead by ~{spool_lag * 5}s. Processing too slow!")
                    logger.warning(f"     Consider: increasing max_in_flight, checking consumer performance.")
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
