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


# Default configuration values
DEFAULT_SPOOL_DIR = "/home/sunrise/BreadCounting/data/spool"
DEFAULT_ACK_TIMEOUT = 10.0  # Reduced from 30.0 - should be much faster with FIFO correlation
DEFAULT_RETRY_COUNT = 2
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_STATS_INTERVAL = 10.0
DEFAULT_STARTUP_GRACE_PERIOD = 10.0  # Seconds to wait for consumer to start
DEFAULT_SPS_PPS_PREPEND = True  # Prepend cached SPS/PPS to first frame of segment


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


def load_config_from_db(db_path: str = AppConfig.db_path) -> ProcessorConfig:
    """Load spool processor configuration from database config table."""
    try:
        db = DatabaseManager(db_path)
        
        spool_dir = db.get_config_value(constants.spool_dir)
        ack_timeout = db.get_config_value(constants.spool_ack_timeout)
        retry_count = db.get_config_value(constants.spool_retry_count)
        
        db.close()
        
        return ProcessorConfig(
            spool_dir=spool_dir if spool_dir else DEFAULT_SPOOL_DIR,
            ack_timeout=float(ack_timeout) if ack_timeout else DEFAULT_ACK_TIMEOUT,
            retry_count=int(retry_count) if retry_count else DEFAULT_RETRY_COUNT,
        )
    except Exception as e:
        logger.warning(f"[SpoolProcessor] Failed to load config from DB: {e}, using defaults")
        return ProcessorConfig()


class SpoolProcessorNode(Node):
    """
    ROS2 Node that processes spooled H.264 frames with strict backpressure.
    
    The processor ensures exactly one frame is in flight at a time:
    1. Read next frame from spool
    2. Publish frame index to /spool/current_frame_index
    3. Publish encoded frame to /spool_image_ch_0
    4. Wait for ACK on /processing_ack with matching index
    5. Only then proceed to next frame
    
    This design implements strict pull-based processing where BagCounterApp
    controls the pace.
    
    Production Reliability Features:
    - Startup synchronization: Waits for consumer before processing
    - Watchdog: Auto-recovery from stuck ACK states
    - SPS/PPS caching: Prepends to segment boundaries for decoder init
    - Graceful degradation: Advances after max retries to prevent deadlock
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
        self._segments_processed = 0
        self._sps_pps_prepends = 0
        self._last_stats_time = time.time()
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
            logger.info(f"[SpoolProcessor] Final stats: "
                       f"session={self._session_id[:8]}, "
                       f"seq={self._seq_counter}, "
                       f"processed={self._frames_processed}, "
                       f"retried={self._frames_retried}, "
                       f"skipped={self._frames_skipped}, "
                       f"timeouts={self._ack_timeouts}, "
                       f"ack_rejected={self._ack_rejected_stale}, "
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
        # This ensures we process frames in order
        # Note: We create a list copy and clear temp_frames to avoid holding references
        buffered_frames_copy = list(temp_frames)
        temp_frames.clear()  # Clear to allow garbage collection
        
        def buffered_generator():
            # First yield buffered frames
            for frame in buffered_frames_copy:
                yield frame
            # Clear the copy after yielding to free memory
            buffered_frames_copy.clear()
            # Then continue with remaining frames
            try:
                while True:
                    yield next(self._frame_generator)
            except StopIteration:
                pass
        
        self._frame_generator = buffered_generator()
        
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
            self._init_frame_generator()
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
    
    def _publish_frame(self, record: FrameRecord) -> tuple:
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
            
            # Structured logging
            logger.info(f"[SpoolProcessor] 📤 Frame published: index={record.index}, seq={seq}, "
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
                    logger.info(f"[SpoolProcessor] ✓ ACK matched: seq={ack.seq}, "
                              f"frame_index={ack.frame_index}, elapsed={elapsed:.3f}s")
                    self._last_ack_time = time.time()
                    return True
                elif ack.frame_index == expected_frame_index:
                    # Frame index matches but seq doesn't - still accept
                    logger.info(f"[SpoolProcessor] ✓ ACK matched (frame_index): seq={ack.seq} "
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
        Main processing loop with strict backpressure.
        
        This loop ensures exactly one frame is in flight:
        1. Wait for consumer startup (startup sync)
        2. Get next frame from spool
        3. Publish frame
        4. Wait for ACK (with retry)
        5. Repeat
        
        Production reliability features:
        - Startup synchronization: Wait for consumer before first frame
        - Watchdog: Detect and recover from stuck states
        - Graceful degradation: Advance after max retries
        """
        logger.info("[SpoolProcessor] Processing loop started")
        
        # Startup synchronization: Wait for consumer READY
        self._wait_for_consumer_ready()
        
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
                
                # Process frame with retry logic
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
                logger.error(f"[SpoolProcessor] Error in processing loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(1.0)
        
        logger.info("[SpoolProcessor] Processing loop stopped")
    
    def _process_frame_with_retry(self, frame: FrameRecord) -> bool:
        """
        Process a frame with retry logic.
        
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
        """Log statistics periodically."""
        current_time = time.time()
        if current_time - self._last_stats_time >= self.config.stats_interval:
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
