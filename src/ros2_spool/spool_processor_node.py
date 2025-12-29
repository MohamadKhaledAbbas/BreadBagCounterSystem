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
    spool_ack_timeout: Timeout waiting for ACK in seconds (default: 30.0)
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
from src.logging.Database import DatabaseManager
from src import constants

# ROS2 imports (only on RDK platform)
if IS_RDK:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from img_msgs.msg import H26XFrame
    from std_msgs.msg import UInt32
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
    SPOOL_EMPTY = "spool_empty"
    STOPPED = "stopped"


# Default configuration values
DEFAULT_SPOOL_DIR = "/home/sunrise/BreadCounting/data/spool"
DEFAULT_ACK_TIMEOUT = 30.0
DEFAULT_RETRY_COUNT = 2
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_STATS_INTERVAL = 10.0


@dataclass
class ProcessorConfig:
    """Configuration for the spool processor."""
    spool_dir: str = DEFAULT_SPOOL_DIR
    ack_timeout: float = DEFAULT_ACK_TIMEOUT
    retry_count: int = DEFAULT_RETRY_COUNT
    poll_interval: float = DEFAULT_POLL_INTERVAL
    stats_interval: float = DEFAULT_STATS_INTERVAL


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
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__('spool_processor')
        
        # Load configuration from database if not provided
        self.config = config or load_config_from_db()
        
        logger.info(f"[SpoolProcessor] Initializing with config: "
                   f"spool_dir={self.config.spool_dir}, "
                   f"ack_timeout={self.config.ack_timeout}s, "
                   f"retry_count={self.config.retry_count}")
        
        # Initialize components
        self._reader = SegmentReader(self.config.spool_dir)
        self._frame_generator: Optional[Generator] = None
        self._current_frame: Optional[FrameRecord] = None
        self._current_frame_index: int = 0
        
        # State management
        self._state = ProcessorState.IDLE
        self._state_lock = threading.Lock()
        self._ack_received = threading.Event()
        self._ack_frame_index: int = -1
        
        # Processing thread
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._frames_processed = 0
        self._frames_retried = 0
        self._frames_skipped = 0
        self._ack_timeouts = 0
        self._last_stats_time = time.time()
        self._stats_lock = threading.Lock()
        
        # ROS2 publishers and subscribers
        if IS_RDK:
            # QoS for reliable messaging
            reliable_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            # Best effort for encoded frames (matches decoder expectations)
            best_effort_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )

            # Publisher for encoded frames (to decoder input)
            self._frame_pub = self.create_publisher(
                H26XFrame,
                '/spool_image_ch_0',
                reliable_qos
            )
            
            # Publisher for current frame index (side channel for ACK correlation)
            self._index_pub = self.create_publisher(
                UInt32,
                '/spool/current_frame_index',
                reliable_qos
            )
            
            # Subscriber for processing ACK
            self._ack_sub = self.create_subscription(
                UInt32,
                '/processing_ack',
                self._ack_callback,
                reliable_qos
            )
            
            # Optional: Pull request topic (for external control)
            self._request_sub = self.create_subscription(
                UInt32,
                '/spool/request_next',
                self._request_callback,
                reliable_qos
            )
            
            logger.info("[SpoolProcessor] ROS2 topics configured: "
                       "/spool_image_ch_0 (pub), /spool/current_frame_index (pub), "
                       "/processing_ack (sub), /spool/request_next (sub)")
    
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
        
        # Wait for processor thread
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
            if self._processor_thread.is_alive():
                logger.warning("[SpoolProcessor] Processor thread did not stop in time")
        
        # Log final stats
        with self._stats_lock:
            logger.info(f"[SpoolProcessor] Final stats: "
                       f"processed={self._frames_processed}, "
                       f"retried={self._frames_retried}, "
                       f"skipped={self._frames_skipped}, "
                       f"timeouts={self._ack_timeouts}")
        
        logger.info("[SpoolProcessor] Stopped")
    
    def _init_frame_generator(self):
        """Initialize the frame generator from oldest segment."""
        oldest = self._reader.get_oldest_segment()
        if oldest is not None:
            logger.info(f"[SpoolProcessor] Starting from segment {oldest}")
            self._frame_generator = self._reader.read_frames(start_segment=oldest)
        else:
            logger.warning("[SpoolProcessor] No segments available")
            self._frame_generator = iter([])
    
    def _get_next_frame(self) -> Optional[FrameRecord]:
        """Get the next frame from the spool."""
        if self._frame_generator is None:
            self._init_frame_generator()
        
        try:
            return next(self._frame_generator)
        except StopIteration:
            # Try to reinitialize from new segments
            self._init_frame_generator()
            try:
                return next(self._frame_generator)
            except StopIteration:
                return None
    
    def _publish_frame(self, record: FrameRecord) -> bool:
        """Publish a frame to the decoder input topic."""
        if not IS_RDK:
            return True
        
        try:
            # First, publish the frame index for correlation
            index_msg = UInt32()
            index_msg.data = record.index
            self._index_pub.publish(index_msg)
            
            # Then publish the encoded frame
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
            
            frame_msg.data = list(record.data)
            
            self._frame_pub.publish(frame_msg)
            
            logger.debug(f"[SpoolProcessor] Published frame {record.index}")
            return True
            
        except Exception as e:
            logger.error(f"[SpoolProcessor] Error publishing frame: {e}")
            return False
    
    def _wait_for_ack(self, frame_index: int, timeout: float) -> bool:
        """
        Wait for ACK for a specific frame index.
        
        Args:
            frame_index: Expected frame index in ACK
            timeout: Maximum time to wait
            
        Returns:
            True if ACK received, False on timeout
        """
        self._ack_received.clear()
        start_time = time.time()
        
        while self._running:
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                return False
            
            if self._ack_received.wait(timeout=min(remaining, 1.0)):
                if self._ack_frame_index == frame_index:
                    return True
                # ACK was for different frame, keep waiting
                self._ack_received.clear()
        
        return False
    
    def _ack_callback(self, msg):
        """Callback for processing ACK messages."""
        self._ack_frame_index = msg.data
        self._ack_received.set()
        logger.debug(f"[SpoolProcessor] Received ACK for frame {msg.data}")
    
    def _request_callback(self, msg):
        """Callback for external pull requests (optional feature)."""
        # This allows external control of frame advancement
        logger.debug(f"[SpoolProcessor] Received request {msg.data}")
    
    def _processor_loop(self):
        """
        Main processing loop with strict backpressure.
        
        This loop ensures exactly one frame is in flight:
        1. Get next frame from spool
        2. Publish frame
        3. Wait for ACK (with retry)
        4. Repeat
        """
        logger.info("[SpoolProcessor] Processing loop started")
        
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
        
        while retries <= self.config.retry_count and self._running:
            with self._state_lock:
                self._state = ProcessorState.IDLE
            
            # Publish frame
            if not self._publish_frame(frame):
                logger.warning(f"[SpoolProcessor] Failed to publish frame {frame.index}")
                retries += 1
                continue
            
            with self._state_lock:
                self._state = ProcessorState.WAITING_FOR_ACK
            
            # Wait for ACK
            if self._wait_for_ack(frame.index, self.config.ack_timeout):
                return True
            
            # Timeout - retry
            with self._stats_lock:
                self._ack_timeouts += 1
            
            if retries < self.config.retry_count:
                with self._stats_lock:
                    self._frames_retried += 1
                logger.warning(f"[SpoolProcessor] ACK timeout for frame {frame.index}, "
                              f"retry {retries + 1}/{self.config.retry_count}")
            
            retries += 1
        
        return False
    
    def _maybe_log_stats(self):
        """Log statistics periodically."""
        current_time = time.time()
        if current_time - self._last_stats_time >= self.config.stats_interval:
            with self._stats_lock:
                logger.info(f"[SpoolProcessor] Stats: "
                           f"processed={self._frames_processed}, "
                           f"retried={self._frames_retried}, "
                           f"skipped={self._frames_skipped}, "
                           f"timeouts={self._ack_timeouts}, "
                           f"state={self._state.value}")
                
                # Log spool status
                segments = self._reader.list_segments()
                logger.info(f"[SpoolProcessor] Spool: "
                           f"segments={len(segments)}, "
                           f"current_frame={self._current_frame_index}")
            
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
