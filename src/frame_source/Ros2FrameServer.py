import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from hbm_img_msgs.msg import HbmMsg1080P
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import UInt32

from src.config.settings import AppConfig
from src.frame_source.FrameSource import FrameSource
from src.logging.Database import DatabaseManager
from src import constants

from src.utils.AppLogging import logger


class FrameServer(Node, FrameSource):
    """
    ROS 2 Subscriber that listens for incoming frames and buffers the latest
    frame for consumption by the main logic thread. This node is designed
    to be added to an external SingleThreadedExecutor.
    
    Accuracy Mode Support:
    - Subscribes to /spool/current_frame_index for frame index tracking
    - Attaches frame index to each yielded frame for ACK correlation
    - Enables BagCounterApp to acknowledge specific processed frames
    """

    def __init__(self, topic='/nv12_images', target_fps=30.0):
        # IMPORTANT: rclpy.init() must be called externally before this class is instantiated.
        super().__init__('frame_server')
        
        # Support ROS_TARGET_FPS environment variable override
        env_fps = os.getenv('ROS_TARGET_FPS')
        if env_fps:
            try:
                target_fps = float(env_fps)
                logger.info(f"[Ros2FrameServer] Using ROS_TARGET_FPS from environment: {target_fps}")
            except ValueError:
                logger.warning(f"[Ros2FrameServer] Invalid ROS_TARGET_FPS value '{env_fps}', using default {target_fps}")
        
        self.subscription = self.create_subscription(
            HbmMsg1080P,
            topic,
            self.listener_callback,
            qos_profile_sensor_data)

        # Buffer more frames to reduce frame drops
        # V3 Performance: Increased from 10 to 30 for better burst handling (1.2s buffer @ 25fps)
        self.frame_queue = queue.Queue(maxsize=30)
        self.last_frame_time = time.time()
        
        # Store target_fps for logging only
        self.target_fps = target_fps
        
        # V3 Performance: Proactive drop threshold (80% of queue size)
        self.proactive_drop_threshold = int(self.frame_queue.maxsize * 0.8)  # 24 frames for size=30
        
        # Stats for debugging
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0  # Track dropped frames
        self.last_stats_log_time = time.time()
        self.stats_log_interval = 5.0  # Log stats every 5 seconds
        
        # Accuracy Mode: Frame index tracking for ACK correlation
        # Subscribe to /spool/current_frame_index published by SpoolProcessorNode
        self._current_frame_index = 0
        self._last_yielded_frame_index = 0  # Track the index of the most recently yielded frame
        self._frame_index_lock = threading.Lock()
        
        # Check if accuracy mode is enabled via database config, fall back to environment
        try:
            db = DatabaseManager(db_path=AppConfig.db_path)
            accuracy_mode_config = db.get_config_value(constants.accuracy_mode_enabled)
            db.close()
            if accuracy_mode_config is not None:
                self._accuracy_mode = accuracy_mode_config == '1'
            else:
                self._accuracy_mode = os.getenv('ACCURACY_MODE', '').lower() in ('1', 'true', 'yes')
        except Exception:
            self._accuracy_mode = os.getenv('ACCURACY_MODE', '').lower() in ('1', 'true', 'yes')
        
        if self._accuracy_mode:
            reliable_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10  # Increased depth for better buffering
            )
            self._index_sub = self.create_subscription(
                UInt32,
                '/spool/current_frame_index',
                self._frame_index_callback,
                reliable_qos
            )
            logger.info("[Ros2FrameServer] Accuracy Mode enabled - subscribing to /spool/current_frame_index")
        
        logger.info(f"[Ros2FrameServer] Initialized with queue_size=30, target_fps={target_fps} (for stats logging only)")

        # --- REMOVED ---
        # Removed the internal _ros_spin_thread logic.
        # The execution (spinning) is now handled by the external ExecutorThread.
        # ---------------
    
    def _frame_index_callback(self, msg):
        """Callback for frame index updates from SpoolProcessorNode."""
        with self._frame_index_lock:
            self._current_frame_index = int(msg.data)
        logger.debug(f"[Ros2FrameServer] Received frame index: {msg.data}")
    
    def get_current_frame_index(self) -> int:
        """Get the current frame index for ACK correlation."""
        with self._frame_index_lock:
            return int(self._current_frame_index)

    def listener_callback(self, msg):
        now = time.time()
        self.frames_received += 1
        
        # No time-based frame skipping - rely only on leaky queue
        self.frames_processed += 1
        
        # V3 Performance: Log stats periodically (time-based for consistent intervals)
        if now - self.last_stats_log_time >= self.stats_log_interval:
            queue_utilization = (self.frame_queue.qsize() / self.frame_queue.maxsize) * 100
            drop_rate = (self.frames_dropped / self.frames_received * 100) if self.frames_received > 0 else 0.0
            logger.info(
                f"[Ros2FrameServer] Stats: received={self.frames_received}, "
                f"processed={self.frames_processed}, dropped={self.frames_dropped}, "
                f"drop_rate={drop_rate:.2f}%, queue_util={queue_utilization:.1f}%"
            )
            self.last_stats_log_time = now
        
        img = np.frombuffer(msg.data, dtype=np.uint8)[:msg.data_size]
        try:
            # NV12 conversion logic
            nv12_img = img.reshape((msg.height * 3 // 2, msg.width))
            bgr = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)
        except Exception as e:
            self.get_logger().error(f"Frame conversion error: {e}")
            return

        latency_ms = (now - self.last_frame_time) * 1000
        self.last_frame_time = now

        # V3 Performance: Leaky queue with drop tracking
        # Proactively drop when queue is heavily utilized (> 80% full)
        queue_size = self.frame_queue.qsize()
        if queue_size >= self.proactive_drop_threshold:
            # Drop oldest frame to make room
            try:
                self.frame_queue.get_nowait()
                self.frames_dropped += 1
                logger.debug(f"[Ros2FrameServer] Proactive drop at queue_size={queue_size}/{self.frame_queue.maxsize} (80% threshold)")
            except queue.Empty:
                pass
        elif self.frame_queue.full():
            # Full queue - drop oldest
            try:
                self.frame_queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
        
        # Enqueue new frame with frame index for accuracy mode
        # The frame index is captured at enqueue time to ensure correlation
        frame_index = self.get_current_frame_index() if self._accuracy_mode else 0
        self.frame_queue.put((bgr, latency_ms, frame_index))

    def frames(self):
        """
        Yield frames from the queue.
        
        Yields:
            Tuple of (frame, latency_ms) for backward compatibility.
            In accuracy mode, frame_index is also available via get_current_frame_index().
        """
        # We check rclpy.ok() to ensure we stop if the ROS context shuts down
        while rclpy.ok():
            try:
                item = self.frame_queue.get(timeout=1)
                # Handle both old format (frame, latency) and new format (frame, latency, index)
                if len(item) == 3:
                    frame, latency_ms, frame_index = item
                    # Store the frame index that was associated with THIS frame when it was enqueued
                    # This is critical for ACK correlation in accuracy mode
                    with self._frame_index_lock:
                        self._current_frame_index = frame_index
                        self._last_yielded_frame_index = frame_index
                    if self._accuracy_mode:
                        logger.debug(f"[Ros2FrameServer] Yielding frame with index {frame_index}")
                    yield frame, latency_ms
                else:
                    frame, latency_ms = item
                    yield frame, latency_ms
            except queue.Empty:
                continue
    
    def get_last_yielded_frame_index(self) -> int:
        """
        Get the frame index of the most recently yielded frame.
        
        This is the correct method to use for ACK correlation, as it returns
        the index that was stored with the frame when it was enqueued,
        not the most recent index received via subscription.
        
        Returns:
            Frame index of the last yielded frame
        """
        with self._frame_index_lock:
            return getattr(self, '_last_yielded_frame_index', self._current_frame_index)
    
    def frames_with_index(self):
        """
        Yield frames with frame index for accuracy mode.
        
        Yields:
            Tuple of (frame, latency_ms, frame_index)
        """
        while rclpy.ok():
            try:
                item = self.frame_queue.get(timeout=1)
                if len(item) == 3:
                    yield item
                else:
                    frame, latency_ms = item
                    yield frame, latency_ms, 0
            except queue.Empty:
                continue

    def cleanup(self):
        """Destroys the node, relying on the main app to shutdown the ROS context."""
        logger.info("[Ros2FrameServer] cleanup called. Destroying node.")

        # Destroy the node itself
        try:
            self.destroy_node()
        except Exception as e:
            logger.debug(f"[Ros2FrameServer] destroy_node() raised (ignored): {e}")

        logger.info("[Ros2FrameServer] cleanup finished")