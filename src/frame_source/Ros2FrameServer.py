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
from std_msgs.msg import String

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
    """

    def __init__(self, topic='/nv12_images', target_fps=20.0):
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
        
        # Initialize ACK publisher for flow control
        self.ack_publisher = self.create_publisher(String, '/spool_ack', 10)

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
        
        logger.info(f"[Ros2FrameServer] Initialized with queue_size=30, target_fps={target_fps} (for stats logging only)")

        # --- REMOVED ---
        # Removed the internal _ros_spin_thread logic.
        # The execution (spinning) is now handled by the external ExecutorThread.
        # ---------------
    
    def _get_timing_stats_string(self) -> str:
        """
        V6: Get formatted timing stats string and reset counters.
        
        Returns:
            Formatted string with timing stats, or empty string if no data
        """
        if not hasattr(self, '_timing_stats') or self._timing_stats['count'] == 0:
            return ""
        
        count = self._timing_stats['count']
        stats_str = (
            f", avg_callback={self._timing_stats['callback'] / count:.2f}ms"
            f" (reshape={self._timing_stats['reshape'] / count:.2f}ms"
            f", nv12_copy={self._timing_stats['nv12_copy'] / count:.2f}ms"
            f", bgr_cvt={self._timing_stats['bgr_convert'] / count:.2f}ms)"
        )
        # Reset timing stats
        self._timing_stats = {'callback': 0, 'reshape': 0, 'nv12_copy': 0, 'bgr_convert': 0, 'count': 0}
        return stats_str


    def listener_callback(self, msg):
        now = time.time()
        self.frames_received += 1
        
        # V6: Detailed timing metrics for performance analysis
        t_callback_start = time.perf_counter()
        
        # No time-based frame skipping - rely only on leaky queue
        self.frames_processed += 1
        
        # V3 Performance: Log stats periodically (time-based for consistent intervals)
        if now - self.last_stats_log_time >= self.stats_log_interval:
            queue_utilization = (self.frame_queue.qsize() / self.frame_queue.maxsize) * 100
            drop_rate = (self.frames_dropped / self.frames_received * 100) if self.frames_received > 0 else 0.0
            
            # V6: Get timing stats (resets counters)
            timing_stats_str = self._get_timing_stats_string()
            
            stats_msg = (
                f"[Ros2FrameServer] Stats: received={self.frames_received}, "
                f"processed={self.frames_processed}, dropped={self.frames_dropped}, "
                f"drop_rate={drop_rate:.2f}%, queue_util={queue_utilization:.1f}%"
            )
            # V6: Add timing stats
            stats_msg += timing_stats_str
            logger.info(stats_msg)
            self.last_stats_log_time = now
        
        # V6: Time each operation
        t_reshape_start = time.perf_counter()
        img = np.frombuffer(msg.data, dtype=np.uint8)[:msg.data_size]
        try:
            # NV12 conversion logic
            nv12_img = img.reshape((msg.height * 3 // 2, msg.width))
            t_reshape_end = time.perf_counter()
            
            # V5 Optimization: Store raw NV12 data to avoid redundant conversions
            # The BPU expects NV12 format, so we can skip BGR→NV12 conversion in detector
            # by passing raw NV12 directly
            t_nv12_copy_start = time.perf_counter()
            nv12_data = nv12_img.copy()  # Copy to ensure data persists after message is released
            t_nv12_copy_end = time.perf_counter()
            
            # V7 Performance: Defer BGR conversion to logic thread to unblock subscriber
            # This prevents blocking the ROS 2 callback, avoiding frame drops on "Best Effort" QoS
            t_bgr_start = time.perf_counter()
            bgr = None  # Deferred to main application thread
            t_bgr_end = time.perf_counter()
        except Exception as e:
            self.get_logger().error(f"Frame conversion error: {e}")
            return
        
        # V6: Accumulate timing stats
        if not hasattr(self, '_timing_stats'):
            self._timing_stats = {'callback': 0, 'reshape': 0, 'nv12_copy': 0, 'bgr_convert': 0, 'count': 0}
        
        t_callback_end = time.perf_counter()
        self._timing_stats['callback'] += (t_callback_end - t_callback_start) * 1000
        self._timing_stats['reshape'] += (t_reshape_end - t_reshape_start) * 1000
        self._timing_stats['nv12_copy'] += (t_nv12_copy_end - t_nv12_copy_start) * 1000
        self._timing_stats['bgr_convert'] += (t_bgr_end - t_bgr_start) * 1000
        self._timing_stats['count'] += 1

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
        
        # Enqueue new frame
        # V7: Extract frame index from message, fallback to frames_received
        spool_frame_index = getattr(msg, 'index', self.frames_received)
        
        # V5 Optimization: Include raw NV12 data in frame tuple for direct BPU input
        # Frame format: (bgr, latency_ms, spool_frame_index, nv12_data, (height, width))
        # Attach spool_frame_index to frame data - it travels through pipeline with this frame
        frame_size = (msg.height, msg.width)
        self.frame_queue.put((bgr, latency_ms, spool_frame_index, nv12_data, frame_size))
        
        # V7: Publish ACK to enable flow control from spool processor
        try:
            ack_msg = String()
            ack_msg.data = str(spool_frame_index)
            self.ack_publisher.publish(ack_msg)
        except Exception as e:
            logger.warning(f"[Ros2FrameServer] Failed to publish ACK: {e}")

    def frames(self):
        """
        Yield frames from the queue.
        
        V5 Optimization: Now yields NV12 data alongside BGR frame.
        
        Yields:
            Tuple of (frame, latency_ms, spool_frame_index, nv12_data, frame_size) - full format
            Tuple of (frame, latency_ms) - normal mode (backward compatible)
            
        Where:
            - frame: BGR numpy array for visualization/classification
            - latency_ms: Frame latency in milliseconds
            - spool_frame_index: Frame index for ACK correlation
            - nv12_data: Raw NV12 numpy array for direct BPU inference (avoids BGR→NV12 conversion)
            - frame_size: Tuple (height, width) of the original frame
        """
        # We check rclpy.ok() to ensure we stop if the ROS context shuts down
        while rclpy.ok():
            try:
                item = self.frame_queue.get(timeout=1)
                
                # V5: Handle new format with NV12 data (5 elements)
                if len(item) == 5:
                    frame, latency_ms, spool_frame_index, nv12_data, frame_size = item
                    # V5: Yield full frame data including NV12
                    yield frame, latency_ms, spool_frame_index, nv12_data, frame_size
                else:
                    frame, latency_ms = item
                    yield frame, latency_ms
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