import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


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
    
    V11: ACK-free mode only - simplified architecture for production reliability.
    Frames are buffered in input_queue and smart degraded mode handles overload.
    """

    def __init__(self, topic='/nv12_images'):
        # IMPORTANT: rclpy.init() must be called externally before this class is instantiated.
        super().__init__('frame_server')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.listener_callback,
            qos)

        # Buffer more frames to reduce frame drops
        # V3 Performance: Increased from 10 to 30 for better burst handling (1.2s buffer @ 25fps)
        self.frame_queue = queue.Queue(maxsize=30)
        self.last_frame_time = time.time()
        
        # V3 Performance: Proactive drop threshold (80% of queue size)
        self.proactive_drop_threshold = int(self.frame_queue.maxsize * 0.8)  # 24 frames for size=30
        
        # Stats for debugging
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0  # Track dropped frames
        self.last_stats_log_time = time.time()
        self.stats_log_interval = 5.0  # Log stats every 5 seconds

        # --- REMOVED ---
        # Removed the internal _ros_spin_thread logic.
        # The execution (spinning) is now handled by the external ExecutorThread.
        # ---------------
    
    def _get_timing_stats_string(self) -> str:
        """
        V6: Get formatted timing stats string and reset counters.
        V7: Removed BGR conversion timing (now done lazily in logic thread).
        
        Returns:
            Formatted string with timing stats, or empty string if no data
        """
        if not hasattr(self, '_timing_stats') or self._timing_stats['count'] == 0:
            return ""
        
        count = self._timing_stats['count']
        stats_str = (
            f", avg_callback={self._timing_stats['callback'] / count:.2f}ms"
            f" (reshape={self._timing_stats['reshape'] / count:.2f}ms"
            f", nv12_copy={self._timing_stats['nv12_copy'] / count:.2f}ms)"
        )
        # Reset timing stats
        self._timing_stats = {'callback': 0, 'reshape': 0, 'nv12_copy': 0, 'count': 0}
        return stats_str


    def listener_callback(self, msg):
        now = time.time()
        self.frames_received += 1
        
        # V7: Detailed timing metrics for performance analysis
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
        
        # V7: Time each operation
        t_reshape_start = time.perf_counter()

        expected = msg.height * msg.width * 3 // 2
        img = np.frombuffer(msg.data, dtype=np.uint8)
        if img.size < expected:
            self.get_logger().error(f"Frame size mismatch: got {img.size}, expected {expected}")
            return
        try:
            # NV12 conversion logic - reshape to NV12 format
            nv12_img = img.reshape((msg.height * 3 // 2, msg.width))
            t_reshape_end = time.perf_counter()
            
            # V7 Optimization: Store raw NV12 data ONLY - skip BGR conversion
            # BGR conversion is now done lazily in logic_thread when needed for:
            # - Visualization (if publishing is enabled)
            # - Classification (classifier needs BGR)
            # Detection uses NV12 directly via _preprocess_nv12
            t_nv12_copy_start = time.perf_counter()
            nv12_data = nv12_img.copy()  # Copy to ensure data persists after message is released
            t_nv12_copy_end = time.perf_counter()
            
        except Exception as e:
            self.get_logger().error(f"Frame conversion error: {e}")
            return
        
        # V7: Accumulate timing stats (removed BGR conversion timing)
        if not hasattr(self, '_timing_stats'):
            self._timing_stats = {'callback': 0, 'reshape': 0, 'nv12_copy': 0, 'count': 0}
        
        t_callback_end = time.perf_counter()
        self._timing_stats['callback'] += (t_callback_end - t_callback_start) * 1000
        self._timing_stats['reshape'] += (t_reshape_end - t_reshape_start) * 1000
        self._timing_stats['nv12_copy'] += (t_nv12_copy_end - t_nv12_copy_start) * 1000
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
        spool_frame_index = 0
        
        # V7 Optimization: Include ONLY raw NV12 data in frame tuple (no BGR)
        # Frame format: (nv12_data, latency_ms, spool_frame_index, (height, width))
        # BGR conversion is done lazily in logic_thread when needed
        # Attach spool_frame_index to frame data - it travels through pipeline with this frame
        frame_size = (msg.height, msg.width)
        self.frame_queue.put((nv12_data, latency_ms, spool_frame_index, frame_size))

    def frames(self):
        """
        Yield frames from the queue.
        
        V7 Optimization: Now yields only NV12 data (no BGR).
        BGR conversion is done lazily in the consumer when needed.
        
        V8 Performance Fix: Increased queue timeout from 10ms to 100ms to reduce
        polling overhead and improve frame acquisition rate. Previous 10ms timeout
        caused excessive polling when queue was empty, leading to ~70ms effective
        frame intervals instead of ~33ms (30 FPS). With 100ms timeout:
        - Reduces CPU overhead from polling
        - Allows blocking on queue for frame arrival
        - Still provides responsive shutdown (checked every 100ms)
        
        Yields:
            Tuple of (nv12_data, latency_ms, spool_frame_index, frame_size)
            
        Where:
            - nv12_data: Raw NV12 numpy array for detection and lazy BGR conversion
            - latency_ms: Frame latency in milliseconds
            - spool_frame_index: Frame index for ACK correlation
            - frame_size: Tuple (height, width) of the original frame
        """
        # We check rclpy.ok() to ensure we stop if the ROS context shuts down
        while rclpy.ok():
            try:
                # V8 Performance: Increased timeout from 10ms to 100ms
                # Previous 10ms timeout caused excessive polling overhead when queue
                # was empty, contributing to low acquisition FPS (~13-15 instead of ~30).
                # With 100ms timeout, we block efficiently waiting for frames while
                # still allowing responsive shutdown checks.
                item = self.frame_queue.get(timeout=0.1)
                
                # V7: New format with NV12 only (4 elements)
                if len(item) == 4:
                    nv12_data, latency_ms, spool_frame_index, frame_size = item
                    yield nv12_data, latency_ms, spool_frame_index, frame_size
                elif len(item) == 5:
                    # Backward compatibility: old format (bgr, latency_ms, spool_frame_index, nv12_data, frame_size)
                    # Skip BGR, yield NV12 data
                    _bgr, latency_ms, spool_frame_index, nv12_data, frame_size = item
                    yield nv12_data, latency_ms, spool_frame_index, frame_size
                elif len(item) == 2:
                    # Legacy format (frame, latency_ms) - convert to NV12-only format with placeholder
                    frame, latency_ms = item
                    yield frame, latency_ms, 0, frame.shape[:2] if hasattr(frame, 'shape') else (720, 1280)
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