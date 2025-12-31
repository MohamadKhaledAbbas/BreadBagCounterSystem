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

# Import frame metadata parsing
try:
    from src.ros2_spool.messages import frame_metadata_from_ros_string
except ImportError:
    # Fallback if messages module not available (shouldn't happen in production)
    frame_metadata_from_ros_string = None


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
        self.frames_index_fallback = 0  # Track fallback to current index (should be rare)
        self.frames_index_lost = 0  # Track frame indices that couldn't be enqueued (severe stall)
        self.last_stats_log_time = time.time()
        self.stats_log_interval = 5.0  # Log stats every 5 seconds
        
        # Accuracy Mode: Frame index tracking for ACK correlation
        # Subscribe to /spool/current_frame_index published by SpoolProcessorNode
        # 
        # SINGLE SOURCE OF TRUTH: spool_frame_index
        # When a frame enters the system, SpoolProcessor assigns ONE canonical ID.
        # This ID travels WITH the frame through the entire pipeline:
        # 1. Published to /spool/current_frame_index (for tracking)
        # 2. Stored in FIFO queue (pending_frame_indices)
        # 3. Attached to decoded NV12 frame when enqueued
        # 4. Yielded WITH frame data to BagCounterApp
        # 5. Echoed back in ACK message
        # No derived or local indices - ACK only what you were given.
        self._current_frame_index = 0
        self._last_yielded_frame_index = 0  # Track the index of the most recently yielded frame
        self._frame_index_lock = threading.Lock()
        
        # FIFO queue for pending frame indices to correlate with decoded frames
        # This solves the race condition where frame indices arrive before decoded frames
        # Queue size of 50 provides buffer for:
        # - 30 FPS: 1.6 seconds of decoder lag
        # - Typical decoder latency: 10-100ms
        # - Burst handling during temporary slowdowns
        self._pending_frame_indices = queue.Queue(maxsize=50)
        
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
                String,
                '/spool/current_frame_metadata',
                self._frame_metadata_callback,
                reliable_qos
            )
            logger.info("[Ros2FrameServer] Accuracy Mode ENABLED - subscribing to /spool/current_frame_metadata")
            logger.warning("[Ros2FrameServer] NOTE: Accuracy mode adds complexity. Consider disabling if not using legacy ACK mode.")
        else:
            self._index_sub = None
            logger.info("[Ros2FrameServer] Accuracy Mode DISABLED - Simple architecture, no frame index correlation")
        
        logger.info(f"[Ros2FrameServer] Initialized with queue_size=30, target_fps={target_fps} (for stats logging only)")

        # --- REMOVED ---
        # Removed the internal _ros_spin_thread logic.
        # The execution (spinning) is now handled by the external ExecutorThread.
        # ---------------
    
    def _frame_metadata_callback(self, msg):
        """
        Callback for frame metadata updates from SpoolProcessorNode.
        
        V7 Update: Now subscribes to /spool/current_frame_metadata (String with JSON)
        instead of legacy /spool/current_frame_index (UInt32).
        
        SINGLE SOURCE OF TRUTH: spool_frame_index from FrameMetadata
        This receives the canonical frame ID assigned by SpoolProcessor.
        Frame indices arrive BEFORE corresponding NV12 frames (decoder latency 10-100ms).
        We enqueue them in FIFO order to correlate with decoded frames in listener_callback.
        
        Args:
            msg: String message containing JSON-encoded FrameMetadata
        """
        if frame_metadata_from_ros_string is None:
            logger.error("[Ros2FrameServer] frame_metadata_from_ros_string not available - cannot parse metadata")
            return
        
        try:
            # Parse the FrameMetadata from JSON string
            metadata = frame_metadata_from_ros_string(msg.data)
            spool_frame_idx = metadata.frame_index
            
            # Update current index for backwards compatibility
            with self._frame_index_lock:
                self._current_frame_index = spool_frame_idx
            
            # Enqueue to pending queue for proper correlation
            try:
                self._pending_frame_indices.put_nowait(spool_frame_idx)
                logger.debug(f"[Ros2FrameServer] Received spool_frame_index: {spool_frame_idx} (seq={metadata.seq}, session={metadata.session_id[:8]}), queue_size={self._pending_frame_indices.qsize()}")
            except queue.Full:
                # Queue full - this indicates decoder is falling behind significantly
                # We must drop the oldest pending index to prevent blocking the publisher
                try:
                    dropped_idx = self._pending_frame_indices.get_nowait()
                    logger.warning(f"[Ros2FrameServer] Pending index queue full, dropped spool_frame_index {dropped_idx}")
                    self.frames_index_lost += 1  # Track lost indices
                except queue.Empty:
                    # Race condition: queue became empty between full check and get
                    # This is rare but harmless - just log it
                    logger.debug(f"[Ros2FrameServer] Pending queue became empty during drop attempt")
                
                # Now enqueue the new index (with another try/except for safety)
                try:
                    self._pending_frame_indices.put_nowait(spool_frame_idx)
                except queue.Full:
                    # Extremely rare: queue filled again between operations
                    # This indicates severe decoder stall - log error and track metric
                    self.frames_index_lost += 1
                    logger.error(
                        f"[Ros2FrameServer] Failed to enqueue frame index {spool_frame_idx} after drop - "
                        f"decoder severely stalled (lost_indices={self.frames_index_lost})"
                    )
        except Exception as e:
            logger.error(f"[Ros2FrameServer] Error parsing frame metadata: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def get_current_frame_index(self) -> int:
        """Get the current frame index for ACK correlation."""
        with self._frame_index_lock:
            return int(self._current_frame_index)
    
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
            
            # Include pending index queue stats for accuracy mode
            if self._accuracy_mode:
                pending_queue_size = self._pending_frame_indices.qsize()
                stats_msg = (
                    f"[Ros2FrameServer] Stats: received={self.frames_received}, "
                    f"processed={self.frames_processed}, dropped={self.frames_dropped}, "
                    f"drop_rate={drop_rate:.2f}%, queue_util={queue_utilization:.1f}%, "
                    f"pending_indices={pending_queue_size}, fallbacks={self.frames_index_fallback}"
                )
                # Add lost_indices to output if non-zero (severe stall indicator)
                if self.frames_index_lost > 0:
                    stats_msg += f", LOST_INDICES={self.frames_index_lost}"
                
                # V6: Add timing stats
                stats_msg += timing_stats_str
                logger.info(stats_msg)
            else:
                stats_msg = (
                    f"[Ros2FrameServer] Stats: received={self.frames_received}, "
                    f"processed={self.frames_processed}, dropped={self.frames_dropped}, "
                    f"drop_rate={drop_rate:.2f}%, queue_util={queue_utilization:.1f}%"
                )
                # V6: Add timing stats for non-accuracy mode too
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
            
            # Still convert to BGR for visualization, classification, and other components
            t_bgr_start = time.perf_counter()
            bgr = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)
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
        
        # Enqueue new frame with spool_frame_index for accuracy mode
        # SINGLE SOURCE OF TRUTH: spool_frame_index travels WITH the frame
        # In accuracy mode, dequeue the next pending spool_frame_index (FIFO correlation)
        spool_frame_index = 0
        if self._accuracy_mode:
            try:
                # Dequeue the next spool_frame_index that corresponds to this decoded frame
                # This maintains proper correlation even with decoder latency (10-100ms)
                # The index that was published BEFORE this H.264 frame is now matched
                # with the decoded NV12 frame that resulted from it.
                spool_frame_index = self._pending_frame_indices.get_nowait()
                logger.debug(f"[Ros2FrameServer] Correlated decoded frame with spool_frame_index {spool_frame_index}, pending_queue={self._pending_frame_indices.qsize()}")
            except queue.Empty:
                # No pending index - this shouldn't happen in normal operation
                # This indicates either:
                # 1. Frame indices not being published (SpoolProcessor issue)
                # 2. More decoded frames than published indices (decoder issue)
                # 3. System startup/shutdown race condition
                # Fall back to current index as best-effort but track this metric
                spool_frame_index = self.get_current_frame_index()
                self.frames_index_fallback += 1
                logger.warning(
                    f"[Ros2FrameServer] No pending frame index available (fallback #{self.frames_index_fallback}), "
                    f"using current index {spool_frame_index}. This may reintroduce race condition."
                )
        
        # V5 Optimization: Include raw NV12 data in frame tuple for direct BPU input
        # Frame format: (bgr, latency_ms, spool_frame_index, nv12_data, (height, width))
        # SINGLE SOURCE OF TRUTH: Attach spool_frame_index to frame data
        # It will travel through the entire pipeline with this frame
        frame_size = (msg.height, msg.width)
        self.frame_queue.put((bgr, latency_ms, spool_frame_index, nv12_data, frame_size))

    def frames(self):
        """
        Yield frames from the queue.
        
        V5 Optimization: Now yields NV12 data alongside BGR frame.
        
        Yields:
            Tuple of (frame, latency_ms, spool_frame_index, nv12_data, frame_size) - full format
            Tuple of (frame, latency_ms, spool_frame_index) - accuracy mode (backward compatible)
            Tuple of (frame, latency_ms) - normal mode (backward compatible)
            
        Where:
            - frame: BGR numpy array for visualization/classification
            - latency_ms: Frame latency in milliseconds
            - spool_frame_index: Frame index for ACK correlation
            - nv12_data: Raw NV12 numpy array for direct BPU inference (avoids BGR→NV12 conversion)
            - frame_size: Tuple (height, width) of the original frame
            
        Single Source of Truth:
            In accuracy mode, spool_frame_index travels WITH the frame data
            through the entire pipeline. This ensures perfect correlation
            for ACK - no need to query separate state.
        """
        # We check rclpy.ok() to ensure we stop if the ROS context shuts down
        while rclpy.ok():
            try:
                item = self.frame_queue.get(timeout=1)
                
                # V5: Handle new format with NV12 data (5 elements)
                if len(item) == 5:
                    frame, latency_ms, spool_frame_index, nv12_data, frame_size = item
                    # Store the frame index that was associated with THIS frame when it was enqueued
                    with self._frame_index_lock:
                        self._current_frame_index = spool_frame_index
                        self._last_yielded_frame_index = spool_frame_index
                    if self._accuracy_mode:
                        logger.debug(f"[Ros2FrameServer] Yielding frame with spool_frame_index {spool_frame_index}, nv12_shape={nv12_data.shape}")
                    # V5: Yield full frame data including NV12
                    yield frame, latency_ms, spool_frame_index, nv12_data, frame_size
                    
                # Handle old format (frame, latency, index) - backward compatible
                elif len(item) == 3:
                    frame, latency_ms, spool_frame_index = item
                    # Store the frame index that was associated with THIS frame when it was enqueued
                    # This is critical for ACK correlation in accuracy mode
                    with self._frame_index_lock:
                        self._current_frame_index = spool_frame_index
                        self._last_yielded_frame_index = spool_frame_index
                    if self._accuracy_mode:
                        logger.debug(f"[Ros2FrameServer] Yielding frame with spool_frame_index {spool_frame_index}")
                    # SINGLE SOURCE OF TRUTH: Yield index WITH frame data
                    yield frame, latency_ms, spool_frame_index
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