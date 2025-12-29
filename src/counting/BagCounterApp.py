"""
BagCounterApp - Main application orchestrator for the BreadBag Counter System.

V3 Performance Optimizations:
- Asynchronous classification pipeline with dedicated thread
- Smart frame dropping with backpressure awareness
- Reduced frame copy operations for memory efficiency
- Optimized queue management with adaptive thresholds
- Frame timing metrics for bottleneck identification

V5 Event-Centric Tracking:
- Optional event-centric tracking system (enabled by default)
- Centroid-based association instead of IoU
- Millisecond-based timing instead of frame counts
- Exit-boundary-based counting
"""

import os
import cv2
import json
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

from src.counting.Visualizer import Visualizer
from src.classifier.ClassifierService import ClassifierService
from src.classifier.BaseClassifier import BaseClassifier
from src.counting.BagStateMonitor import BagStateMonitor
from src.detection.BaseDetection import BaseDetector
from src.logging.Database import DatabaseManager
from src.frame_source.FrameSourceFactory import FrameSource, FrameSourceFactory
from src.tracking.BaseTracker import BaseTracker
from src import constants
from src.config.settings import config
from src.config.tracking_config import tracking_config

from src.logging.ConfigWatcher import ConfigWatcher
from src.utils.AppLogging import logger, structured_logger
from src.utils.platform import IS_RDK, IS_WINDOWS
from src.utils.PipelineMetrics import pipeline_metrics

from src.counting.IPC import ExecutorThread, init_ros2_context, shutdown_ros2_context
from src.counting.FramePublisherNode import FramePublisher

if IS_RDK:
    from rclpy.node import Node
    from std_msgs.msg import String
    from src.ros2_spool.messages import (
        generate_session_id,
        get_current_time_ros,
        ProcessingAck,
        ProcessingReady,
        processing_ack_to_ros_string,
        processing_ready_to_ros_string,
        frame_metadata_from_ros_string
    )
else:
    class Node:
        pass
    # Stubs for non-RDK platforms
    generate_session_id = lambda: "stub-session-id"
    get_current_time_ros = lambda: (0, 0)
    ProcessingAck = None
    ProcessingReady = None
    processing_ack_to_ros_string = lambda x: ""
    processing_ready_to_ros_string = lambda x: ""
    frame_metadata_from_ros_string = lambda x: None



class BagCounterApp:
    """
    Main application orchestrator for the BreadBag Counter System.
    
    V3 Performance Optimizations:
    - Asynchronous classification with dedicated thread to avoid blocking detection
    - Smart frame dropping based on queue pressure and processing time
    - Minimized frame copying to reduce memory bandwidth
    - Adaptive frame skip when pipeline is overloaded
    """
    
    # Queue configuration constants - V3: Optimized for 20fps at 720p (reduced memory pressure)
    INPUT_QUEUE_SIZE = 100  # Reduced from 500 (2.4s buffer vs 20s) - reduces memory from 1318MB to 158MB
    CLASSIFICATION_QUEUE_SIZE = 30  # Phase 2: Increased from 20 to reduce queue pressure
    CLASSIFICATION_WORKERS = 2  # Phase 2: Multiple workers for parallel classification
    QUEUE_WARNING_THRESHOLD = 50.0  # 50% - Lowered from 70% for earlier warnings (percentage 0-100)
    CRITICAL_QUEUE_THRESHOLD = 90.0  # 90% - emergency dropping threshold (percentage 0-100)
    STATS_LOG_INTERVAL = 5.0  # Log statistics every N seconds
    
    # Phase 1 Optimization: Visualization decimation for performance
    VISUALIZATION_DECIMATION = 2  # Visualize every Nth frame (2 = 50% reduction, publish at 10fps)
    
    # V3: Performance tuning constants - relaxed for 20fps target
    TARGET_FPS = 20.0  # Reduced from 25.0 for more headroom
    TARGET_FRAME_TIME_MS = 1000.0 / TARGET_FPS  # Computed dynamically (50ms for 20fps)
    MAX_DETECTION_TIME_MS = 40.0  # Increased from 31.0ms for more headroom
    ADAPTIVE_SKIP_THRESHOLD = 0.5  # Lowered from 0.7 for earlier intervention (decimal 0.0-1.0)
    PROACTIVE_BACKPRESSURE_THRESHOLD = 0.5  # Queue utilization to trigger proactive backpressure (decimal 0.0-1.0)
    
    # Skip rate cap configuration - more permissive under load
    SKIP_RATE_CAP = 0.07  # Increased from 0.02 (7% vs 2%)
    SKIP_RATE_WINDOW = 500  # Number of frames to track for skip rate calculation
    MIN_SKIP_SAMPLES = 10  # Minimum samples needed before applying skip rate logic
    SKIP_CAP_LOG_FREQUENCY = 5  # Log every Nth skip cap block to avoid flooding
    
    # System monitoring configuration
    SYSTEM_STATUS_LOG_INTERVAL = 900.0  # Log system status every 15 minutes (900 seconds)

    def __init__(
        self,
        video_path: str,
        detector_engine: BaseDetector,
        classifier_engine: BaseClassifier,
        db: DatabaseManager,
        is_development: bool,
    ):

        logger.info("[BagCounterApp] Initializing...")
        self.db = db
        self.detector = detector_engine
        self.classifier_service = ClassifierService(classifier_engine)

        self.config_watcher = ConfigWatcher(db.db_path, poll_interval=5)
        self.config_watcher.add_watch(constants.show_ui_screen_key, self.on_show_ui_changed)
        self.config_watcher.add_watch(constants.is_recording_key, self.on_is_recording_changed)

        self.is_running = False

        # Recording removed; snapshots only, but flag will now toggle based on config
        self.is_recording = False
        logger.info(
            f"[BagCounterApp] Video Recording: {'ENABLED' if self.is_recording else 'DISABLED'} (snapshots controlled by is_recording flag)")

        # Snapshot directory
        self.recording_dir = db.get_config_value(constants.recording_dir) or config.recording_dir
        self.snapshot_dir = os.path.join(self.recording_dir, "snapshots")
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"[Snapshot Saving] snapshot saving error due to -> {e}")
        logger.info(f"[BagCounterApp] Snapshot directory: {self.snapshot_dir}")

        # Determine if testing mode should be enabled for OpenCV frame source
        use_testing_mode = False
        if not IS_RDK:
            if getattr(config, "opencv_testing_mode", False):
                use_testing_mode = True
                logger.info("[BagCounterApp] Testing mode enabled via OPENCV_TESTING_MODE config")
            elif is_development:
                use_testing_mode = True
                logger.info("[BagCounterApp] Testing mode auto-enabled (development mode on non-RDK)")
            elif IS_WINDOWS:
                use_testing_mode = True
                logger.info("[BagCounterApp] Testing mode auto-enabled (on Windows)")

        self.testing_mode = use_testing_mode

        # Frame source instantiation (with testing_mode logic)
        if is_development:
            self.frame_source = FrameSourceFactory.create(
                "opencv",
                source=video_path,
                target_fps=None if use_testing_mode else 30.0,
                testing_mode=use_testing_mode
            )
            logger.info(f"[BagCounterApp] Development mode: reading from {video_path}")
        else:
            if IS_RDK:
                os.environ["HOME"] = "/home/sunrise"
                self.ros_executor = init_ros2_context()  # <-- initialize ROS2 *before* creating nodes
                self.frame_source = FrameSourceFactory.create("ros2", target_fps=30.0)
                logger.info("[BagCounterApp] Production mode: reading from ROS 2 stream")
            else:
                self.frame_source = FrameSourceFactory.create(
                    "opencv",
                    source=video_path,
                    target_fps=None if use_testing_mode else 30.0,
                    testing_mode=use_testing_mode
                )
                logger.info(f"[BagCounterApp] Windows mode: reading from {video_path}")

        if self.testing_mode:
            self.input_queue = None
        else:
            self.input_queue = queue.Queue(maxsize=self.INPUT_QUEUE_SIZE)

        # Async classification queue as normal
        self.classification_queue = queue.Queue(maxsize=self.CLASSIFICATION_QUEUE_SIZE)
        # Phase 2: Classification workers list (multiple threads)
        self._classification_threads = []
        self._classification_running = False
        
        # V4 Phase 1: Detection results queue (decouple detection from monitor)
        self.detection_queue_enabled = tracking_config.detection_queue_enabled
        if self.detection_queue_enabled and not self.testing_mode:
            self.detection_queue = queue.Queue(maxsize=tracking_config.detection_queue_size)
            self._monitor_thread = None
            self._monitor_running = False
            logger.info(
                f"[BagCounterApp] V4 Phase 1: Detection queue enabled "
                f"(size={tracking_config.detection_queue_size}, "
                f"warning_threshold={tracking_config.detection_queue_warning_threshold:.0%})"
            )
        else:
            self.detection_queue = None
            logger.info("[BagCounterApp] Detection queue disabled (legacy mode)")
        
        # V4 Phase 2: Batch inference configuration
        self.batch_inference_enabled = tracking_config.detection_batch_enabled
        if self.batch_inference_enabled:
            self._frame_batch = []
            self._batch_frame_data = []
            self._batch_start_time = None
            logger.info(
                f"[BagCounterApp] V4 Phase 2: Batch inference enabled "
                f"(batch_size={tracking_config.detection_batch_size}, "
                f"timeout={tracking_config.detection_batch_timeout_ms}ms)"
            )
        else:
            logger.info("[BagCounterApp] Batch inference disabled (legacy mode)")

        # Queue monitoring statistics
        self.stats_lock = threading.Lock()
        self.input_queue_drops = 0
        self.classification_queue_drops = 0  # V3: Track classification queue drops
        self.detection_queue_drops = 0  # V4 Phase 1: Track detection queue drops
        self.last_queue_stats_log_time = time.perf_counter()
        
        # V3: Performance metrics for adaptive processing
        self._recent_detection_times: deque = deque(maxlen=30)  # Last 30 detection times
        self._recent_frame_times: deque = deque(maxlen=30)  # Last 30 total frame times
        self._frames_skipped = 0
        self._adaptive_skip_enabled = True
        
        # Skip rate cap tracking
        self._skip_decisions: deque = deque(maxlen=self.SKIP_RATE_WINDOW)  # Track skip decisions
        self._skip_cap_blocks = 0  # Count how many times skip cap prevented skipping
        
        # Phase 1 Optimization: Visualization decimation counter
        self._visualization_counter = 0
        
        # System monitoring
        self._last_system_status_log = time.perf_counter()
        self._psutil_available = False
        self._psutil_module = None
        try:
            import psutil
            self._psutil_available = True
            self._psutil_module = psutil
            logger.info("[BagCounterApp] psutil available - system monitoring enabled")
        except ImportError:
            logger.info("[BagCounterApp] psutil not installed - system monitoring will be limited")
        
        # Degraded mode tracking
        self._degraded_mode_active = False
        self._recent_queue_delays: deque = deque(maxlen=20)  # Track queue delay
        self._last_degraded_mode_check = time.perf_counter()
        self._degraded_mode_check_interval = 2.0  # Check every 2 seconds
        
        # Smart frame skipping state
        self._smart_skip_frame_counter = 0  # Counter for pattern-based skipping
        self._frames_processed_in_degraded = 0  # Frames processed while in degraded mode
        self._frames_skipped_by_pattern = 0  # Frames skipped by smart pattern
        self._event_frame_counts: Dict[int, int] = {}  # Track frames per event: {event_id: frame_count}

        names = self.detector.class_names
        logger.info(f"[BagCounterApp] Detector class names: {names}")
        name_to_id = {v: k for k, v in names.items()}

        try:
            open_id = name_to_id["bread-bag-opened"]
            closed_id = name_to_id["bread-bag-closed"]
            logger.debug(f"[BagCounterApp] open_id={open_id}, closed_id={closed_id}")
        except KeyError as e:
            logger.error(f"[BagCounterApp] Model missing required class: {e}")
            logger.error(f"[BagCounterApp] Available classes: {list(name_to_id.keys())}")
            raise ValueError("Model missing required classes: bread-bag-opened, bread-bag-closed")

        # V5: Choose between event-centric and legacy tracking
        self.use_event_centric = tracking_config.use_event_centric_tracking
        if self.use_event_centric:
            from src.counting.EventCentricStateMonitor import EventCentricStateMonitor
            self.monitor = EventCentricStateMonitor(open_id, closed_id)
            logger.info("[BagCounterApp] Using V5 Event-Centric Tracking (centroid-based)")
        else:
            self.monitor = BagStateMonitor(open_id, closed_id)
            logger.info("[BagCounterApp] Using legacy IoU-based tracking")
        
        self.visualizer = Visualizer(names)
        # Set exit boundary margin for visualization
        if self.use_event_centric:
            self.visualizer.set_exit_margin(tracking_config.exit_boundary_margin_px)
        
        self.classifier_service.register_callback(self.on_classification_result)
        self.ui_counts = {}

        self.is_publishing = db.get_config_value(constants.show_ui_screen_key) == "1"
        logger.info(f"[BagCounterApp] IPC Publishing: {'ENABLED' if self.is_publishing else 'DISABLED'}")

        self.ipc_publisher = FramePublisher(publish_rate_hz=25.0)

        # -------------------------
        # Accuracy Mode: ACK publisher and READY publisher (for SpoolProcessor backpressure)
        # -------------------------
        self._accuracy_mode = (self.db.get_config_value(constants.accuracy_mode_enabled) == "1")
        self._ack_publisher = None
        self._ready_publisher = None
        self._metadata_subscriber = None
        self._ack_publisher_node = None
        self._current_frame_metadata = None  # Store latest frame metadata for ACK construction

        if IS_RDK and self._accuracy_mode:
            import rclpy
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
            
            # Generate session ID for this consumer instance
            self._consumer_session_id = generate_session_id()
            logger.info(f"[BagCounterApp] Accuracy Mode: session_id={self._consumer_session_id[:8]}")

            # QoS for READY - TRANSIENT_LOCAL for late joiners
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
                depth=20,
            )

            # Create a dedicated node for publishing ACKs and READY
            self._ack_publisher_node = rclpy.create_node("processing_ack_ready_publisher")
            
            # Publisher for ACK messages
            self._ack_publisher = self._ack_publisher_node.create_publisher(
                String,
                "/processing_ack",
                control_qos,
            )
            
            # Publisher for READY messages
            self._ready_publisher = self._ack_publisher_node.create_publisher(
                String,
                "/processing_ready",
                ready_qos,
            )
            
            # Subscriber for frame metadata
            self._metadata_subscriber = self._ack_publisher_node.create_subscription(
                String,
                "/spool/current_frame_metadata",
                self._frame_metadata_callback,
                control_qos
            )
            
            logger.info("[BagCounterApp] Accuracy Mode enabled - "
                       "ACK publisher on /processing_ack, "
                       "READY publisher on /processing_ready, "
                       "metadata subscriber on /spool/current_frame_metadata")

        if IS_RDK and self.ros_executor is not None:
            self.ros_executor.add_node(self.ipc_publisher)

            if self._accuracy_mode and self._ack_publisher_node is not None:
                self.ros_executor.add_node(self._ack_publisher_node)

        if IS_RDK and self.ros_executor is not None and isinstance(self.frame_source, Node):
            self.ros_executor.add_node(self.frame_source)
            logger.debug("[BagCounterApp] FrameSource added to ROS 2 executor")

        self.ros_thread = ExecutorThread(self.ros_executor)
        self.ros_thread.start()
        if IS_RDK:
            logger.debug("[BagCounterApp] ROS 2 executor thread started")

        # V3: Log performance configuration
        logger.info(
            f"[BagCounterApp] V3 Performance Config: target_fps={self.TARGET_FPS}, "
            f"target_frame_time={self.TARGET_FRAME_TIME_MS:.1f}ms, "
            f"max_detection_time={self.MAX_DETECTION_TIME_MS:.1f}ms, "
            f"adaptive_skip_threshold={self.ADAPTIVE_SKIP_THRESHOLD:.1%}, "
            f"skip_rate_cap={self.SKIP_RATE_CAP:.1%}, "
            f"input_queue={self.INPUT_QUEUE_SIZE}, classification_queue={self.CLASSIFICATION_QUEUE_SIZE}, "
            f"queue_warning_threshold={self.QUEUE_WARNING_THRESHOLD:.0f}%, "
            f"critical_queue_threshold={self.CRITICAL_QUEUE_THRESHOLD:.0f}%"
        )
        logger.info("[BagCounterApp] Initialization complete")

    def on_show_ui_changed(self, new_value):
        if new_value == "1":
            self.is_publishing = True
            logger.info("[BagCounterApp] IPC Publishing ENABLED")
        else:
            self.is_publishing = False
            logger.info("[BagCounterApp] IPC Publishing DISABLED")

    def on_is_recording_changed(self, new_value):
        # Now toggle flag based on config key
        self.is_recording = (new_value == "1" or new_value is True or new_value == 1)
        logger.info(f"[BagCounterApp] is_recording set to {self.is_recording}")

    def _convert_detections(self, detections):
        """Normalize detector output to a list of dicts with box, class_id, conf."""
        if detections is None:
            return []

        # Case 1: single object with .boxes (e.g., BpuResultWrapper)
        if hasattr(detections, "boxes"):
            det_obj = detections
        else:
            # Case 2: sequence, take first element if present
            try:
                if len(detections) == 0:
                    return []
                det_obj = detections[0]
            except TypeError:
                # Not len()-able, assume single detection object
                det_obj = detections

        if not hasattr(det_obj, "boxes") or det_obj.boxes is None:
            return []

        boxes = det_obj.boxes
        if not hasattr(boxes, "xyxy") or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        return [
            {"box": xyxy[i], "class_id": cls_ids[i], "conf": confidences[i]}
            for i in range(len(cls_ids))
        ]

    def _log_system_status(self):
        """
        Log system resource usage (CPU, RAM) if psutil is available.
        Called periodically (every 15 minutes) to monitor system health.
        """
        if not self._psutil_available or self._psutil_module is None:
            return
        
        try:
            # Use cached psutil module
            psutil = self._psutil_module
            
            # Get CPU usage (average over short interval)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            
            # Log system status
            logger.info(
                f"[SystemStatus] CPU: {cpu_percent:.1f}%, "
                f"RAM: {memory_percent:.1f}% ({memory_used_mb:.1f}MB / {memory_total_mb:.1f}MB)"
            )
            
            # Also log via structured logging
            structured_logger.pipeline_summary({
                "event": "system_status",
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb
            })
            
        except Exception as e:
            logger.warning(f"[SystemStatus] Failed to log system status: {e}")
    
    def _log_queue_stats(self):
        """
        Phase 2 Optimization: Extracted queue stats logging to reduce hot path overhead.
        Called every STATS_LOG_INTERVAL seconds from the frame capture thread.
        """
        input_size = self.input_queue.qsize()
        input_utilization = (input_size / self.INPUT_QUEUE_SIZE) * 100
        class_size = self.classification_queue.qsize()
        class_utilization = (class_size / self.CLASSIFICATION_QUEUE_SIZE) * 100
        
        with self.stats_lock:
            input_drops = self.input_queue_drops
            class_drops = self.classification_queue_drops
            detection_drops = self.detection_queue_drops if self.detection_queue_enabled else 0
        
        # V4 Phase 1: Detection queue stats
        detection_stats = ""
        if self.detection_queue_enabled and self.detection_queue is not None:
            detection_size = self.detection_queue.qsize()
            detection_utilization = (detection_size / tracking_config.detection_queue_size) * 100
            detection_stats = (
                f" | Detection: {detection_size}/{tracking_config.detection_queue_size} "
                f"({detection_utilization:.1f}% full, drops={detection_drops})"
            )
        
        # Calculate current skip rate (O(n) but acceptable since this runs every 5s)
        current_skip_rate = 0.0
        if len(self._skip_decisions) > 0:
            current_skip_rate = (sum(self._skip_decisions) / len(self._skip_decisions)) * 100
        
        # Calculate smart skip statistics
        smart_skip_info = ""
        if tracking_config.degraded_mode_smart_skip_enabled and self._frames_processed_in_degraded > 0:
            total_degraded_frames = self._frames_processed_in_degraded + self._frames_skipped_by_pattern
            pattern_skip_rate = (self._frames_skipped_by_pattern / total_degraded_frames) * 100
            smart_skip_info = f" | SmartSkip: {self._frames_skipped_by_pattern} frames (rate={pattern_skip_rate:.1f}% in degraded)"
        
        logger.info(
            f"[QueueStats] Input: {input_size}/{self.INPUT_QUEUE_SIZE} "
            f"({input_utilization:.1f}% full, drops={input_drops})"
            f"{detection_stats} | "
            f"Classification: {class_size}/{self.CLASSIFICATION_QUEUE_SIZE} "
            f"({class_utilization:.1f}% full, drops={class_drops}) | "
            f"Skipped: {self._frames_skipped} (rate={current_skip_rate:.1f}%, cap={self.SKIP_RATE_CAP*100:.1f}%) | "
            f"SkipCapBlocks: {self._skip_cap_blocks}"
            f"{smart_skip_info}"
        )
        
        if input_utilization > self.QUEUE_WARNING_THRESHOLD:
            # Enhanced warning with root cause information
            avg_detect_time = (
                sum(self._recent_detection_times) / len(self._recent_detection_times)
                if len(self._recent_detection_times) > 0 else 0.0
            )
            logger.warning(
                f"[InputQueuePressure] High queue utilization: {input_utilization:.1f}% "
                f"(threshold={self.QUEUE_WARNING_THRESHOLD:.0f}%) - "
                f"Root cause: avg_detection_time={avg_detect_time:.1f}ms "
                f"(target={self.TARGET_FRAME_TIME_MS:.1f}ms). "
                f"Risk: frames may be dropped if processing doesn't improve."
            )
        
        if class_utilization > self.QUEUE_WARNING_THRESHOLD:
            logger.warning(
                f"[ClassificationQueuePressure] High queue utilization: {class_utilization:.1f}% "
                f"(threshold={self.QUEUE_WARNING_THRESHOLD:.0f}%) - "
                f"classification thread is falling behind. "
                f"Risk: classification tasks may be dropped."
            )
        
        # V4 Phase 1: Detection queue warning
        if self.detection_queue_enabled and self.detection_queue is not None:
            detection_size = self.detection_queue.qsize()
            detection_utilization = detection_size / tracking_config.detection_queue_size
            if detection_utilization > tracking_config.detection_queue_warning_threshold:
                logger.warning(
                    f"[DetectionQueuePressure] High queue utilization: {detection_utilization:.1%} "
                    f"(threshold={tracking_config.detection_queue_warning_threshold:.0%}) - "
                    f"monitor thread is falling behind detection. "
                    f"Risk: detection results may be dropped."
                )
    
    def _check_degraded_mode(self, queue_utilization: float) -> bool:
        """
        Check if system should enter degraded mode based on load metrics.
        
        Degraded mode reduces non-critical work to maintain tracking reliability.
        
        Args:
            queue_utilization: Current input queue utilization (0.0 - 1.0)
            
        Returns:
            True if degraded mode should be active
        """
        if not tracking_config.degraded_mode_enabled:
            return False
        
        current_time = time.perf_counter()
        
        # Only check periodically to avoid overhead
        if current_time - self._last_degraded_mode_check < self._degraded_mode_check_interval:
            return self._degraded_mode_active
        
        self._last_degraded_mode_check = current_time
        
        # Check queue utilization threshold
        queue_overload = queue_utilization > tracking_config.degraded_mode_queue_threshold
        
        # Check average queue delay if we have enough samples
        delay_overload = False
        if len(self._recent_queue_delays) >= 10:
            avg_delay_ms = sum(self._recent_queue_delays) / len(self._recent_queue_delays)
            delay_overload = avg_delay_ms > tracking_config.degraded_mode_delay_threshold_ms
        
        # Activate degraded mode if either condition is met
        should_activate = queue_overload or delay_overload
        
        # Log state transitions
        if should_activate and not self._degraded_mode_active:
            logger.warning(
                f"[BagCounterApp] ENTERING DEGRADED MODE: "
                f"queue_util={queue_utilization:.1%}, "
                f"avg_delay={sum(self._recent_queue_delays) / len(self._recent_queue_delays) if self._recent_queue_delays else 0:.1f}ms"
            )
            structured_logger.pipeline_error(
                component="BagCounterApp",
                operation="degraded_mode_activation",
                error_type="PerformanceDegradation",
                error_message="System entering degraded mode due to overload",
                affected_ids=None,
                context={
                    "queue_utilization": queue_utilization,
                    "avg_queue_delay_ms": sum(self._recent_queue_delays) / len(self._recent_queue_delays) if self._recent_queue_delays else 0,
                    "queue_threshold": tracking_config.degraded_mode_queue_threshold,
                    "delay_threshold_ms": tracking_config.degraded_mode_delay_threshold_ms
                }
            )
        elif not should_activate and self._degraded_mode_active:
            logger.info(
                f"[BagCounterApp] EXITING DEGRADED MODE: "
                f"queue_util={queue_utilization:.1%}, system recovered"
            )
            # Reset smart skip counters when exiting degraded mode
            if tracking_config.degraded_mode_smart_skip_enabled:
                self._smart_skip_frame_counter = 0
        
        self._degraded_mode_active = should_activate
        return self._degraded_mode_active
    
    def _should_smart_skip_frame(self, queue_utilization: float, active_events: list) -> Tuple[bool, str]:
        """
        Determine if current frame should be skipped using smart pattern-based logic.
        
        Smart skipping ensures:
        - Events get minimum required frames for tracking
        - Critical states are never skipped
        - Skip pattern adapts to queue pressure
        
        Args:
            queue_utilization: Current queue utilization (0.0 - 1.0)
            active_events: List of currently active events
            
        Returns:
            Tuple of (should_skip: bool, reason: str)
        """
        if not tracking_config.degraded_mode_smart_skip_enabled:
            return (False, "smart_skip_disabled")
        
        if not self._degraded_mode_active:
            return (False, "not_in_degraded_mode")
        
        # Check if we should skip when no active events
        if len(active_events) == 0:
            if tracking_config.degraded_mode_skip_with_active_events_only:
                return (False, "no_active_events")
        
        # Check for critical states that should never be skipped
        if tracking_config.degraded_mode_preserve_critical_states:
            for event in active_events:
                # Check for CLOSING state (critical for state transition)
                if hasattr(event, 'state') and event.state == 'CLOSING':
                    return (False, "event_in_closing_state")
                
                # Check for early OPEN state (critical for initial tracking)
                if hasattr(event, 'state') and event.state == 'OPEN':
                    event_id = getattr(event, 'event_id', None) or getattr(event, 'id', None)
                    if event_id is not None:
                        frames_for_this_event = self._event_frame_counts.get(event_id, 0)
                        if frames_for_this_event < tracking_config.degraded_mode_critical_state_frame_threshold:
                            return (False, f"event_{event_id}_early_open")
        
        # Check minimum frames per event requirement
        for event in active_events:
            event_id = getattr(event, 'event_id', None) or getattr(event, 'id', None)
            if event_id is not None:
                frames_for_this_event = self._event_frame_counts.get(event_id, 0)
                if frames_for_this_event < tracking_config.degraded_mode_min_frames_per_event:
                    # Don't skip - event needs more frames
                    return (False, f"event_{event_id}_needs_frames")
        
        # Determine skip pattern based on configuration
        skip_pattern = tracking_config.degraded_mode_skip_pattern
        
        # Increment frame counter
        self._smart_skip_frame_counter += 1
        
        should_skip = False
        reason = ""
        
        if skip_pattern == 'every_2nd':
            # Skip every 2nd frame (50% reduction)
            should_skip = (self._smart_skip_frame_counter % 2 == 0)
            reason = "pattern_every_2nd"
            
        elif skip_pattern == 'every_3rd':
            # Skip every 3rd frame (33% reduction)
            should_skip = (self._smart_skip_frame_counter % 3 == 0)
            reason = "pattern_every_3rd"
            
        elif skip_pattern == 'adaptive':
            # Adaptive skip based on queue pressure
            if queue_utilization < 0.5:
                # Below 50%: no pattern skipping (rely on existing adaptive skip)
                should_skip = False
                reason = "adaptive_low_queue"
                
            elif queue_utilization < 0.7:
                # 50-70%: skip every 3rd frame (mild load)
                should_skip = (self._smart_skip_frame_counter % 3 == 0)
                reason = "adaptive_every_3rd"
                
            elif queue_utilization < 0.85:
                # 70-85%: skip every 2nd frame (moderate load)
                should_skip = (self._smart_skip_frame_counter % 2 == 0)
                reason = "adaptive_every_2nd"
                
            elif queue_utilization < 0.95:
                # 85-95%: skip 2 out of 3 frames (heavy load)
                should_skip = (self._smart_skip_frame_counter % 3 != 0)
                reason = "adaptive_2_of_3"
                
            else:
                # 95%+: skip 3 out of 4 frames (critical load)
                should_skip = (self._smart_skip_frame_counter % 4 != 0)
                reason = "adaptive_3_of_4"
        
        # Enforce max skip rate when events are active
        if should_skip and len(active_events) > 0:
            # Calculate what the skip rate would be
            if self._frames_processed_in_degraded > 0:
                current_pattern_skip_rate = self._frames_skipped_by_pattern / (
                    self._frames_processed_in_degraded + self._frames_skipped_by_pattern
                )
                if current_pattern_skip_rate >= tracking_config.degraded_mode_max_skip_rate_with_events:
                    return (False, "max_skip_rate_exceeded")
        
        return (should_skip, reason)

    # --- Snapshot helpers ---
    def _annotate_frame(self, frame, detections, label, conf, event_box=None):
        annotated = frame.copy()
        for det in detections:
            box = det.get("box")
            cls_id = det.get("class_id")
            det_conf = det.get("conf", 0)
            if box is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls_id == self.monitor.closed_id else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            name = self.detector.class_names.get(cls_id, "cls")
            cv2.putText(
                annotated,
                f"{name}:{det_conf:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        if event_box is not None:
            x1, y1, x2, y2 = map(int, event_box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                annotated,
                "event",
                (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"FINAL: {label} ({conf:.2f})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _save_snapshot(self, track_id, label, conf, phash, image_path, candidates_count, context: Dict[str, Any]):
        frame = context.get("frame")
        detections = context.get("detections", [])
        event_box = context.get("event_box")
        event_stats = context.get("event_stats")
        frame_id = context.get("frame_id")
        ts_epoch = context.get("timestamp", time.time())

        timestamp_str = datetime.fromtimestamp(float(ts_epoch)).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"{timestamp_str}_track{track_id}_{label}"
        orig_file = os.path.join(self.snapshot_dir, base_name + "_orig.jpg")
        ann_file = os.path.join(self.snapshot_dir, base_name + "_ann.jpg")
        meta_file = os.path.join(self.snapshot_dir, base_name + ".json")

        os.makedirs(self.snapshot_dir, exist_ok=True)

        cv2.imwrite(orig_file, frame)
        annotated = self._annotate_frame(frame, detections, label, conf, event_box)
        cv2.imwrite(ann_file, annotated)

        det_json = []
        for d in detections:
            box = d.get("box")
            det_json.append({
                "box": (
                    box.tolist() if hasattr(box, "tolist")
                    else [float(x) for x in box] if box is not None
                    else None
                ),
                "class_id": int(d.get("class_id")) if d.get("class_id") is not None else None,
                "class_name": self.detector.class_names.get(
                    int(d.get("class_id")), "Unknown"
                ) if d.get("class_id") is not None else "Unknown",
                "conf": float(d.get("conf", 0)),
            })

        meta = {
            "timestamp": timestamp_str,
            "timestamp_epoch": float(ts_epoch),
            "frame_id": int(frame_id) if frame_id is not None else None,
            "track_id": int(track_id) if track_id is not None else None,
            "label": label,
            "confidence": float(conf),
            "phash": phash,
            "roi_saved_path": image_path,
            "candidates_evaluated": int(candidates_count) if candidates_count is not None else None,
            "event_box": (
                event_box.tolist() if hasattr(event_box, "tolist")
                else [float(x) for x in event_box] if event_box is not None
                else None
            ),
            "event_stats": event_stats,
            "detections": det_json,
            "files": {
                "original": orig_file,
                "annotated": ann_file,
            },
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[BagCounterApp] Snapshot saved: {meta_file}")

    # --- Callbacks ---
    
    def _frame_metadata_callback(self, msg):
        """
        Callback for receiving frame metadata from spool processor.
        
        Stores the metadata for use when constructing ACKs.
        """
        try:
            metadata = frame_metadata_from_ros_string(msg.data)
            self._current_frame_metadata = metadata
            logger.debug(f"[BagCounterApp] Frame metadata received: "
                        f"frame_index={metadata.frame_index}, seq={metadata.seq}, "
                        f"session_id={metadata.session_id[:8]}")
        except Exception as e:
            logger.error(f"[BagCounterApp] Error parsing frame metadata: {e}")

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        label = data["label"]
        phash = data["phash"]
        image_path = data["image_path"]
        conf = data.get("confidence", 1.0)
        candidates_count = data.get("candidates_evaluated", 1)
        context = data.get("context")
        metadata = data.get("metadata", {})

        # V7.2: Determine confidence tier
        # Priority:
        # 1. Use track_confidence_tier from metadata if available (from disambiguation/evidence accumulator)
        # 2. Fall back to threshold-based determination
        confidence_tier = metadata.get('track_confidence_tier')
        if not confidence_tier:
            # Legacy fallback: use confidence threshold
            confidence_tier = 'high' if conf >= tracking_config.high_confidence_threshold else 'low'
        
        bag_type_id = self.db.get_or_create_bag_type(label, phash, image_path)
        self.db.log_event(bag_type_id, track_id, conf, confidence_tier)

        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1
        
        # Structured logging for count update
        structured_logger.count_updated(
            bag_type=label,
            new_count=self.ui_counts[label],
            track_id=track_id,
            confidence=conf,
            phash=phash if phash else "None",
            candidates_evaluated=candidates_count
        )

        # Save snapshot only if context is available and is_recording flag enabled
        if self.is_recording and context and context.get("frame") is not None:
            try:
                self._save_snapshot(track_id, label, conf, phash, image_path, candidates_count, context)
            except Exception as e:
                structured_logger.pipeline_error(
                    component="BagCounterApp",
                    operation="snapshot_save",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=[track_id],
                    context={"label": label, "phash": phash}
                )

    def _publish_processing_ack(self, spool_frame_index: int):
        """
        Publish processing ACK for Accuracy Mode using structured message.
        
        Uses the frame metadata received from the processor to construct a proper ACK
        with session_id, seq, and timing information.
        
        Args:
            spool_frame_index: The canonical frame index (for backward compatibility check)
        
        CRITICAL: This must be called IMMEDIATELY after a frame is consumed.
        """
        if not IS_RDK or not getattr(self, "_accuracy_mode", False) or self._ack_publisher is None:
            return
        
        try:
            # Use stored metadata if available, otherwise construct minimal ACK
            if self._current_frame_metadata is not None:
                metadata = self._current_frame_metadata
                
                # Validate frame index matches
                if metadata.frame_index != spool_frame_index:
                    logger.warning(f"[BagCounterApp] ⚠ Frame index mismatch: "
                                 f"expected {spool_frame_index}, metadata has {metadata.frame_index}")
                
                # Construct ACK with full metadata
                ack = ProcessingAck(
                    frame_index=metadata.frame_index,
                    session_id=metadata.session_id,
                    seq=metadata.seq,
                    sent_time_sec=metadata.sent_time_sec,
                    sent_time_nsec=metadata.sent_time_nsec,
                    segment_num=metadata.segment_num
                )
                
                # Publish ACK
                ack_msg = String()
                ack_msg.data = processing_ack_to_ros_string(ack)
                self._ack_publisher.publish(ack_msg)
                
                logger.info(f"[BagCounterApp] ✓ ACK published: frame_index={ack.frame_index}, "
                          f"seq={ack.seq}, session={ack.session_id[:8]}")
            else:
                # Fallback: construct ACK without full metadata (should not happen normally)
                logger.warning(f"[BagCounterApp] ⚠ No frame metadata available for frame {spool_frame_index}, "
                             "constructing minimal ACK")
                
                sent_time_sec, sent_time_nsec = get_current_time_ros()
                ack = ProcessingAck(
                    frame_index=spool_frame_index,
                    session_id=self._consumer_session_id,
                    seq=0,  # Unknown seq
                    sent_time_sec=sent_time_sec,
                    sent_time_nsec=sent_time_nsec,
                    segment_num=-1
                )
                
                ack_msg = String()
                ack_msg.data = processing_ack_to_ros_string(ack)
                self._ack_publisher.publish(ack_msg)
                
                logger.info(f"[BagCounterApp] ✓ ACK published (minimal): frame_index={ack.frame_index}")
                
        except Exception as e:
            logger.warning(f"[BagCounterApp] Failed to publish ACK for frame {spool_frame_index}: {e}")
    
    def _publish_processing_ready(self):
        """
        Publish READY signal for Accuracy Mode.
        
        This tells the spool processor that the consumer is ready to process frames.
        Should be called after all publishers/subscribers are initialized.
        """
        if not IS_RDK or not getattr(self, "_accuracy_mode", False) or self._ready_publisher is None:
            return
        
        try:
            ready_time_sec, ready_time_nsec = get_current_time_ros()
            ready = ProcessingReady(
                session_id=self._consumer_session_id,
                ready_time_sec=ready_time_sec,
                ready_time_nsec=ready_time_nsec
            )
            
            ready_msg = String()
            ready_msg.data = processing_ready_to_ros_string(ready)
            self._ready_publisher.publish(ready_msg)
            
            logger.info(f"[BagCounterApp] ✓ READY published: session_id={ready.session_id[:8]}")
        except Exception as e:
            logger.error(f"[BagCounterApp] Failed to publish READY: {e}")

    # --- V3: Async Classification Thread ---
    
    def _classification_thread_loop(self):
        """
        V3: Dedicated thread for classification processing.
        
        This separates classification from the main detection loop, allowing
        detection to continue while classification runs in parallel. This is
        critical for achieving 25fps throughput.
        """
        logger.info("[ClassificationThread] Started")
        
        while self._classification_running:
            try:
                # Wait for classification task with timeout
                task = self.classification_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._classification_running:
                    break
                continue
            except Exception as e:
                structured_logger.pipeline_error(
                    component="ClassificationThread",
                    operation="queue_get",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=None,
                    context={"queue_size": self.classification_queue.qsize()}
                )
                continue
            
            try:
                event_id, candidates, context = task
                classify_start = time.perf_counter()
                
                # Process classification (this is the slow part)
                self.classifier_service.process(event_id, candidates, context=context)
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                
                # Structured error logging
                structured_logger.pipeline_error(
                    component='ClassificationThread',
                    operation='classification_processing',
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=[event_id] if 'event_id' in locals() else None,
                    context={
                        'candidates_count': len(candidates) if 'candidates' in locals() else 0,
                        'classification_queue_size': self.classification_queue.qsize()
                    },
                    traceback=error_trace
                )
        
        logger.info("[ClassificationThread] Stopped")
    
    # --- V4 Phase 1: Monitor Thread (consumes detection results) ---
    
    def _monitor_thread_loop(self):
        """
        V4 Phase 1: Dedicated thread for monitor processing.
        
        Consumes detection results from detection_queue and processes them
        through the monitor (EventCentricStateMonitor). This decouples
        detection from monitoring, allowing detection to run at full BPU speed.
        
        Expected benefit: ~30-40% performance improvement.
        """
        logger.info("[MonitorThread] Started (V4 Phase 1: Detection Queue decoupling)")
        
        while self._monitor_running:
            try:
                # Wait for detection result with timeout
                detection_result = self.detection_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._monitor_running:
                    break
                continue
            except Exception as e:
                structured_logger.pipeline_error(
                    component="MonitorThread",
                    operation="queue_get",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=None,
                    context={"queue_size": self.detection_queue.qsize()}
                )
                continue
            
            try:
                # Unpack detection result
                current_frame_detections, frame_count, frame, detect_time = detection_result
                
                # Process through monitor
                monitor_start = time.perf_counter()
                ready_events = self.monitor.update(current_frame_detections,
                                                   {"frame_count": frame_count, "frame": frame})
                monitor_end = time.perf_counter()
                monitor_time = (monitor_end - monitor_start) * 1000
                
                # Track degraded mode state for ROI saving
                in_degraded_mode = self._degraded_mode_active
                
                # Queue ready events for classification
                if ready_events:
                    for event_id, candidates, event_box, event_stats in ready_events:
                        # Determine if we should save snapshot
                        should_save_snapshot = self.is_recording and not (
                            in_degraded_mode and tracking_config.degraded_mode_disable_roi_saving
                        )
                        
                        # Build context
                        if should_save_snapshot:
                            det_copy = []
                            for d in current_frame_detections:
                                det_copy.append({
                                    "box": d["box"].copy(),
                                    "class_id": d["class_id"],
                                    "conf": float(d.get("conf", 0)),
                                })
                            try:
                                event_box_copy = event_box.copy() if hasattr(event_box, 'copy') else list(event_box)
                            except (TypeError, AttributeError):
                                event_box_copy = event_box
                            context = {
                                "frame": frame.copy(),
                                "detections": det_copy,
                                "event_box": event_box_copy,
                                "event_stats": event_stats,
                                "frame_id": frame_count,
                                "timestamp": time.time(),
                            }
                        else:
                            context = {
                                "frame": None,
                                "detections": [],
                                "event_box": event_box,
                                "event_stats": event_stats,
                                "frame_id": frame_count,
                                "timestamp": time.time(),
                            }
                        
                        # Enqueue for classification
                        self._enqueue_classification(event_id, candidates, context)
                
                # Record metrics
                pipeline_metrics.record_detection(
                    current_frame_detections,
                    detect_time,
                    self.monitor.open_id,
                    self.monitor.closed_id
                )
                
                # Log timing periodically
                if frame_count % 30 == 0:
                    logger.info(
                        f"[MonitorThread] Frame {frame_count}: "
                        f"detect={detect_time:.1f}ms, monitor={monitor_time:.1f}ms, "
                        f"queue_size={self.detection_queue.qsize()}/{tracking_config.detection_queue_size}"
                    )
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                
                structured_logger.pipeline_error(
                    component='MonitorThread',
                    operation='monitor_processing',
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=[frame_count] if 'frame_count' in locals() else None,
                    context={
                        'detections_count': len(current_frame_detections) if 'current_frame_detections' in locals() else 0,
                        'detection_queue_size': self.detection_queue.qsize()
                    },
                    traceback=error_trace
                )
        
        logger.info("[MonitorThread] Stopped")
    
    def _enqueue_classification(self, event_id: int, candidates: List, context: Dict[str, Any]) -> bool:
        """
        V3: Enqueue classification task for async processing.
        
        Returns True if task was enqueued, False if queue is full (task dropped).
        """
        task = (event_id, candidates, context)
        
        try:
            self.classification_queue.put_nowait(task)
            return True
        except queue.Full:
            # Queue is full - drop the oldest task and try again
            try:
                dropped = self.classification_queue.get_nowait()
                dropped_event_id = dropped[0] if dropped and len(dropped) > 0 else None
                with self.stats_lock:
                    self.classification_queue_drops += 1
                queue_util = self.classification_queue.qsize() / self.CLASSIFICATION_QUEUE_SIZE if self.CLASSIFICATION_QUEUE_SIZE > 0 else 1.0
                structured_logger.queue_backpressure(
                    queue_name='classification_queue',
                    utilization=queue_util,
                    drops=self.classification_queue_drops,
                    action='drop_oldest_task',
                    dropped_event_id=dropped_event_id,
                    new_event_id=event_id
                )
                self.classification_queue.put_nowait(task)
                return True
            except queue.Empty:
                pass
            except queue.Full:
                with self.stats_lock:
                    self.classification_queue_drops += 1
                queue_util = self.classification_queue.qsize() / self.CLASSIFICATION_QUEUE_SIZE if self.CLASSIFICATION_QUEUE_SIZE > 0 else 1.0
                structured_logger.queue_backpressure(
                    queue_name='classification_queue',
                    utilization=queue_util,
                    drops=self.classification_queue_drops,
                    action='failed_enqueue',
                    event_id=event_id
                )
                return False
        
        return False

    # --- If testing mode, process frames inline, else keep old threaded pipeline ---
    def _process_frame_inline(self, frame, frame_count):
        try:
            frame_start = time.perf_counter()
            detect_start = time.perf_counter()
            detections = self.detector.predict(frame)
            detect_end = time.perf_counter()
            detect_time = (detect_end - detect_start) * 1000
            self._recent_detection_times.append(detect_time)
            current_frame_detections = []
            if len(detections) > 0 and hasattr(detections[0], "boxes") and len(detections[0].boxes) > 0:
                xyxy = detections[0].boxes.xyxy.cpu().numpy()
                cls_ids = detections[0].boxes.cls.cpu().numpy().astype(int)
                confidences = detections[0].boxes.conf.cpu().numpy()
                for i in range(len(cls_ids)):
                    current_frame_detections.append(
                        {"box": xyxy[i], "class_id": cls_ids[i], "conf": confidences[i]}
                    )
            pipeline_metrics.record_detection(
                current_frame_detections,
                detect_time,
                self.monitor.open_id,
                self.monitor.closed_id,
            )
            ready_events = self.monitor.update(current_frame_detections,
                                               {"frame_count": frame_count, "frame": frame})
            if ready_events:
                for event_id, candidates, event_box, event_stats in ready_events:
                    if self.is_recording:
                        det_copy = []
                        for d in current_frame_detections:
                            det_copy.append({
                                "box": d["box"].copy(),
                                "class_id": d["class_id"],
                                "conf": float(d.get("conf", 0)),
                            })
                        try:
                            event_box_copy = event_box.copy() if hasattr(event_box, 'copy') else list(event_box)
                        except Exception:
                            event_box_copy = event_box
                        context = {
                            "frame": frame.copy(),
                            "detections": det_copy,
                            "event_box": event_box_copy,
                            "event_stats": event_stats,
                            "frame_id": frame_count,
                            "timestamp": time.time(),
                        }
                    else:
                        context = {
                            "frame": None,
                            "detections": [],
                            "event_box": event_box,
                            "event_stats": event_stats,
                            "frame_id": frame_count,
                            "timestamp": time.time(),
                        }
                    self._enqueue_classification(event_id, candidates, context)
            if self.is_publishing:
                # V3 Performance: Resize BEFORE visualization
                annotated_frame = cv2.resize(frame, (1280, 720))
                self.visualizer.render_all(
                    annotated_frame,
                    [],
                    self.monitor.active_events,
                    counts=self.ui_counts,
                    fps=0,
                )
                # V3 Performance: No need to resize again
                self.ipc_publisher.publish(annotated_frame)
            pipeline_metrics.maybe_log_summary()
        except Exception as e:
            logger.error(f"[Inline] Error processing frame {frame_count}: {e}")
            import traceback
            logger.debug(f"[Inline] Traceback:\n{traceback.format_exc()}")

    # --- Main logic thread ---

    def _logic_thread_loop(self):
        """
        V4: Optimized main logic loop with detection queue and batch inference.
        
        Key optimizations:
        - V3: Classification offloaded to separate thread
        - V4 Phase 1: Detection decoupled from monitor via queue
        - V4 Phase 2: Batch inference for 40-60% speedup
        - Frame skipping when detection is too slow
        - Reduced frame copying operations
        """
        if self.detection_queue_enabled:
            logger.info("[LogicThread] Started (V4 with detection queue and batch inference)")
        elif self.batch_inference_enabled:
            logger.info("[LogicThread] Started (V4 with batch inference)")
        else:
            logger.info("[LogicThread] Started (V3 with async classification)")

        TIMING_LOG_INTERVAL = 30
        frame_count = 0

        while self.is_running:
            try:
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                if not self.is_running:
                    break
                continue
            except Exception as e:
                structured_logger.pipeline_error(
                    component="LogicThread",
                    operation="queue_get",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=None,
                    context={"input_queue_size": self.input_queue.qsize()}
                )
                continue

            try:
                frame_count += 1
                # NOTE: Accuracy Mode ACK is now published in the frame capture loop (run() method)
                # immediately when a frame is consumed from frame_source.frames().
                # This ensures the ACK has the correct frame index that was stored with that frame.
                # 
                # HISTORICAL CONTEXT (why ACK was moved from here):
                # Previously, ACK was published here using get_current_frame_index(), but this
                # caused a race condition: the frame index in Ros2FrameServer could be updated
                # by the subscription callback before the logic thread processed the frame,
                # resulting in ACKs with mismatched indices and pipeline deadlock.
                # Moving the ACK to the capture loop ensures we use the index that was
                # captured when the specific frame was enqueued, not a potentially newer index.
                #
                # DO NOT publish ACK here as it would cause duplicate ACKs and potential race conditions.

                frame_start = time.perf_counter()
                
                # V3: Check if we should skip processing due to backpressure
                queue_utilization = self.input_queue.qsize() / self.INPUT_QUEUE_SIZE
                
                # Check for degraded mode
                in_degraded_mode = self._check_degraded_mode(queue_utilization)
                
                # V3: Compute average detection time efficiently (deque sum is O(n) but n is small)
                avg_detection_time = (
                    sum(self._recent_detection_times) / len(self._recent_detection_times)
                    if len(self._recent_detection_times) > 5 else 0.0
                )
                
                # Calculate current skip rate - cache sum to avoid redundant O(n) operations
                current_skip_rate = 0.0
                skip_sum = 0
                if len(self._skip_decisions) >= self.MIN_SKIP_SAMPLES:
                    skip_sum = sum(self._skip_decisions)
                    current_skip_rate = skip_sum / len(self._skip_decisions)
                
                # V3 Performance: Critical queue threshold for emergency dropping
                critical_queue_exceeded = queue_utilization >= (self.CRITICAL_QUEUE_THRESHOLD / 100.0)
                
                # V3 Performance: Proactive backpressure at 50% queue utilization
                proactive_backpressure = queue_utilization >= self.PROACTIVE_BACKPRESSURE_THRESHOLD and avg_detection_time > self.MAX_DETECTION_TIME_MS
                
                # Determine if adaptive skip conditions are met
                adaptive_skip_conditions_met = (
                    self._adaptive_skip_enabled and (
                        critical_queue_exceeded or  # Emergency: always skip
                        proactive_backpressure or   # Proactive: skip at 50% if slow
                        (queue_utilization > self.ADAPTIVE_SKIP_THRESHOLD and avg_detection_time > self.MAX_DETECTION_TIME_MS)
                    )
                )
                
                # Check if skip rate cap would be exceeded
                skip_rate_cap_exceeded = False
                if adaptive_skip_conditions_met and len(self._skip_decisions) >= self.MIN_SKIP_SAMPLES:
                    # Predict what the skip rate would be if we skip this frame (use cached sum)
                    # When deque is at maxlen, adding an element removes the oldest, so length stays constant
                    deque_len = len(self._skip_decisions)
                    is_at_capacity = deque_len >= self.SKIP_RATE_WINDOW
                    
                    if is_at_capacity:
                        # At capacity: new skip replaces oldest decision
                        # Conservative assumption: oldest was 0 (no skip)
                        # This may slightly overestimate skip rate, but that's safer for skip cap
                        future_skip_rate = (skip_sum + 1) / deque_len
                    else:
                        # Not at capacity: new skip increases length
                        future_skip_rate = (skip_sum + 1) / (deque_len + 1)
                    
                    skip_rate_cap_exceeded = future_skip_rate > self.SKIP_RATE_CAP
                
                # Legacy adaptive skip decision
                legacy_should_skip = adaptive_skip_conditions_met and not skip_rate_cap_exceeded
                
                # Smart pattern-based skip decision (in degraded mode)
                smart_should_skip, smart_skip_reason = self._should_smart_skip_frame(
                    queue_utilization, 
                    self.monitor.active_events
                )
                
                # Final skip decision: use smart skip if enabled and in degraded mode, otherwise legacy
                if in_degraded_mode and tracking_config.degraded_mode_smart_skip_enabled:
                    should_skip = smart_should_skip
                    skip_reason = smart_skip_reason
                else:
                    should_skip = legacy_should_skip
                    skip_reason = "critical_queue" if critical_queue_exceeded else ("proactive_bp" if proactive_backpressure else "adaptive")
                
                # Track skip decision for rate calculation
                self._skip_decisions.append(1 if should_skip else 0)
                
                if should_skip:
                    self._frames_skipped += 1
                    
                    # Track smart skip statistics
                    if smart_should_skip and in_degraded_mode:
                        self._frames_skipped_by_pattern += 1
                    
                    if self._frames_skipped % 10 == 0:
                        # Improved structured logging for adaptive skip
                        structured_logger.queue_backpressure(
                            queue_name='input_queue',
                            utilization=queue_utilization,
                            drops=self.input_queue_drops,
                            action='smart_skip' if smart_should_skip else 'adaptive_skip',
                            avg_detection_time_ms=avg_detection_time,
                            frames_skipped=self._frames_skipped
                        )
                        logger.warning(
                            f"[{'SmartSkip' if smart_should_skip else 'AdaptiveSkip'}] Frame skipped ({skip_reason}): "
                            f"queue={queue_utilization:.1%}, avg_detect={avg_detection_time:.1f}ms "
                            f"(threshold={self.MAX_DETECTION_TIME_MS:.1f}ms), "
                            f"skip_rate={current_skip_rate:.1%}, total_skipped={self._frames_skipped}"
                        )
                    continue
                
                # Log when skip cap prevents skipping
                if adaptive_skip_conditions_met and skip_rate_cap_exceeded:
                    self._skip_cap_blocks += 1
                    # Log on first occurrence and then every Nth (1, 6, 11, 16, ...)
                    # This prevents flooding while still showing the pattern
                    if self._skip_cap_blocks % self.SKIP_CAP_LOG_FREQUENCY == 1:
                        logger.warning(
                            f"[SkipCapBlock] Skip rate cap preventing frame skip: "
                            f"current_rate={current_skip_rate:.1%}, cap={self.SKIP_RATE_CAP:.1%}, "
                            f"queue={queue_utilization:.1%}, avg_detect={avg_detection_time:.1f}ms, "
                            f"blocks={self._skip_cap_blocks}"
                        )
                        structured_logger.pipeline_metric(
                            component="BagCounterApp",
                            metric_name="skip_cap_block",
                            metric_value=self._skip_cap_blocks,
                            context={
                                "current_skip_rate": current_skip_rate,
                                "skip_rate_cap": self.SKIP_RATE_CAP,
                                "queue_utilization": queue_utilization,
                                "avg_detection_time_ms": avg_detection_time
                            }
                        )

                # 1. Run Detector (V4: With batch inference support)
                detect_start = time.perf_counter()
                
                # V4 Phase 2: Batch inference
                if self.batch_inference_enabled:
                    self._frame_batch.append(frame)
                    self._batch_frame_data.append((frame, frame_count))
                    
                    if self._batch_start_time is None:
                        self._batch_start_time = time.perf_counter()
                    
                    # Check if batch is ready (full or timeout)
                    batch_elapsed_ms = (time.perf_counter() - self._batch_start_time) * 1000
                    batch_ready = (
                        len(self._frame_batch) >= tracking_config.detection_batch_size or
                        batch_elapsed_ms >= tracking_config.detection_batch_timeout_ms
                    )
                    
                    if not batch_ready:
                        # Batch not ready yet, continue accumulating
                        continue
                    
                    # Process batch
                    try:
                        detections_batch = self.detector.predict_batch(self._frame_batch)
                    except Exception as e:
                        # Fallback to single-frame processing on error
                        logger.warning(f"[LogicThread] Batch inference failed: {e}, falling back to single-frame")
                        detections_batch = [self.detector.predict(f) for f in self._frame_batch]
                    
                    detect_end = time.perf_counter()
                    detect_time = (detect_end - detect_start) * 1000
                    
                    # Process each frame in the batch
                    # Process each frame in the batch
                    for i, (batch_frame, batch_frame_count) in enumerate(self._batch_frame_data):
                        detections = detections_batch[i]
                        frame_detect_time = detect_time / len(self._frame_batch)  # Approximate per-frame time

                        # V3: Track detection time for adaptive skipping
                        self._recent_detection_times.append(frame_detect_time)

                        # Extract detections using _convert_detections helper (handles BpuResultWrapper)
                        current_frame_detections = self._convert_detections(detections)
                        
                        # V4 Phase 1: Enqueue detection result or process inline
                        if self.detection_queue_enabled:
                            # Enqueue detection result for monitor thread
                            detection_result = (current_frame_detections, batch_frame_count, batch_frame, frame_detect_time)
                            try:
                                self.detection_queue.put_nowait(detection_result)
                            except queue.Full:
                                # Queue full - drop oldest and try again
                                try:
                                    dropped = self.detection_queue.get_nowait()
                                    with self.stats_lock:
                                        self.detection_queue_drops += 1
                                    self.detection_queue.put_nowait(detection_result)
                                    logger.warning(f"[LogicThread] Detection queue full, dropped frame {dropped[1]}")
                                except (queue.Empty, queue.Full):
                                    with self.stats_lock:
                                        self.detection_queue_drops += 1
                                    logger.warning(f"[LogicThread] Failed to enqueue detection result for frame {batch_frame_count}")
                        else:
                            # Legacy: Process monitor inline
                            pipeline_metrics.record_detection(
                                current_frame_detections, 
                                frame_detect_time,
                                self.monitor.open_id,
                                self.monitor.closed_id
                            )
                            
                            monitor_start = time.perf_counter()
                            ready_events = self.monitor.update(current_frame_detections,
                                                               {"frame_count": batch_frame_count, "frame": batch_frame})
                            monitor_end = time.perf_counter()
                            monitor_time = (monitor_end - monitor_start) * 1000
                            
                            # Queue ready events for classification
                            if ready_events:
                                for event_id, candidates, event_box, event_stats in ready_events:
                                    should_save_snapshot = self.is_recording and not (
                                        in_degraded_mode and tracking_config.degraded_mode_disable_roi_saving
                                    )
                                    
                                    if should_save_snapshot:
                                        det_copy = [{"box": d["box"].copy(), "class_id": d["class_id"], "conf": float(d.get("conf", 0))} for d in current_frame_detections]
                                        try:
                                            event_box_copy = event_box.copy() if hasattr(event_box, 'copy') else list(event_box)
                                        except (TypeError, AttributeError):
                                            event_box_copy = event_box
                                        context = {
                                            "frame": batch_frame.copy(),
                                            "detections": det_copy,
                                            "event_box": event_box_copy,
                                            "event_stats": event_stats,
                                            "frame_id": batch_frame_count,
                                            "timestamp": time.time(),
                                        }
                                    else:
                                        context = {
                                            "frame": None,
                                            "detections": [],
                                            "event_box": event_box,
                                            "event_stats": event_stats,
                                            "frame_id": batch_frame_count,
                                            "timestamp": time.time(),
                                        }
                                    
                                    self._enqueue_classification(event_id, candidates, context)
                    
                    # Reset batch
                    self._frame_batch = []
                    self._batch_start_time = None
                    self._batch_frame_data = []
                    
                else:
                    # V3: Single-frame detection (legacy)
                    detections = self.detector.predict(frame)
                    detect_end = time.perf_counter()
                    detect_time = (detect_end - detect_start) * 1000
                    
                    # V3: Track detection time for adaptive skipping
                    self._recent_detection_times.append(detect_time)

                    current_frame_detections = []

                    if len(detections) > 0 and hasattr(detections[0], "boxes") and len(detections[0].boxes) > 0:
                        # Phase 1 Optimization: Vectorized detection extraction (2-3x faster)
                        boxes = detections[0].boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        confidences = boxes.conf.cpu().numpy()

                        # Use list comprehension for vectorized creation (faster than append loop)
                        current_frame_detections = [
                            {"box": xyxy[i], "class_id": cls_ids[i], "conf": confidences[i]}
                            for i in range(len(cls_ids))
                        ]
                    
                    # Degraded mode: skip frames with no detections and no active events
                    if (in_degraded_mode and 
                        tracking_config.degraded_mode_skip_low_detection_frames and
                        len(current_frame_detections) == 0 and
                        len(self.monitor.active_events) == 0):
                        # Skip this frame to save processing time
                        continue
                    
                    # V4 Phase 1: Enqueue detection result or process inline
                    if self.detection_queue_enabled:
                        # Enqueue detection result for monitor thread
                        detection_result = (current_frame_detections, frame_count, frame, detect_time)
                        try:
                            self.detection_queue.put_nowait(detection_result)
                        except queue.Full:
                            # Queue full - drop oldest and try again
                            try:
                                dropped = self.detection_queue.get_nowait()
                                with self.stats_lock:
                                    self.detection_queue_drops += 1
                                self.detection_queue.put_nowait(detection_result)
                                logger.warning(f"[LogicThread] Detection queue full, dropped frame {dropped[1]}")
                            except (queue.Empty, queue.Full):
                                with self.stats_lock:
                                    self.detection_queue_drops += 1
                                logger.warning(f"[LogicThread] Failed to enqueue detection result for frame {frame_count}")
                    else:
                        # Legacy: Process monitor inline
                        # Record detection metrics
                        pipeline_metrics.record_detection(
                            current_frame_detections, 
                            detect_time,
                            self.monitor.open_id,
                            self.monitor.closed_id
                        )

                        # 2. Update Monitor
                        monitor_start = time.perf_counter()
                        ready_events = self.monitor.update(current_frame_detections,
                                                           {"frame_count": frame_count, "frame": frame})
                        monitor_end = time.perf_counter()
                        monitor_time = (monitor_end - monitor_start) * 1000
                        
                        # Track frames processed in degraded mode for statistics
                        if in_degraded_mode:
                            self._frames_processed_in_degraded += 1
                        
                        # Update event frame counts for smart skip logic
                        if tracking_config.degraded_mode_smart_skip_enabled:
                            # Update counts for active events
                            current_event_ids = set()
                            for event in self.monitor.active_events:
                                event_id = getattr(event, 'event_id', None) or getattr(event, 'id', None)
                                if event_id is not None:
                                    current_event_ids.add(event_id)
                                    self._event_frame_counts[event_id] = self._event_frame_counts.get(event_id, 0) + 1
                            
                            # Clean up counts for events that are no longer active
                            events_to_remove = [eid for eid in self._event_frame_counts.keys() if eid not in current_event_ids]
                            for eid in events_to_remove:
                                del self._event_frame_counts[eid]

                        # 3. V3: Queue ready events for async classification (non-blocking)
                        enqueue_time = 0.0
                        if ready_events:
                            enqueue_start = time.perf_counter()
                            for event_id, candidates, event_box, event_stats in ready_events:
                                # Determine if we should save snapshot based on recording flag and degraded mode
                                should_save_snapshot = self.is_recording and not (
                                    in_degraded_mode and tracking_config.degraded_mode_disable_roi_saving
                                )
                                
                                # V3: Only copy frame if snapshot saving is enabled
                                if should_save_snapshot:
                                    det_copy = []
                                    for d in current_frame_detections:
                                        det_copy.append({
                                            "box": d["box"].copy(),
                                            "class_id": d["class_id"],
                                            "conf": float(d.get("conf", 0)),
                                        })
                                    # V3: Safe copy of event_box - handle numpy arrays and lists
                                    try:
                                        event_box_copy = event_box.copy() if hasattr(event_box, 'copy') else list(event_box)
                                    except (TypeError, AttributeError):
                                        event_box_copy = event_box  # Fallback to reference if copy fails
                                    context = {
                                        "frame": frame.copy(),  # Only copy when needed
                                        "detections": det_copy,
                                        "event_box": event_box_copy,
                                        "event_stats": event_stats,
                                        "frame_id": frame_count,
                                        "timestamp": time.time(),
                                    }
                                else:
                                    # V3: Lightweight context when recording is off or in degraded mode
                                    context = {
                                        "frame": None,
                                        "detections": [],
                                        "event_box": event_box,
                                        "event_stats": event_stats,
                                        "frame_id": frame_count,
                                        "timestamp": time.time(),
                                    }
                                
                                # V3: Non-blocking enqueue
                                self._enqueue_classification(event_id, candidates, context)
                            
                            enqueue_time = (time.perf_counter() - enqueue_start) * 1000

                        # 4. Publishing logic
                        publish_time = 0.0
                        
                        # Phase 1 Optimization: Visualization decimation
                        self._visualization_counter += 1
                        should_visualize_this_frame = (
                            self.is_publishing and 
                            self._visualization_counter % self.VISUALIZATION_DECIMATION == 0 and
                            not (in_degraded_mode and tracking_config.degraded_mode_disable_visualization)
                        )
                        
                        if should_visualize_this_frame:
                            publish_start = time.perf_counter()

                            # V3 Performance: Resize BEFORE visualization (process at 720p throughout)
                            # This reduces the amount of pixels to process during visualization
                            annotated_frame = cv2.resize(frame, (1280, 720))
                            
                            frame_mid = time.perf_counter()
                            mid_time = (frame_mid - frame_start) * 1000
                            fps_display = 1000 / mid_time if mid_time > 0 else 0

                            self.visualizer.render_all(
                                annotated_frame,
                                [],
                                self.monitor.active_events,
                                counts=self.ui_counts,
                                fps=fps_display,
                            )

                            # V3 Performance: No need to resize again - already at 1280x720
                            self.ipc_publisher.publish(annotated_frame)

                            publish_end = time.perf_counter()
                            publish_time = (publish_end - publish_start) * 1000

                        # 5. Timing logs and pipeline metrics
                        frame_end = time.perf_counter()
                        total_time = (frame_end - frame_start) * 1000
                        fps = 1000 / total_time if total_time > 0 else 0
                        
                        # V3: Track frame processing time
                        self._recent_frame_times.append(total_time)

                        if frame_count % TIMING_LOG_INTERVAL == 0 or enqueue_time > 0:
                            timing_msg = (
                                f"[Frame {frame_count}] Total: {total_time:.1f}ms | "
                                f"Detect: {detect_time:.1f}ms | "
                                f"Monitor: {monitor_time:.1f}ms"
                            )
                            if enqueue_time > 0:
                                timing_msg += f" | Queue: {enqueue_time:.1f}ms"
                            if publish_time > 0:
                                timing_msg += f" | Publish: {publish_time:.1f}ms"
                            timing_msg += f" | FPS: {fps:.1f}"
                            
                            # V3: Add queue status
                            input_q_size = self.input_queue.qsize()
                            class_q_size = self.classification_queue.qsize()
                            timing_msg += f" | InputQ: {input_q_size}/{self.INPUT_QUEUE_SIZE}"
                            timing_msg += f" | ClassQ: {class_q_size}/{self.CLASSIFICATION_QUEUE_SIZE}"
                            
                            logger.info(timing_msg)
                            
                            # Structured logging for frame processing
                            structured_logger.frame_processed(
                                frame_id=frame_count,
                                detection_time_ms=detect_time,
                                monitor_time_ms=monitor_time,
                                total_time_ms=total_time,
                                detections_count=len(current_frame_detections),
                                events_ready=len(ready_events) if ready_events else 0,
                                queue_sizes={
                                    'input': input_q_size,
                                    'classification': class_q_size
                                },
                                fps=fps
                            )

                        # Log pipeline metrics periodically
                        pipeline_metrics.maybe_log_summary()
                        
                        # Log system status periodically (every 15 minutes)
                        current_time = time.perf_counter()
                        if current_time - self._last_system_status_log >= self.SYSTEM_STATUS_LOG_INTERVAL:
                            self._log_system_status()
                            self._last_system_status_log = current_time

                
                # V4 Phase 1: Skip visualization/timing when using detection queue (handled by monitor thread)
                if not self.detection_queue_enabled:
                    pass  # All processing done above in legacy path

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                
                # Structured error logging
                structured_logger.pipeline_error(
                    component='LogicThread',
                    operation='frame_processing',
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=[frame_count],
                    context={
                        'detections_count': len(current_frame_detections) if 'current_frame_detections' in locals() else 0,
                        'active_events': len(self.monitor.active_events) if hasattr(self, 'monitor') else 0,
                        'input_queue_size': self.input_queue.qsize(),
                        'classification_queue_size': self.classification_queue.qsize()
                    },
                    traceback=error_trace
                )

        logger.info("[LogicThread] Stopped")

    def run(self):
        logger.info("[BagCounterApp] Starting main loop (V3 with async classification)")
        self.is_running = True
        
        # Phase 2: Start multiple classification worker threads
        self._classification_running = True
        self._classification_threads = []
        for i in range(self.CLASSIFICATION_WORKERS):
            thread = threading.Thread(
                target=self._classification_thread_loop, 
                daemon=True,
                name=f"ClassificationThread-{i}"
            )
            thread.start()
            self._classification_threads.append(thread)
        logger.info(f"[BagCounterApp] Started {self.CLASSIFICATION_WORKERS} classification worker threads")
        
        # V4 Phase 1: Start monitor thread if detection queue is enabled
        if self.detection_queue_enabled and self.detection_queue is not None:
            self._monitor_running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_thread_loop,
                daemon=True,
                name="MonitorThread"
            )
            self._monitor_thread.start()
            logger.info("[BagCounterApp] Monitor thread started (V4 Phase 1: Detection queue decoupling)")

        self.config_watcher.start()
        logger.debug("[BagCounterApp] Config watcher started")
        
        # Accuracy Mode: Publish READY signal after initialization
        if self._accuracy_mode:
            # Give ROS2 a moment to establish connections
            time.sleep(0.5)
            self._publish_processing_ready()

        if self.testing_mode:
            logger.info("[BagCounterApp] TESTING MODE: Inline processing, no input queue, zero frame drops.")
            frame_count = 0
            try:
                for frame, latencyMs in self.frame_source.frames():
                    frame_count += 1
                    self._process_frame_inline(frame, frame_count)
            except KeyboardInterrupt:
                logger.info("[BagCounterApp] Interrupted by user in testing mode")
            except Exception as e:
                logger.error(f"[BagCounterApp] Error in main loop: {e}")
                import traceback
                logger.debug(f"[BagCounterApp] Traceback:\n{traceback.format_exc()}")
            finally:
                self.shutdown_procedure(None)
        else:
            logic_thread = threading.Thread(target=self._logic_thread_loop, daemon=True, name="LogicThread")
            logic_thread.start()
            FRAME_STATS_INTERVAL = 100
            TIMING_EPSILON = 1e-6
            frame_count = 0
            last_frame_time = None
            frame_interval_sum = 0.0
            frame_interval_count = 0
            last_queue_stats_time = time.perf_counter()
            try:
                for frame_data in self.frame_source.frames():
                    frame_count += 1
                    
                    # SINGLE SOURCE OF TRUTH: Extract frame and spool_frame_index from tuple
                    # In accuracy mode, spool_frame_index travels WITH the frame data
                    # In normal mode, tuple is (frame, latency) - backward compatible
                    if len(frame_data) == 3:
                        frame, latencyMs, spool_frame_index = frame_data
                    else:
                        frame, latencyMs = frame_data
                        spool_frame_index = None
                    
                    # Accuracy Mode: Publish ACK IMMEDIATELY when frame is consumed
                    # CRITICAL: Use spool_frame_index that traveled WITH this specific frame
                    # This ensures perfect correlation - no separate state queries needed
                    if getattr(self, "_accuracy_mode", False) and spool_frame_index is not None:
                        self._publish_processing_ack(spool_frame_index)
                    
                    current_time = time.perf_counter()
                    if last_frame_time is not None:
                        frame_interval = current_time - last_frame_time
                        frame_interval_sum += frame_interval
                        frame_interval_count += 1
                    last_frame_time = current_time
                    if frame_count % FRAME_STATS_INTERVAL == 0 and frame_interval_count > 0:
                        avg_interval = frame_interval_sum / frame_interval_count
                        if avg_interval > TIMING_EPSILON:
                            acquisition_fps = 1.0 / avg_interval
                            logger.info(
                                f"[BagCounterApp] Frame acquisition stats: "
                                f"frames={frame_count}, avg_interval={avg_interval * 1000:.1f}ms, "
                                f"acquisition_fps={acquisition_fps:.1f}"
                            )
                        else:
                            logger.warning(
                                f"[BagCounterApp] Invalid frame timing detected: "
                                f"frames={frame_count}, avg_interval={avg_interval * 1000:.6f}ms "
                                f"(below {TIMING_EPSILON * 1000:.6f}ms threshold) - skipping FPS calculation"
                            )
                        frame_interval_sum = 0.0
                        frame_interval_count = 0
                    try:
                        self.input_queue.put_nowait(frame)
                    except queue.Full:
                        frame_dropped = False
                        try:
                            self.input_queue.get_nowait()
                            frame_dropped = True
                            try:
                                self.input_queue.put_nowait(frame)
                            except queue.Full:
                                frame_dropped = True
                                logger.debug(
                                    f"[BagCounterApp] Frame {frame_count} dropped: "
                                    "queue refilled immediately after clearing"
                                )
                            if frame_dropped:
                                with self.stats_lock:
                                    self.input_queue_drops += 1
                                    drops = self.input_queue_drops
                                logger.warning(
                                    f"[BagCounterApp] Dropped old frame (input queue full, "
                                    f"total drops: {drops})"
                                )
                        except queue.Empty:
                            try:
                                self.input_queue.put_nowait(frame)
                            except queue.Full:
                                with self.stats_lock:
                                    self.input_queue_drops += 1
                                    drops = self.input_queue_drops
                                logger.warning(
                                    f"[BagCounterApp] Frame {frame_count} dropped: "
                                    f"queue refilled by another thread (total drops: {drops})"
                                )
                    current_time = time.perf_counter()
                    if current_time - last_queue_stats_time >= self.STATS_LOG_INTERVAL:
                        # Phase 2 Optimization: Collect stats efficiently
                        self._log_queue_stats()
                        last_queue_stats_time = current_time
            except KeyboardInterrupt:
                logger.info("[BagCounterApp] Interrupted by user")
            except Exception as e:
                logger.error(f"[BagCounterApp] Error in main loop: {e}")
                import traceback
                logger.debug(f"[BagCounterApp] Traceback:\n{traceback.format_exc()}")
            finally:
                self.shutdown_procedure(logic_thread)

    def shutdown_procedure(self, logic_thread=None):
        logger.info(f"[BagCounterApp] Shutting down...")
        self.is_running = False
        THREAD_SHUTDOWN_TIMEOUT = 3.0
        
        # Phase 2: Shutdown all classification worker threads
        self._classification_running = False
        for i, thread in enumerate(self._classification_threads):
            if thread and thread.is_alive():
                thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
                if thread.is_alive():
                    logger.warning(
                        f"[BagCounterApp] Classification thread {i} did not stop within "
                        f"{THREAD_SHUTDOWN_TIMEOUT}s timeout"
                    )
                else:
                    logger.debug(f"[BagCounterApp] Classification thread {i} joined")
        
        # V4 Phase 1: Shutdown monitor thread if detection queue is enabled
        if self.detection_queue_enabled and hasattr(self, '_monitor_thread') and self._monitor_thread is not None:
            self._monitor_running = False
            if self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
                if self._monitor_thread.is_alive():
                    logger.warning(
                        f"[BagCounterApp] Monitor thread did not stop within "
                        f"{THREAD_SHUTDOWN_TIMEOUT}s timeout"
                    )
                else:
                    logger.debug("[BagCounterApp] Monitor thread joined")
        
        self.frame_source.cleanup()
        logger.debug("[BagCounterApp] Frame source cleaned up")
        self.config_watcher.stop()
        logger.debug("[BagCounterApp] Config watcher stopped")
        if IS_RDK and self.ros_executor is not None:
            self.ros_executor.remove_node(self.ipc_publisher)
            if isinstance(self.frame_source, Node):
                self.ros_executor.remove_node(self.frame_source)
            if getattr(self, "_accuracy_mode", False) and getattr(self, "_ack_publisher_node", None) is not None:
                self.ros_executor.remove_node(self._ack_publisher_node)
                self._ack_publisher_node.destroy_node()
        self.ipc_publisher.close_node()
        shutdown_ros2_context()
        if IS_RDK:
            logger.debug("[BagCounterApp] ROS 2 context shutdown")
        if self.ros_thread.is_alive():
            self.ros_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
            if self.ros_thread.is_alive():
                logger.warning(
                    f"[BagCounterApp] ROS thread did not stop within "
                    f"{THREAD_SHUTDOWN_TIMEOUT}s timeout"
                )
            else:
                logger.debug("[BagCounterApp] ROS thread joined")
        if logic_thread is not None and logic_thread.is_alive():
            logic_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
            if logic_thread.is_alive():
                logger.warning(
                    f"[BagCounterApp] Logic thread did not stop within "
                    f"{THREAD_SHUTDOWN_TIMEOUT}s timeout"
                )
            else:
                logger.debug("[BagCounterApp] Logic thread joined")
        self.db.close()
        logger.debug("[BagCounterApp] Database connection closed")
        with self.stats_lock:
            input_drops = self.input_queue_drops if self.input_queue is not None else 0
            class_drops = self.classification_queue_drops
            detection_drops = self.detection_queue_drops if self.detection_queue_enabled else 0
        
        # Calculate final skip rate
        final_skip_rate = 0.0
        if len(self._skip_decisions) > 0:
            final_skip_rate = (sum(self._skip_decisions) / len(self._skip_decisions)) * 100
        
        # Calculate smart skip statistics
        smart_skip_stats = ""
        if tracking_config.degraded_mode_smart_skip_enabled and self._frames_processed_in_degraded > 0:
            total_degraded_frames = self._frames_processed_in_degraded + self._frames_skipped_by_pattern
            pattern_skip_rate = (self._frames_skipped_by_pattern / total_degraded_frames) * 100
            smart_skip_stats = (
                f", smart_skip_frames={self._frames_skipped_by_pattern}, "
                f"smart_skip_rate={pattern_skip_rate:.2f}%, "
                f"frames_in_degraded={total_degraded_frames}"
            )
        
        logger.info(
            f"[BagCounterApp] Final Stats: "
            f"input_drops={input_drops}, "
            f"detection_drops={detection_drops}, "
            f"classification_drops={class_drops}, "
            f"frames_skipped={self._frames_skipped}, skip_rate={final_skip_rate:.2f}%, "
            f"skip_cap_blocks={self._skip_cap_blocks}"
            f"{smart_skip_stats}"
        )
        logger.info("[BagCounterApp] Shutdown complete")