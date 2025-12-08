import os
import cv2
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Any

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

from src.logging.ConfigWatcher import ConfigWatcher
from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK

# Import the ROS 2 helper functions from IPC.py (these handle platform detection internally)
from src.counting.IPC import ExecutorThread, init_ros2_context, shutdown_ros2_context
from src.counting.FramePublisherNode import FramePublisher

# Conditional ROS2 Node import
if IS_RDK:
    from rclpy.node import Node
else:
    # Stub Node class for isinstance checks on non-RDK platforms
    class Node:
        pass


class BagCounterApp:
    # Queue configuration constants
    INPUT_QUEUE_SIZE = 100  # Buffer size for input frames (100 frames @ 25fps = ~4 seconds)
    RECORDING_QUEUE_SIZE = 100  # Buffer size for recording frames (100 frames @ 25fps = ~4 seconds)
    QUEUE_WARNING_THRESHOLD = 80  # Percentage threshold for queue utilization warnings
    STATS_LOG_INTERVAL = 5.0  # Log statistics every N seconds
    MIN_RECORDING_FPS = 1.0  # Minimum valid recording FPS to prevent division issues
    
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

        # Recording state
        self.is_recording = db.get_config_value(constants.is_recording_key) == "1"
        self.video_writer = None
        # Use DB-configured path if present, else fall back to config
        self.recording_dir = db.get_config_value(constants.recording_dir) or config.recording_dir
        # Segment length (seconds). Default 600 (10 minutes) if not provided in config/env.

        self.recording_segment_seconds = db.get_config_value(constants.recording_seconds)
        try:
            self.recording_segment_seconds = int(self.recording_segment_seconds)
        except Exception:
            default_segment_seconds = 600
            logger.error(
                f"[BagCounterApp] Invalid RECORDING_SEGMENT_SECONDS={self.recording_segment_seconds}; "
                f"falling back to {default_segment_seconds}"
            )
            self.recording_segment_seconds = default_segment_seconds

        # Recording FPS - configurable via RECORDING_FPS environment variable, default 30.0
        recording_fps_str = db.get_config_value(constants.recording_fps)
        try:
            self.recording_fps = float(recording_fps_str)
            # Validate FPS is positive and meets minimum threshold
            if self.recording_fps <= 0:
                logger.warning(
                    f"[BagCounterApp] Recording FPS {self.recording_fps} is not positive, "
                    f"using minimum value {self.MIN_RECORDING_FPS}"
                )
                self.recording_fps = self.MIN_RECORDING_FPS
            elif self.recording_fps < self.MIN_RECORDING_FPS:
                logger.warning(
                    f"[BagCounterApp] Recording FPS {self.recording_fps} is below minimum {self.MIN_RECORDING_FPS}, "
                    f"using minimum value"
                )
                self.recording_fps = self.MIN_RECORDING_FPS
        except (ValueError, TypeError):
            fallback_fps = 30.0
            logger.warning(
                f"[BagCounterApp] Invalid RECORDING_FPS value '{recording_fps_str}', "
                f"using default {fallback_fps}"
            )
            self.recording_fps = fallback_fps
        
        logger.info(f"[BagCounterApp] Using RECORDING_FPS: {self.recording_fps}")

        self.segment_start_time = None
        self.segment_counter = 0

        logger.info(f"[BagCounterApp] Video Recording: {'ENABLED' if self.is_recording else 'DISABLED'}")
        logger.info(f"[BagCounterApp] Recording directory: {self.recording_dir}")
        logger.info(f"[BagCounterApp] Recording segment length: {self.recording_segment_seconds}s")
        logger.info(f"[BagCounterApp] Recording FPS: {self.recording_fps}")

        # Input queue size set to 100 frames for better buffering with 25 fps RTSP stream
        self.input_queue = queue.Queue(maxsize=self.INPUT_QUEUE_SIZE)
        # Queue for asynchronous video recording - increased for better margin
        self.recording_queue = queue.Queue(maxsize=self.RECORDING_QUEUE_SIZE)
        self.recording_thread = None
        # Lock for video_writer access synchronization
        self.video_writer_lock = threading.Lock()
        
        # Queue monitoring statistics (thread-safe counters)
        self.stats_lock = threading.Lock()
        self.input_queue_drops = 0
        self.recording_queue_drops = 0
        self.last_queue_stats_log_time = time.perf_counter()
        
        # Recording frame rate limiting
        self.last_recording_frame_time = None
        # Safe to divide since we validated recording_fps >= MIN_RECORDING_FPS above
        self.recording_frame_interval = 1.0 / self.recording_fps

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

        self.monitor = BagStateMonitor(open_id, closed_id)
        self.visualizer = Visualizer(names)
        self.classifier_service.register_callback(self.on_classification_result)
        self.ui_counts = {}

        # --- IPC SETUP (ROS 2 - Executor Pattern) ---
        from src.utils.platform import IS_RDK

        logger.debug("[BagCounterApp] Initializing ROS 2 context...")
        self.ros_executor = init_ros2_context()

        self.is_publishing = db.get_config_value(constants.show_ui_screen_key) == "1"
        logger.info(f"[BagCounterApp] IPC Publishing: {'ENABLED' if self.is_publishing else 'DISABLED'}")

        self.ipc_publisher = FramePublisher(publish_rate_hz=30.0)


        if IS_RDK and self.ros_executor is not None:
            self.ros_executor.add_node(self.ipc_publisher)

        if is_development:
            self.frame_source = FrameSourceFactory.create("opencv", source=video_path, target_fps=self.recording_fps)
            logger.info(f"[BagCounterApp] Development mode: reading from {video_path}")
        else:
            if IS_RDK:
                os.environ["HOME"] = "/home/sunrise"
                self.frame_source = FrameSourceFactory.create("ros2", target_fps=30.0)
                logger.info("[BagCounterApp] Production mode: reading from ROS 2 stream")
            else:
                # On Windows, fall back to OpenCV even in production mode
                self.frame_source = FrameSourceFactory.create("opencv", source=video_path, target_fps=30.0)
                logger.info(f"[BagCounterApp] Windows mode: reading from {video_path}")

        if IS_RDK and self.ros_executor is not None and isinstance(self.frame_source, Node):
            self.ros_executor.add_node(self.frame_source)
            logger.debug("[BagCounterApp] FrameSource added to ROS 2 executor")

        self.ros_thread = ExecutorThread(self.ros_executor)
        self.ros_thread.start()
        if IS_RDK:
            logger.debug("[BagCounterApp] ROS 2 executor thread started")

        logger.info("[BagCounterApp] Initialization complete")

    def on_show_ui_changed(self, new_value):
        if new_value == "1":
            self.is_publishing = True
            logger.info("[BagCounterApp] IPC Publishing ENABLED")
        else:
            self.is_publishing = False
            logger.info("[BagCounterApp] IPC Publishing DISABLED")

    def on_is_recording_changed(self, new_value):
        if new_value == "1":
            self.is_recording = True
            # Reset recording frame timing for new recording session
            self.last_recording_frame_time = None
            logger.info("[BagCounterApp] Video Recording ENABLED")
        else:
            self.is_recording = False
            logger.info("[BagCounterApp] Video Recording DISABLED")
            # Reset segment metadata so next start begins fresh
            self.segment_counter = 0
            self.segment_start_time = None
            self.last_recording_frame_time = None

    def _open_video_writer(self, frame):
        """Open a new video writer and return (writer, filename) or (None, None) on failure."""
        try:
            os.makedirs(self.recording_dir, exist_ok=True)
            if not os.access(self.recording_dir, os.W_OK):
                logger.error(f"[Recording] Directory not writable: {self.recording_dir}")
                return None, None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            video_filename = os.path.join(self.recording_dir, f"{timestamp}_p{self.segment_counter:03d}.mp4")

            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_filename, fourcc, self.recording_fps, (width, height))

            if writer.isOpened():
                self.segment_start_time = time.perf_counter()
                self.segment_counter += 1
                logger.info(f"[Recording] Opened video writer: {video_filename} at {self.recording_fps} FPS")
                return writer, video_filename
            else:
                logger.error(f"[Recording] Failed to open video writer: {video_filename}")
                return None, None
        except Exception as e:
            logger.error(f"[Recording] Error starting recording: {e}")
            return None, None

    def _recording_thread_loop(self):
        """Background thread that handles video recording asynchronously."""
        logger.info("[RecordingThread] Started")
        
        frame_write_count = 0
        last_stats_log = time.perf_counter()
        
        while self.is_running:
            try:
                # Get frame from recording queue with timeout
                frame = self.recording_queue.get(timeout=1.0)
            except queue.Empty:
                if not self.is_running:
                    break
                continue
            except Exception as e:
                logger.error(f"[RecordingThread] Queue error: {e}")
                continue
            
            # Periodic queue stats logging
            current_time = time.perf_counter()
            if current_time - last_stats_log >= self.STATS_LOG_INTERVAL:
                queue_size = self.recording_queue.qsize()
                queue_utilization = (queue_size / self.RECORDING_QUEUE_SIZE) * 100
                
                # Thread-safe read of drop counter
                with self.stats_lock:
                    drops = self.recording_queue_drops
                
                logger.info(
                    f"[RecordingThread] Queue stats: size={queue_size}/{self.RECORDING_QUEUE_SIZE} "
                    f"({queue_utilization:.1f}% full), frames_written={frame_write_count}, "
                    f"drops={drops}"
                )
                last_stats_log = current_time
            
            try:
                with self.video_writer_lock:
                    # Start writer if needed
                    if self.video_writer is None:
                        self.video_writer, opened_filename = self._open_video_writer(frame)
                        if opened_filename:
                            logger.info(f"[Recording] Started recording to: {opened_filename}")
                    
                    # Rotate segment if duration exceeded
                    if (
                        self.video_writer is not None
                        and self.video_writer.isOpened()
                        and self.segment_start_time is not None
                        and (time.perf_counter() - self.segment_start_time) >= self.recording_segment_seconds
                    ):
                        try:
                            self.video_writer.release()
                            logger.info("[Recording] Closed segment (time limit reached)")
                        except Exception as e:
                            logger.error(f"[Recording] Error closing segment: {e}")
                        finally:
                            self.video_writer = None
                            self.segment_start_time = None
                        
                        # Open next segment immediately
                        self.video_writer, opened_filename = self._open_video_writer(frame)
                        if opened_filename:
                            logger.info(f"[Recording] Started new segment: {opened_filename}")
                    
                    # Write frame if writer is open
                    if self.video_writer is not None and self.video_writer.isOpened():
                        try:
                            self.video_writer.write(frame)
                            frame_write_count += 1
                        except Exception as e:
                            logger.error(f"[Recording] Error writing frame: {e}")
                        
            except Exception as e:
                logger.error(f"[RecordingThread] Error processing frame: {e}")
        
        logger.info("[RecordingThread] Stopped")

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        label = data["label"]
        phash = data["phash"]
        image_path = data["image_path"]
        conf = data.get("confidence", 1.0)
        candidates_count = data.get("candidates_evaluated", 1)

        logger.info(
            f"[BagCounterApp] Classification result: track={track_id}, "
            f"label={label}, conf={conf:.3f}"
        )
        logger.debug(
            f"[BagCounterApp] Result details: phash={phash}, "
            f"image_path={image_path}, candidates={candidates_count}"
        )

        bag_type_id = self.db.get_or_create_bag_type(label, phash, image_path)
        self.db.log_event(bag_type_id, track_id, conf)

        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1
        logger.info(f"[BagCounterApp] Count updated: {label} = {self.ui_counts[label]}")

    def _logic_thread_loop(self):
        logger.info("[LogicThread] Started")

        class SimpleTrack:
            def __init__(self, tid, box, cid):
                self.track_id = tid
                self.box = box
                self.class_id = cid

        # Configuration constants
        TIMING_LOG_INTERVAL = 30  # Log timing every N frames to reduce log spam

        frame_count = 0

        while self.is_running:
            try:
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                if not self.is_running:
                    break
                logger.debug("[LogicThread] Input queue empty, waiting...")
                continue
            except Exception as e:
                logger.error(f"[LogicThread] Input queue error: {e}")
                continue

            try:
                frame_count += 1

                # Frame timing metrics (using time.perf_counter for precision)
                frame_start = time.perf_counter()

                # 1. Run Detector
                detect_start = time.perf_counter()
                detections = self.detector.predict(frame)
                detect_end = time.perf_counter()
                detect_time = (detect_end - detect_start) * 1000  # Convert to ms

                current_frame_detections = []

                if len(detections) > 0 and hasattr(detections[0], "boxes") and len(detections[0].boxes) > 0:
                    xyxy = detections[0].boxes.xyxy.cpu().numpy()
                    cls_ids = detections[0].boxes.cls.cpu().numpy().astype(int)
                    confidences = detections[0].boxes.conf.cpu().numpy()

                    for i in range(len(cls_ids)):
                        current_frame_detections.append(
                            {"box": xyxy[i], "class_id": cls_ids[i], "conf": confidences[i]}
                        )

                    # Log detection confidence for debugging
                    logger.debug(f"[LogicThread] Frame {frame_count}: {len(current_frame_detections)} detections")

                    if len(current_frame_detections) > 0:
                        for det in current_frame_detections:
                            class_name = self.detector.class_names.get(det["class_id"], "Unknown")
                            logger.debug(
                                f"[RAW DETECTION] class={class_name} (id={det['class_id']}), "
                                f"conf={det['conf']:.3f}, box=[{det['box'][0]:.1f}, {det['box'][1]:.1f}, "
                                f"{det['box'][2]:.1f}, {det['box'][3]:.1f}]"
                            )
                else:
                    logger.debug(f"[LogicThread] Frame {frame_count}: No detections")

                # 2. Update Monitor
                monitor_start = time.perf_counter()
                ready_events = self.monitor.update(current_frame_detections, frame)
                monitor_end = time.perf_counter()
                monitor_time = (monitor_end - monitor_start) * 1000  # Convert to ms

                # 3. Process Ready Events
                classify_time = 0.0
                if ready_events:
                    logger.info(
                        f"[LogicThread] Frame {frame_count}: "
                        f"{len(ready_events)} events ready for classification"
                    )

                    classify_start = time.perf_counter()
                    for event_id, candidates in ready_events:
                        logger.debug(
                            f"[LogicThread] Sending event {event_id} to classifier "
                            f"({len(candidates)} candidates)"
                        )
                        self.classifier_service.process(event_id, candidates)
                    classify_end = time.perf_counter()
                    classify_time = (classify_end - classify_start) * 1000  # Convert to ms

                # --- 4. RECORDING LOGIC (Independent from publishing) ---
                record_time = 0.0

                # Enqueue frame for asynchronous recording at target FPS
                if self.is_recording:
                    record_start = time.perf_counter()
                    should_record_frame = False
                    
                    # Rate limit recording to target FPS using elapsed time tracking
                    # We update the reference to the current moment (not increment by interval) which:
                    # - Prevents cumulative drift from small timing errors
                    # - Allows recording to adapt if processing temporarily slows
                    # - Maintains average FPS close to target over time
                    if self.last_recording_frame_time is None:
                        # Record first frame immediately
                        should_record_frame = True
                        self.last_recording_frame_time = record_start
                    else:
                        # Check if enough time has elapsed since last recorded frame
                        time_since_last = record_start - self.last_recording_frame_time
                        if time_since_last >= self.recording_frame_interval:
                            should_record_frame = True
                            # Update reference to current time (not += interval) to prevent cumulative drift
                            self.last_recording_frame_time = record_start
                    
                    if should_record_frame:
                        # Non-blocking enqueue: drop frame if queue is full
                        try:
                            self.recording_queue.put_nowait(frame.copy())
                        except queue.Full:
                            with self.stats_lock:
                                self.recording_queue_drops += 1
                                drops = self.recording_queue_drops
                            logger.warning(
                                f"[Recording] Recording queue full, dropping frame "
                                f"(total drops: {drops})"
                            )
                    
                    record_end = time.perf_counter()
                    record_time = (record_end - record_start) * 1000  # Convert to ms
                else:
                    # Falling edge: Stop recording
                    # Clear the recording queue and let recording thread finish
                    with self.video_writer_lock:
                        if self.video_writer is not None:
                            # Clear queue to ensure recording thread processes remaining frames
                            while not self.recording_queue.empty():
                                try:
                                    self.recording_queue.get_nowait()
                                except queue.Empty:
                                    break
                        self.segment_counter = 0

                # --- 5. PUBLISHING LOGIC ---
                publish_time = 0.0

                if self.is_publishing:
                    publish_start = time.perf_counter()

                    annotated_frame = frame.copy()

                    # Calculate total time so far for FPS display
                    frame_mid = time.perf_counter()
                    mid_time = (frame_mid - frame_start) * 1000  # Convert to ms
                    fps_display = 1000 / mid_time if mid_time > 0 else 0

                    self.visualizer.render_all(
                        annotated_frame,
                        # raw detection dicts or tracked objects:
                        current_frame_detections,
                        # event objects with .id, .state, .box
                        self.monitor.active_events,
                        counts=self.ui_counts,
                        fps=fps_display,
                    )

                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    self.ipc_publisher.publish(annotated_frame)

                    publish_end = time.perf_counter()
                    publish_time = (publish_end - publish_start) * 1000  # Convert to ms

                # Calculate total frame time including all operations
                frame_end = time.perf_counter()
                total_time = (frame_end - frame_start) * 1000  # Convert to ms
                fps = 1000 / total_time if total_time > 0 else 0

                # Log frame timing breakdown (at configured interval or when classification occurs)
                if frame_count % TIMING_LOG_INTERVAL == 0 or classify_time > 0:
                    timing_msg = (
                        f"[Frame {frame_count}] Total: {total_time:.1f}ms | "
                        f"Detect: {detect_time:.1f}ms | "
                        f"Monitor: {monitor_time:.1f}ms"
                    )
                    if classify_time > 0:
                        timing_msg += f" | Classify: {classify_time:.1f}ms"
                    if record_time > 0:
                        timing_msg += f" | Record: {record_time:.1f}ms"
                    if publish_time > 0:
                        timing_msg += f" | Publish: {publish_time:.1f}ms"
                    timing_msg += f" | FPS: {fps:.1f}"

                    logger.info(timing_msg)

            except Exception as e:
                logger.error(f"[LogicThread] Error processing frame {frame_count}: {e}")
                import traceback

                logger.debug(f"[LogicThread] Traceback:\n{traceback.format_exc()}")

        logger.info("[LogicThread] Stopped")

    def run(self):
        logger.info("[BagCounterApp] Starting main loop")
        self.is_running = True

        logic_thread = threading.Thread(target=self._logic_thread_loop, daemon=True)
        logic_thread.start()

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_thread_loop, daemon=True)
        self.recording_thread.start()
        logger.debug("[BagCounterApp] Recording thread started")

        self.config_watcher.start()
        logger.debug("[BagCounterApp] Config watcher started")

        # Configuration constants for frame acquisition monitoring
        FRAME_STATS_INTERVAL = 100  # Log acquisition FPS every N frames
        TIMING_EPSILON = 1e-6  # 1 microsecond - minimum valid frame interval
        
        frame_count = 0
        last_frame_time = None
        frame_interval_sum = 0.0
        frame_interval_count = 0
        last_queue_stats_time = time.perf_counter()

        try:
            for frame, latencyMs in self.frame_source.frames():
                frame_count += 1
                current_time = time.perf_counter()
                
                # Track frame-to-frame interval for FPS calculation
                if last_frame_time is not None:
                    frame_interval = current_time - last_frame_time
                    frame_interval_sum += frame_interval
                    frame_interval_count += 1
                
                last_frame_time = current_time
                
                # Log frame acquisition FPS periodically
                if frame_count % FRAME_STATS_INTERVAL == 0 and frame_interval_count > 0:
                    avg_interval = frame_interval_sum / frame_interval_count
                    # Guard against invalid timing measurements
                    if avg_interval > TIMING_EPSILON:
                        acquisition_fps = 1.0 / avg_interval
                        logger.info(
                            f"[BagCounterApp] Frame acquisition stats: "
                            f"frames={frame_count}, avg_interval={avg_interval*1000:.1f}ms, "
                            f"acquisition_fps={acquisition_fps:.1f}"
                        )
                    else:
                        # Extremely unlikely, but handle invalid timing measurements
                        logger.warning(
                            f"[BagCounterApp] Invalid frame timing detected: "
                            f"frames={frame_count}, avg_interval={avg_interval*1000:.6f}ms "
                            f"(below {TIMING_EPSILON*1000:.6f}ms threshold) - skipping FPS calculation"
                        )
                    # Reset for next interval
                    frame_interval_sum = 0.0
                    frame_interval_count = 0

                # Use non-blocking put with leaky queue behavior to avoid race conditions
                try:
                    self.input_queue.put_nowait(frame)
                except queue.Full:
                    # Queue is full, drop oldest frame and try again (leaky queue behavior)
                    try:
                        self.input_queue.get_nowait()
                        # Successfully dropped oldest frame, now add new frame
                        try:
                            self.input_queue.put_nowait(frame)
                        except queue.Full:
                            # Extremely rare: queue filled again between get and put
                            logger.debug(
                                f"[BagCounterApp] Frame {frame_count} dropped: "
                                "queue refilled immediately after clearing"
                            )
                        # Increment drop counter
                        with self.stats_lock:
                            self.input_queue_drops += 1
                            drops = self.input_queue_drops
                        logger.warning(
                            f"[BagCounterApp] Dropped frame {frame_count} (input queue full, "
                            f"total drops: {drops})"
                        )
                    except queue.Empty:
                        # Queue was drained by another thread, retry putting the frame
                        try:
                            self.input_queue.put_nowait(frame)
                        except queue.Full:
                            # Still full after being drained, skip this frame
                            logger.debug(
                                f"[BagCounterApp] Frame {frame_count} dropped: "
                                "queue refilled by another thread"
                            )
                
                # Periodic queue statistics logging
                if current_time - last_queue_stats_time >= self.STATS_LOG_INTERVAL:
                    input_size = self.input_queue.qsize()
                    input_utilization = (input_size / self.INPUT_QUEUE_SIZE) * 100
                    recording_size = self.recording_queue.qsize()
                    recording_utilization = (recording_size / self.RECORDING_QUEUE_SIZE) * 100
                    
                    # Thread-safe read of drop counters
                    with self.stats_lock:
                        input_drops = self.input_queue_drops
                        recording_drops = self.recording_queue_drops
                    
                    logger.info(
                        f"[QueueStats] Input queue: {input_size}/{self.INPUT_QUEUE_SIZE} "
                        f"({input_utilization:.1f}% full, drops={input_drops}) | "
                        f"Recording queue: {recording_size}/{self.RECORDING_QUEUE_SIZE} "
                        f"({recording_utilization:.1f}% full, drops={recording_drops})"
                    )
                    
                    # Warning if queues are getting full
                    if input_utilization > self.QUEUE_WARNING_THRESHOLD:
                        logger.warning(
                            f"[QueueStats] Input queue utilization high: {input_utilization:.1f}% - "
                            "frames may be dropped if processing doesn't keep up"
                        )
                    if recording_utilization > self.QUEUE_WARNING_THRESHOLD:
                        logger.warning(
                            f"[QueueStats] Recording queue utilization high: {recording_utilization:.1f}% - "
                            "recording frames may be dropped if disk writes don't keep up"
                        )
                    
                    last_queue_stats_time = current_time

        except KeyboardInterrupt:
            logger.info("[BagCounterApp] Interrupted by user")
        except Exception as e:
            logger.error(f"[BagCounterApp] Error in main loop: {e}")
            import traceback

            logger.debug(f"[BagCounterApp] Traceback:\n{traceback.format_exc()}")
        finally:
            logger.info(f"[BagCounterApp] Shutting down (processed {frame_count} frames)...")
            self.is_running = False

            # Wait for recording thread to finish before releasing video writer
            # Timeout calculated for 30 frames at ~80ms each = ~2.4s, plus margin = 10s
            if self.recording_thread is not None and self.recording_thread.is_alive():
                logger.debug("[BagCounterApp] Waiting for recording thread to finish...")
                self.recording_thread.join(timeout=10)
                logger.debug("[BagCounterApp] Recording thread finished")

            # Release video writer if still open (recording thread should have handled this)
            with self.video_writer_lock:
                if self.video_writer is not None:
                    try:
                        self.video_writer.release()
                        logger.info("[BagCounterApp] Video writer released")
                    except Exception as e:
                        logger.error(f"[BagCounterApp] Error releasing video writer: {e}")
                    finally:
                        self.video_writer = None
                        self.segment_start_time = None
                        self.segment_counter = 0

            self.frame_source.cleanup()
            logger.debug("[BagCounterApp] Frame source cleaned up")

            self.config_watcher.stop()
            logger.debug("[BagCounterApp] Config watcher stopped")

            # --- ROS 2 CLEANUP ---
            if IS_RDK and self.ros_executor is not None:
                self.ros_executor.remove_node(self.ipc_publisher)

                if isinstance(self.frame_source, Node):
                    self.ros_executor.remove_node(self.frame_source)

            self.ipc_publisher.close_node()

            shutdown_ros2_context()
            if IS_RDK:
                logger.debug("[BagCounterApp] ROS 2 context shutdown")

            if self.ros_thread.is_alive():
                self.ros_thread.join(timeout=3)
                logger.debug("[BagCounterApp] ROS thread joined")

            if logic_thread.is_alive():
                logic_thread.join()
                logger.debug("[BagCounterApp] Logic thread joined")

            # Close database connection
            self.db.close()
            logger.debug("[BagCounterApp] Database connection closed")

            logger.info("[BagCounterApp] Shutdown complete")