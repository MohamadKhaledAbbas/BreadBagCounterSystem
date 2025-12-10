import os
import cv2
import queue
import threading
import time
import json
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
    INPUT_QUEUE_SIZE = 200  # Buffer size for input frames (200 frames @ 25fps = ~8 seconds)
    QUEUE_WARNING_THRESHOLD = 80  # Percentage threshold for queue utilization warnings
    STATS_LOG_INTERVAL = 5.0  # Log statistics every N seconds
    MIN_RECORDING_FPS = 1.0  # Minimum valid recording FPS to prevent division issues
    MIN_DETECTION_CONFIDENCE = 0.5  # Filter out low-confidence detections early
    
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
        self.classifier_service = ClassifierService(classifier_engine, max_workers=4)

        self.config_watcher = ConfigWatcher(db.db_path, poll_interval=5)
        self.config_watcher.add_watch(constants.show_ui_screen_key, self.on_show_ui_changed)
        self.config_watcher.add_watch(constants.is_recording_key, self.on_is_recording_changed)

        self.is_running = False

        # Recording state (frame-based)
        self.is_recording = db.get_config_value(constants.is_recording_key) == "1"
        # Use DB-configured path if present, else fall back to config
        self.recording_dir = db.get_config_value(constants.recording_dir) or config.recording_dir
        
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
        
        # Session directory for current recording session
        self.recording_session_dir = None
        self.recording_frame_counter = 0

        logger.info(f"[BagCounterApp] Frame Recording: {'ENABLED' if self.is_recording else 'DISABLED'}")
        logger.info(f"[BagCounterApp] Recording directory: {self.recording_dir}")
        logger.info(f"[BagCounterApp] Recording FPS: {self.recording_fps}")

        # Input queue size set to 100 frames for better buffering with 25 fps RTSP stream
        self.input_queue = queue.Queue(maxsize=self.INPUT_QUEUE_SIZE)
        
        # Queue monitoring statistics (thread-safe counters)
        self.stats_lock = threading.Lock()
        self.input_queue_drops = 0
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
        self.total_count = 0
        self.counted_events = set()  # events counted immediately when ready
        self.classified_events = set()  # events that completed classification callback
        self.classified_total = 0

        # --- IPC SETUP (ROS 2 - Executor Pattern) ---
        from src.utils.platform import IS_RDK

        logger.debug("[BagCounterApp] Initializing ROS 2 context...")
        self.ros_executor = init_ros2_context()

        self.is_publishing = db.get_config_value(constants.show_ui_screen_key) == "1"
        logger.info(f"[BagCounterApp] IPC Publishing: {'ENABLED' if self.is_publishing else 'DISABLED'}")

        self.ipc_publisher = FramePublisher(publish_rate_hz=25.0)


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
            # Create new session directory
            self._create_recording_session()
            logger.info("[BagCounterApp] Frame Recording ENABLED")
        else:
            self.is_recording = False
            logger.info("[BagCounterApp] Frame Recording DISABLED")
            # Reset session metadata
            self.recording_session_dir = None
            self.recording_frame_counter = 0
            self.last_recording_frame_time = None

    def _create_recording_session(self):
        """Create a new recording session directory."""
        try:
            os.makedirs(self.recording_dir, exist_ok=True)
            if not os.access(self.recording_dir, os.W_OK):
                logger.error(f"[Recording] Directory not writable: {self.recording_dir}")
                return False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_session_dir = os.path.join(self.recording_dir, f"session_{timestamp}")
            os.makedirs(self.recording_session_dir, exist_ok=True)
            self.recording_frame_counter = 0
            
            logger.info(f"[Recording] Created recording session: {self.recording_session_dir}")
            return True
        except Exception as e:
            logger.error(f"[Recording] Error creating recording session: {e}")
            return False

    def _save_frame_data(self, frame_raw, frame_annotated, detections_data, events_data, frame_number):
        """Save frame images (raw and annotated) and metadata as JSON."""
        if not self.recording_session_dir:
            return False
        
        try:
            # Generate filenames with zero-padded frame number
            frame_id = f"frame_{frame_number:06d}"
            raw_path = os.path.join(self.recording_session_dir, f"{frame_id}_raw.png")
            annotated_path = os.path.join(self.recording_session_dir, f"{frame_id}_annotated.png")
            json_path = os.path.join(self.recording_session_dir, f"{frame_id}.json")
            
            # Save raw frame
            if not cv2.imwrite(raw_path, frame_raw):
                logger.error(f"[Recording] Failed to save raw frame: {raw_path} (check disk space and permissions)")
                return False
            
            # Save annotated frame
            if not cv2.imwrite(annotated_path, frame_annotated):
                logger.error(f"[Recording] Failed to save annotated frame: {annotated_path} (check disk space and permissions)")
                return False
            
            # Prepare metadata
            metadata = {
                "frame_number": frame_number,
                "timestamp": datetime.now().isoformat(),
                "detections": detections_data,
                "events": events_data,
            }
            
            # Save JSON metadata
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"[Recording] Error saving frame data: {e}")
            return False

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        label = data["label"]
        phash = data["phash"]
        image_path = data["image_path"]
        conf = data.get("confidence", 1.0)
        candidates_count = data.get("candidates_evaluated", 1)
        is_low_confidence = data.get("is_low_confidence", False)
        decision_margin = data.get("decision_margin", None)

        if track_id in self.classified_events:
            logger.debug(f"[BagCounterApp] Duplicate classification ignored for track {track_id}")
            return
        self.classified_events.add(track_id)

        logger.info(
            f"[BagCounterApp] Classification result: track={track_id}, "
            f"label={label}, conf={conf:.3f}, low_conf={is_low_confidence}, "
            f"margin={decision_margin:.3f if decision_margin is not None else 'N/A'}"
        )
        logger.debug(
            f"[BagCounterApp] Result details: phash={phash}, "
            f"image_path={image_path}, candidates={candidates_count}"
        )

        bag_type_id = self.db.get_or_create_bag_type(label, phash, image_path)
        self.db.log_event(bag_type_id, track_id, conf, is_low_confidence, decision_margin)

        if track_id in self.counted_events:
            unclassified_count = self.ui_counts.get("Unclassified", 0)
            if unclassified_count > 0:
                self.ui_counts["Unclassified"] = unclassified_count - 1
                if self.ui_counts["Unclassified"] == 0:
                    self.ui_counts.pop("Unclassified", None)

        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1
        self.classified_total += 1

        if self.classified_total > self.total_count:
            logger.warning(
                f"[BagCounterApp] Classified count ({self.classified_total}) exceeds "
                f"preliminary total ({self.total_count})"
            )
        elif self.total_count > self.classified_total:
            logger.debug(
                f"[BagCounterApp] Awaiting classifications: "
                f"preliminary={self.total_count}, classified={self.classified_total}"
            )

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
                        if confidences[i] < self.MIN_DETECTION_CONFIDENCE:
                            logger.debug(
                                f"[LogicThread] Skipping low-conf detection "
                                f"class_id={cls_ids[i]}, conf={confidences[i]:.3f} "
                                f"(min={self.MIN_DETECTION_CONFIDENCE})"
                            )
                            continue
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

                    for event_id, _ in ready_events:
                        if event_id not in self.counted_events:
                            self.counted_events.add(event_id)
                            self.total_count += 1
                            self.ui_counts["Total"] = self.total_count
                            self.ui_counts["Unclassified"] = self.ui_counts.get("Unclassified", 0) + 1

                    classify_start = time.perf_counter()
                    for event_id, candidates in ready_events:
                        logger.debug(
                            f"[LogicThread] Sending event {event_id} to classifier "
                            f"({len(candidates)} candidates)"
                        )
                        self.classifier_service.process(event_id, candidates)
                    classify_end = time.perf_counter()
                    classify_time = (classify_end - classify_start) * 1000  # Convert to ms

                # --- 4. ANNOTATION & VISUALIZATION (shared between publishing and recording) ---
                annotated_frame = None
                visualization_time = 0.0
                
                # Check if we need to record this frame
                should_record_frame = False
                if self.is_recording:
                    if self.last_recording_frame_time is None:
                        should_record_frame = True
                        self.last_recording_frame_time = time.perf_counter()
                    else:
                        time_since_last = time.perf_counter() - self.last_recording_frame_time
                        if time_since_last >= self.recording_frame_interval:
                            should_record_frame = True
                            self.last_recording_frame_time = time.perf_counter()
                
                # Create annotated frame if needed for publishing or recording
                if self.is_publishing or should_record_frame:
                    viz_start = time.perf_counter()
                    
                    annotated_frame = frame.copy()
                    
                    # Calculate FPS for display
                    frame_mid = time.perf_counter()
                    mid_time = (frame_mid - frame_start) * 1000
                    fps_display = 1000 / mid_time if mid_time > 0 else 0
                    
                    self.visualizer.render_all(
                        annotated_frame,
                        current_frame_detections,
                        self.monitor.active_events,
                        counts=self.ui_counts,
                        fps=fps_display,
                    )
                    
                    viz_end = time.perf_counter()
                    visualization_time = (viz_end - viz_start) * 1000

                # --- 5. PUBLISHING LOGIC ---
                publish_time = 0.0

                if self.is_publishing and annotated_frame is not None:
                    publish_start = time.perf_counter()
                    
                    annotated_frame_resized = cv2.resize(annotated_frame, (1280, 720))
                    self.ipc_publisher.publish(annotated_frame_resized)

                    publish_end = time.perf_counter()
                    publish_time = (publish_end - publish_start) * 1000  # Convert to ms

                # --- 6. RECORDING LOGIC (Frame-based) ---
                record_time = 0.0

                if should_record_frame and annotated_frame is not None:
                    record_start = time.perf_counter()
                    
                    # Prepare detection data for JSON
                    detections_data = []
                    for det in current_frame_detections:
                        class_name = self.detector.class_names.get(det["class_id"], "Unknown")
                        detections_data.append({
                            "class_id": int(det["class_id"]),
                            "class_name": class_name,
                            "confidence": float(det["conf"]),
                            "bbox": [float(det["box"][0]), float(det["box"][1]), 
                                    float(det["box"][2]), float(det["box"][3])],
                        })
                    
                    # Prepare events data for JSON
                    events_data = []
                    for event in self.monitor.active_events:
                        events_data.append({
                            "id": event.id,
                            "state": event.state,
                            "bbox": [float(event.box[0]), float(event.box[1]), 
                                    float(event.box[2]), float(event.box[3])],
                        })
                    
                    # Save frame data
                    self._save_frame_data(
                        frame_raw=frame,
                        frame_annotated=annotated_frame,
                        detections_data=detections_data,
                        events_data=events_data,
                        frame_number=self.recording_frame_counter
                    )
                    self.recording_frame_counter += 1
                    
                    record_end = time.perf_counter()
                    record_time = (record_end - record_start) * 1000  # Convert to ms

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
                    if visualization_time > 0:
                        timing_msg += f" | Visualize: {visualization_time:.1f}ms"
                    if publish_time > 0:
                        timing_msg += f" | Publish: {publish_time:.1f}ms"
                    if record_time > 0:
                        timing_msg += f" | Record: {record_time:.1f}ms"
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
        
        # Create recording session if recording is enabled at startup
        if self.is_recording:
            self._create_recording_session()

        logic_thread = threading.Thread(target=self._logic_thread_loop, daemon=True)
        logic_thread.start()

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
                    frame_dropped = False
                    try:
                        self.input_queue.get_nowait()
                        # Successfully dropped oldest frame, increment counter
                        frame_dropped = True
                        # Now try to add new frame
                        try:
                            self.input_queue.put_nowait(frame)
                        except queue.Full:
                            # Extremely rare: queue filled again between get and put
                            # Current frame also dropped
                            frame_dropped = True
                            logger.debug(
                                f"[BagCounterApp] Frame {frame_count} dropped: "
                                "queue refilled immediately after clearing"
                            )
                        # Log the drop of the old frame
                        if frame_dropped:
                            with self.stats_lock:
                                self.input_queue_drops += 1
                                drops = self.input_queue_drops
                            logger.warning(
                                f"[BagCounterApp] Dropped old frame (input queue full, "
                                f"total drops: {drops})"
                            )
                    except queue.Empty:
                        # Queue was drained by another thread, retry putting the frame
                        try:
                            self.input_queue.put_nowait(frame)
                        except queue.Full:
                            # Still full after being drained, current frame must be dropped
                            with self.stats_lock:
                                self.input_queue_drops += 1
                                drops = self.input_queue_drops
                            logger.warning(
                                f"[BagCounterApp] Frame {frame_count} dropped: "
                                f"queue refilled by another thread (total drops: {drops})"
                            )
                
                # Periodic queue statistics logging
                if current_time - last_queue_stats_time >= self.STATS_LOG_INTERVAL:
                    input_size = self.input_queue.qsize()
                    input_utilization = (input_size / self.INPUT_QUEUE_SIZE) * 100
                    
                    # Thread-safe read of drop counter
                    with self.stats_lock:
                        input_drops = self.input_queue_drops
                    
                    logger.info(
                        f"[QueueStats] Input queue: {input_size}/{self.INPUT_QUEUE_SIZE} "
                        f"({input_utilization:.1f}% full, drops={input_drops})"
                    )
                    
                    # Warning if queue is getting full
                    if input_utilization > self.QUEUE_WARNING_THRESHOLD:
                        logger.warning(
                            f"[QueueStats] Input queue utilization high: {input_utilization:.1f}% - "
                            "frames may be dropped if processing doesn't keep up"
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

            # Log final recording stats if applicable
            if self.recording_session_dir:
                logger.info(f"[BagCounterApp] Recording session saved {self.recording_frame_counter} frames to: {self.recording_session_dir}")

            self.frame_source.cleanup()
            logger.debug("[BagCounterApp] Frame source cleaned up")

            self.config_watcher.stop()
            logger.debug("[BagCounterApp] Config watcher stopped")

            # Shutdown classifier service
            self.classifier_service.shutdown(wait=True)
            logger.debug("[BagCounterApp] Classifier service shutdown")

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
