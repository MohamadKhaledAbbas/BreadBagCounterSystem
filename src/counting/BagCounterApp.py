import os
import cv2
import json
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

from src.counting.IPC import ExecutorThread, init_ros2_context, shutdown_ros2_context
from src.counting.FramePublisherNode import FramePublisher

if IS_RDK:
    from rclpy.node import Node
else:
    class Node:
        pass


class BagCounterApp:
    # Queue configuration constants
    INPUT_QUEUE_SIZE = 100  # Buffer size for input frames
    QUEUE_WARNING_THRESHOLD = 80  # Percentage threshold for queue utilization warnings
    STATS_LOG_INTERVAL = 5.0  # Log statistics every N seconds

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
        # Recording is removed; keep watcher to avoid errors if config changes exist
        self.config_watcher.add_watch(constants.is_recording_key, self.on_is_recording_changed)

        self.is_running = False

        # Recording removed; snapshots only
        self.is_recording = False
        logger.info("[BagCounterApp] Video Recording: DISABLED (snapshots only)")

        # Snapshot directory
        self.recording_dir = db.get_config_value(constants.recording_dir) or config.recording_dir
        self.snapshot_dir = os.path.join(self.recording_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        logger.info(f"[BagCounterApp] Snapshot directory: {self.snapshot_dir}")

        # Input queue
        self.input_queue = queue.Queue(maxsize=self.INPUT_QUEUE_SIZE)

        # Queue monitoring statistics
        self.stats_lock = threading.Lock()
        self.input_queue_drops = 0
        self.last_queue_stats_log_time = time.perf_counter()

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

        logger.debug("[BagCounterApp] Initializing ROS 2 context...")
        self.ros_executor = init_ros2_context()

        self.is_publishing = db.get_config_value(constants.show_ui_screen_key) == "1"
        logger.info(f"[BagCounterApp] IPC Publishing: {'ENABLED' if self.is_publishing else 'DISABLED'}")

        self.ipc_publisher = FramePublisher(publish_rate_hz=25.0)

        if IS_RDK and self.ros_executor is not None:
            self.ros_executor.add_node(self.ipc_publisher)

        if is_development:
            self.frame_source = FrameSourceFactory.create("opencv", source=video_path, target_fps=30.0)
            logger.info(f"[BagCounterApp] Development mode: reading from {video_path}")
        else:
            if IS_RDK:
                os.environ["HOME"] = "/home/sunrise"
                self.frame_source = FrameSourceFactory.create("ros2", target_fps=30.0)
                logger.info("[BagCounterApp] Production mode: reading from ROS 2 stream")
            else:
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
        # Recording is disabled; keep for compatibility/logging
        logger.info("[BagCounterApp] Recording change ignored (recording disabled)")

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

        timestamp_str = datetime.fromtimestamp(ts_epoch).strftime("%Y%m%d_%H%M%S_%f")[:-3]
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
                "box": box.tolist() if hasattr(box, "tolist") else list(box) if box is not None else None,
                "class_id": d.get("class_id"),
                "class_name": self.detector.class_names.get(d.get("class_id"), "Unknown"),
                "conf": float(d.get("conf", 0)),
            })

        meta = {
            "timestamp": timestamp_str,
            "timestamp_epoch": ts_epoch,
            "frame_id": frame_id,
            "track_id": track_id,
            "label": label,
            "confidence": float(conf),
            "phash": phash,
            "roi_saved_path": image_path,
            "candidates_evaluated": candidates_count,
            "event_box": event_box.tolist() if hasattr(event_box, "tolist") else list(event_box) if event_box is not None else None,
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

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        label = data["label"]
        phash = data["phash"]
        image_path = data["image_path"]
        conf = data.get("confidence", 1.0)
        candidates_count = data.get("candidates_evaluated", 1)
        context = data.get("context")

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

        # Save snapshot if context is available
        if context and context.get("frame") is not None:
            try:
                self._save_snapshot(track_id, label, conf, phash, image_path, candidates_count, context)
            except Exception as e:
                logger.error(f"[BagCounterApp] Snapshot save error: {e}")

    # --- Main logic thread ---

    def _logic_thread_loop(self):
        logger.info("[LogicThread] Started")

        TIMING_LOG_INTERVAL = 30
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
                frame_start = time.perf_counter()

                # 1. Run Detector
                detect_start = time.perf_counter()
                detections = self.detector.predict(frame)
                detect_end = time.perf_counter()
                detect_time = (detect_end - detect_start) * 1000

                current_frame_detections = []

                if len(detections) > 0 and hasattr(detections[0], "boxes") and len(detections[0].boxes) > 0:
                    xyxy = detections[0].boxes.xyxy.cpu().numpy()
                    cls_ids = detections[0].boxes.cls.cpu().numpy().astype(int)
                    confidences = detections[0].boxes.conf.cpu().numpy()

                    for i in range(len(cls_ids)):
                        current_frame_detections.append(
                            {"box": xyxy[i], "class_id": cls_ids[i], "conf": confidences[i]}
                        )

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
                monitor_time = (monitor_end - monitor_start) * 1000

                # 3. Process Ready Events (classification + snapshot context)
                classify_time = 0.0
                if ready_events:
                    logger.info(
                        f"[LogicThread] Frame {frame_count}: "
                        f"{len(ready_events)} events ready for classification"
                    )

                    classify_start = time.perf_counter()
                    for event_id, candidates, event_box, event_stats in ready_events:
                        logger.debug(
                            f"[LogicThread] Sending event {event_id} to classifier "
                            f"({len(candidates)} candidates)"
                        )
                        det_copy = []
                        for d in current_frame_detections:
                            det_copy.append({
                                "box": d["box"].copy(),
                                "class_id": d["class_id"],
                                "conf": float(d.get("conf", 0)),
                            })
                        context = {
                            "frame": frame.copy(),
                            "detections": det_copy,
                            "event_box": event_box,
                            "event_stats": event_stats,
                            "frame_id": frame_count,
                            "timestamp": time.time(),
                        }
                        self.classifier_service.process(event_id, candidates, context=context)
                    classify_end = time.perf_counter()
                    classify_time = (classify_end - classify_start) * 1000

                # 4. Publishing logic (unchanged)
                publish_time = 0.0
                if self.is_publishing:
                    publish_start = time.perf_counter()

                    annotated_frame = frame.copy()
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

                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    self.ipc_publisher.publish(annotated_frame)

                    publish_end = time.perf_counter()
                    publish_time = (publish_end - publish_start) * 1000

                # 5. Timing logs
                frame_end = time.perf_counter()
                total_time = (frame_end - frame_start) * 1000
                fps = 1000 / total_time if total_time > 0 else 0

                if frame_count % TIMING_LOG_INTERVAL == 0 or classify_time > 0:
                    timing_msg = (
                        f"[Frame {frame_count}] Total: {total_time:.1f}ms | "
                        f"Detect: {detect_time:.1f}ms | "
                        f"Monitor: {monitor_time:.1f}ms"
                    )
                    if classify_time > 0:
                        timing_msg += f" | Classify: {classify_time:.1f}ms"
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

        self.config_watcher.start()
        logger.debug("[BagCounterApp] Config watcher started")

        FRAME_STATS_INTERVAL = 100
        TIMING_EPSILON = 1e-6

        frame_count = 0
        last_frame_time = None
        frame_interval_sum = 0.0
        frame_interval_count = 0
        last_queue_stats_time = time.perf_counter()

        try:
            for frame, latencyMs in self.frame_source.frames():
                frame_count += 1
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
                            f"frames={frame_count}, avg_interval={avg_interval*1000:.1f}ms, "
                            f"acquisition_fps={acquisition_fps:.1f}"
                        )
                    else:
                        logger.warning(
                            f"[BagCounterApp] Invalid frame timing detected: "
                            f"frames={frame_count}, avg_interval={avg_interval*1000:.6f}ms "
                            f"(below {TIMING_EPSILON*1000:.6f}ms threshold) - skipping FPS calculation"
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

                # Queue stats (input queue only)
                if current_time - last_queue_stats_time >= self.STATS_LOG_INTERVAL:
                    input_size = self.input_queue.qsize()
                    input_utilization = (input_size / self.INPUT_QUEUE_SIZE) * 100

                    with self.stats_lock:
                        input_drops = self.input_queue_drops

                    logger.info(
                        f"[QueueStats] Input queue: {input_size}/{self.INPUT_QUEUE_SIZE} "
                        f"({input_utilization:.1f}% full, drops={input_drops})"
                    )

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