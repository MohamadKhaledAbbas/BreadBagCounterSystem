import os
import cv2
import queue
import threading
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

from src.logging.ConfigWatcher import ConfigWatcher
from src.utils.AppLogging import logger

# --- ROS 2 Check Imports ---
from rclpy.node import Node
# ---------------------------
# Import the ROS 2 helper functions from IPC. py
from src.counting.IPC import ExecutorThread, init_ros2_context, shutdown_ros2_context
from src.counting.FramePublisherNode import FramePublisher


# -------------------


def on_is_recording_changed(new_value):
    if new_value == "1":
        logger.info("[ConfigWatcher] Recording ENABLED")
    else:
        logger.info("[ConfigWatcher] Recording DISABLED")


class BagCounterApp:
    def __init__(self,
                 video_path: str,
                 detector_engine: BaseDetector,
                 tracker: BaseTracker,
                 classifier_engine: BaseClassifier,
                 db: DatabaseManager,
                 is_development: bool
                 ):

        logger.info("[BagCounterApp] Initializing...")
        self.db = db
        self.detector = detector_engine
        self.tracker = tracker
        self.classifier_service = ClassifierService(classifier_engine)

        self.config_watcher = ConfigWatcher(db.db_path, poll_interval=5)
        self.config_watcher.add_watch(constants.show_ui_screen_key, self.on_show_ui_changed)
        self.config_watcher.add_watch(constants.is_recording_key, on_is_recording_changed)

        self.is_running = False

        self.input_queue = queue.Queue(maxsize=1)

        names = self.detector.class_names
        logger.info(f"[BagCounterApp] Detector class names: {names}")
        name_to_id = {v: k for k, v in names.items()}

        try:
            open_id = name_to_id['bread-bag-opened']
            closed_id = name_to_id['bread-bag-closed']
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
        logger.debug("[BagCounterApp] Initializing ROS 2 context...")
        self.ros_executor = init_ros2_context()

        self.is_publishing = db.get_config_value(constants.show_ui_screen_key) == "1"
        logger.info(f"[BagCounterApp] IPC Publishing: {'ENABLED' if self.is_publishing else 'DISABLED'}")

        self.ipc_publisher = FramePublisher(publish_rate_hz=30.0)
        self.ros_executor.add_node(self.ipc_publisher)

        if is_development:
            self.frame_source = FrameSourceFactory.create("opencv", source=video_path)
            logger.info(f"[BagCounterApp] Development mode: reading from {video_path}")
        else:
            os.environ["HOME"] = "/home/sunrise"
            self.frame_source = FrameSourceFactory.create("ros2")
            logger.info("[BagCounterApp] Production mode: reading from ROS 2 stream")

        if isinstance(self.frame_source, Node):
            self.ros_executor.add_node(self.frame_source)
            logger.debug("[BagCounterApp] FrameSource added to ROS 2 executor")

        self.ros_thread = ExecutorThread(self.ros_executor)
        self.ros_thread.start()
        logger.debug("[BagCounterApp] ROS 2 executor thread started")

        logger.info("[BagCounterApp] Initialization complete")

    def on_show_ui_changed(self, new_value):
        if new_value == "1":
            self.is_publishing = True
            logger.info("[BagCounterApp] IPC Publishing ENABLED")
        else:
            self.is_publishing = False
            logger.info("[BagCounterApp] IPC Publishing DISABLED")

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        label = data['label']
        phash = data['phash']
        image_path = data['image_path']
        conf = data.get('confidence', 1.0)
        candidates_count = data.get('candidates_evaluated', 1)

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
                t1 = cv2.getTickCount()

                # 1. Run Detector
                detections = self.detector.predict(frame)
                current_frame_detections = []

                if len(detections) > 0 and hasattr(detections[0], 'boxes') and len(detections[0].boxes) > 0:
                    xyxy = detections[0].boxes.xyxy.cpu().numpy()
                    cls_ids = detections[0].boxes.cls.cpu().numpy().astype(int)
                    confidences = detections[0].boxes.conf.cpu().numpy()

                    for i in range(len(cls_ids)):
                        current_frame_detections.append({
                            'box': xyxy[i],
                            'class_id': cls_ids[i],
                            'conf': confidences[i]
                        })

                    logger.debug(
                        f"[LogicThread] Frame {frame_count}: "
                        f"{len(current_frame_detections)} detections"
                    )

                # 2. Update Monitor
                ready_events = self.monitor.update(current_frame_detections, frame)

                # 3.  Process Ready Events
                if ready_events:
                    logger.info(
                        f"[LogicThread] Frame {frame_count}: "
                        f"{len(ready_events)} events ready for classification"
                    )

                for event_id, candidates in ready_events:
                    logger.debug(
                        f"[LogicThread] Sending event {event_id} to classifier "
                        f"({len(candidates)} candidates)"
                    )
                    self.classifier_service.process(event_id, candidates)

                # --- 4.  PUBLISHING LOGIC ---
                if self.is_publishing:
                    tracks_for_ui = []
                    for event in self.monitor.active_events:
                        t = SimpleTrack(
                            event.id,
                            event.box,
                            event.open_id if event.state == 'detecting_open' else event.closed_id
                        )
                        tracks_for_ui.append(t)

                    t2 = cv2.getTickCount()
                    latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
                    fps = 1000 / latency if latency else 0

                    annotated_frame = frame.copy()
                    self.visualizer.draw_detections(annotated_frame, tracks_for_ui)
                    self.visualizer.draw_stats(annotated_frame, self.ui_counts)
                    cv2.putText(
                        annotated_frame, f"FPS: {int(fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))

                    self.ipc_publisher.publish(annotated_frame)
                    logger.debug(f"[LogicThread] Frame {frame_count}: Published (FPS={fps:.1f})")

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

        frame_count = 0

        try:
            for frame, latencyMs in self.frame_source.frames():
                frame_count += 1

                if self.input_queue.full():
                    try:
                        self.input_queue.get_nowait()
                        logger.debug(f"[BagCounterApp] Dropped frame {frame_count} (queue full)")
                    except queue.Empty:
                        pass

                self.input_queue.put(frame)

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
            self.ros_executor.remove_node(self.ipc_publisher)

            if isinstance(self.frame_source, Node):
                self.ros_executor.remove_node(self.frame_source)

            self.ipc_publisher.close_node()
            self.frame_source.cleanup()

            shutdown_ros2_context()
            logger.debug("[BagCounterApp] ROS 2 context shutdown")

            if self.ros_thread.is_alive():
                self.ros_thread.join(timeout=3)
                logger.debug("[BagCounterApp] ROS thread joined")

            if logic_thread.is_alive():
                logic_thread.join()
                logger.debug("[BagCounterApp] Logic thread joined")

            logger.info("[BagCounterApp] Shutdown complete")