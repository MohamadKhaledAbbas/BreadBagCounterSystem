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
# Import the ROS 2 helper functions from IPC.py
from src.counting.IPC import ExecutorThread, init_ros2_context, shutdown_ros2_context
from src.counting.FramePublisherNode import FramePublisher
# -------------------

def on_is_recording_changed(new_value):
    if new_value == "1":
        logger.debug("[ConfigStateMonitor] recording enabled")
    else:
        logger.debug("[ConfigStateMonitor] recording disabled")


class BagCounterApp:
    def __init__(self,
                 video_path: str,
                 detector_engine: BaseDetector,
                 tracker: BaseTracker,
                 classifier_engine: BaseClassifier,
                 db: DatabaseManager,
                 is_development: bool
                 ):

        logger.debug("[INIT] BagCounterApp initializing...")
        self.db = db
        self.detector = detector_engine
        self.tracker = tracker
        self.classifier_service = ClassifierService(classifier_engine)

        self.config_watcher = ConfigWatcher(db.db_path, poll_interval=5)
        # Point to the modified handler
        self.config_watcher.add_watch(constants.show_ui_screen_key, self.on_show_ui_changed)
        self.config_watcher.add_watch(constants.is_recording_key, on_is_recording_changed)

        self.is_running = False

        self.input_queue = queue.Queue(maxsize=1)
        # Result queue removed: we publish directly from logic thread

        names = self.detector.class_names
        logger.info(f'[INIT] Found detector class names: {names}')
        name_to_id = {v: k for k, v in names.items()}
        try:
            open_id = name_to_id['bread-bag-opened']
            closed_id = name_to_id['bread-bag-closed']
        except KeyError:
            logger.error("[ERROR] Model missing required classes")
            raise ValueError("Model missing required classes")

        self.monitor = BagStateMonitor(open_id, closed_id)
        self.visualizer = Visualizer(names)
        self.classifier_service.register_callback(self.on_classification_result)
        self.ui_counts = {}

        # --- IPC SETUP (ROS 2 - Executor Pattern) ---
        # 1. Initialize ROS 2 context and get the Executor instance
        self.ros_executor = init_ros2_context()

        self.is_publishing = db.get_config_value(constants.show_ui_screen_key) == "1"

        # 2. Create the ROS 2 Publisher Node
        self.ipc_publisher = FramePublisher(publish_rate_hz=30.0)

        # 3. Add necessary nodes to the shared Executor
        self.ros_executor.add_node(self.ipc_publisher)

        if is_development:
            self.frame_source = FrameSourceFactory.create("opencv", source=video_path)
            logger.info("[INFO] running in development mode, reading from Video file...")
            logger.info("[INFO] loading video from {}".format(video_path))
        else:
            os.environ["HOME"] = "/home/sunrise"
            self.frame_source = FrameSourceFactory.create("ros2")
            logger.info("[INFO] running in production mode, reading from RTSP stream...")

        # Conditional check: If the FrameSource passed is a ROS 2 Node, add it to the executor.
        if isinstance(self.frame_source, Node):
            self.ros_executor.add_node(self.frame_source)
            logger.info("[INIT] FrameSource is a ROS 2 Node and has been added to the shared Executor.")

        # 4. Start the dedicated thread for the ROS 2 executor's spin
        self.ros_thread = ExecutorThread(self.ros_executor)
        self.ros_thread.start()

        logger.info("[INIT] ROS 2 Executor Thread initialized. All ROS nodes are spinning.")


        logger.info("[INIT] Initialization complete")

    def on_show_ui_changed(self, new_value):
        # Instead of opening windows, we just toggle the publishing flag
        if new_value == "1":
            self.is_publishing = True
            logger.info("[ConfigStateMonitor] UI flag set: IPC Publishing ENABLED")
        else:
            self.is_publishing = False
            logger.info("[ConfigStateMonitor] UI flag unset: IPC Publishing DISABLED")

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        logger.debug(f"[CALLBACK] on_classification_result called for Track {track_id}")
        label = data['label']
        phash = data['phash']
        image_path = data['image_path']
        conf = data.get('confidence', 1.0)

        bag_type_id = self.db.get_or_create_bag_type(label, phash, image_path)
        self.db.log_event(bag_type_id, track_id, conf)
        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1

    def _logic_thread_loop(self):
        logger.debug("[THREAD] ✅ Logic Thread Started")

        # Define SimpleTrack class here or at module level
        class SimpleTrack:
            def __init__(self, tid, box, cid):
                self.track_id = tid
                self.box = box
                self.class_id = cid

        while self.is_running:
            try:
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                if not self.is_running: break
                continue
            except Exception as e:
                logger.error(f"[THREAD] Input Queue Error: {e}")
                continue

            try:
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

                # 2. Update Monitor
                ready_events = self.monitor.update(current_frame_detections, frame)

                # 3. Process Ready Events
                for event_id, best_roi in ready_events:
                    self.classifier_service.process(event_id, best_roi)

                # --- 4. PUBLISHING LOGIC (Only if enabled) ---
                if self.is_publishing:
                    # Prepare data for visualization
                    tracks_for_ui = []
                    for event in self.monitor.active_events:
                        t = SimpleTrack(event.id, event.box,
                                        event.open_id if event.state == 'detecting_open' else event.closed_id)
                        tracks_for_ui.append(t)

                    # Calculate FPS
                    t2 = cv2.getTickCount()
                    latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
                    fps = 1000 / latency if latency else 0

                    # Draw on a COPY of the frame (so we don't mess up next logic steps if any)
                    annotated_frame = frame.copy()

                    # Use Visualizer to draw boxes/text
                    self.visualizer.draw_detections(annotated_frame, tracks_for_ui)
                    self.visualizer.draw_stats(annotated_frame, self.ui_counts)
                    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
                    # Send to Shared Memory
                    self.ipc_publisher.publish(annotated_frame)
                    logger.debug(f"[LOGIC-THREAD] Frame sent to IPC buffer. Publishing: {self.is_publishing}")
                # ---------------------------------------------

            except Exception as e:
                logger.error(f"[THREAD ERROR] Exception in logic: {e}")
                import traceback
                traceback.print_exc()

    def run(self):
        logger.info("[RUN] Starting BagCounterApp.run() [HEADLESS MODE]")
        self.is_running = True

        logic_thread = threading.Thread(target=self._logic_thread_loop, daemon=True)
        logic_thread.start()

        self.config_watcher.start()

        try:
            # Main loop now ONLY handles ingestion. No UI.
            for frame, latencyMs in self.frame_source.frames():
                if self.input_queue.full():
                    try:
                        self.input_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.input_queue.put(frame)

                # Small sleep to prevent CPU hogging if source is faster than processing
                # but usually frame_source.frames() controls timing.

        except Exception as e:
            logger.error(f"[RUN ERROR] Exception in main loop: {e}")
        finally:
            logger.info("[RUN] Cleaning up...")
            self.is_running = False
            self.frame_source.cleanup()
            self.config_watcher.stop()
            # --- ROS 2 CLEANUP (Executor Pattern) ---
            # 1. Remove Publisher node from the executor
            self.ros_executor.remove_node(self.ipc_publisher)

            # 2. Remove FrameSource node if it was a ROS 2 Node
            if isinstance(self.frame_source, Node):
                self.ros_executor.remove_node(self.frame_source)

            # 3. Destroy the node instances and cleanup the source
            self.ipc_publisher.close_node()
            self.frame_source.cleanup()  # Calls cleanup (which calls destroy_node for FrameServer)

            # 4. Stop the ROS 2 execution context and shut down the executor
            shutdown_ros2_context()

            if self.ros_thread.is_alive():
                self.ros_thread.join(timeout=3)
            # ---------------------
            if logic_thread.is_alive():
                logic_thread.join()

            logger.info("[RUN] App Closed.")