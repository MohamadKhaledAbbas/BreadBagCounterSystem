import cv2
import time
import threading
import queue
from typing import Dict, Any

from src.counting.Visualizer import Visualizer
from src.classifier.ClassifierService import ClassifierService
from src.classifier.BaseClassifier import BaseClassifier
from src.counting.BagStateMonitor import BagStateMonitor
from src.detection.BaseDetection import BaseDetector
from src.logging.Database import DatabaseManager
from src.frame_source.FrameSourceFactory import FrameSource
from src.tracking.BaseTracker import BaseTracker
from src import constants

from src.logging.ConfigWatcher import ConfigWatcher
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
                 frame_source: FrameSource,
                 show_ui_screen: bool,
                 ):

        logger.debug("[INIT] BagCounterApp initializing...")
        self.db = db
        self.show_ui_screen = show_ui_screen
        self.video_path = video_path
        self.detector = detector_engine
        self.tracker = tracker
        self.classifier_service = ClassifierService(classifier_engine)
        self.frame_source = frame_source

        self.config_watcher = ConfigWatcher(db.db_path, poll_interval=5)
        self.config_watcher.add_watch(constants.show_ui_screen_key, self.on_show_ui_changed)
        self.config_watcher.add_watch(constants.is_recording_key, on_is_recording_changed)

        self.is_running = False

        self.input_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

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
        self.ui_should_destroy = False

        logger.info("[INIT] Initialization complete")

    def on_show_ui_changed(self, new_value):
        # expect new_value as string, e.g. "1" or "0"
        if new_value == "1":
            self.show_ui_screen = True
            logger.debug("[ConfigStateMonitor] UI enabled → showing window allowed.")
        else:
            logger.debug("[ConfigStateMonitor] UI disabled → closing OpenCV windows...")
            self.show_ui_screen = False
            self.ui_should_destroy = True
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            # Destroy windows in main loop



    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        logger.debug(f"[CALLBACK] on_classification_result called for Track {track_id}")
        label = data['label']
        phash = data['phash']
        image_path = data['image_path']
        conf = data.get('confidence', 1.0)
        logger.debug(f"[CALLBACK] Received result: label={label}, phash={phash}, image_path={image_path}, conf={conf}")
        bag_type_id = self.db.get_or_create_bag_type(label, phash, image_path)
        logger.debug(f"[CALLBACK] Bag type id got: {bag_type_id}")
        self.db.log_event(bag_type_id, track_id, conf)
        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1

    def _logic_thread_loop(self):
        logger.debug("[THREAD] ✅ Logic Thread Started")
        while self.is_running:
            try:
                logger.debug("[THREAD] Trying to get frame from input_queue...")
                frame = self.input_queue.get(timeout=1.0)
                logger.debug("[THREAD] Logic thread got frame")
            except queue.Empty:
                logger.debug("[THREAD] input_queue empty, retrying...")
                if not self.is_running:
                    logger.debug("[THREAD] Detected is_running=False, exiting logic thread!")
                    break
                continue
            except Exception as e:
                logger.error(f"[THREAD ERROR] Exception on input_queue.get: {e}")
                continue

            try:
                t1 = cv2.getTickCount()
                logger.debug("[THREAD] Running detector.predict...")
                detections = self.detector.predict(frame)
                logger.debug(f"[THREAD] Detector returned {len(detections)} detection batches.")
                tracks = []

                if len(detections) > 0 and hasattr(detections[0], 'boxes') and len(detections[0].boxes) > 0:
                    logger.debug(f"[THREAD] Detector boxes found: {len(detections[0].boxes)}")
                    xyxy = detections[0].boxes.xyxy.cpu().numpy()
                    conf = detections[0].boxes.conf.cpu().numpy()
                    cls_ids = detections[0].boxes.cls.cpu().numpy().astype(int)

                    logger.debug(f"[THREAD] Calling tracker.update...")
                    tracks = self.tracker.update(xyxy, conf, cls_ids)
                    logger.debug(f"[THREAD] Tracker returned {len(tracks)} tracks")

                    active_ids = set()
                    for det in tracks:
                        active_ids.add(det.track_id)
                        logger.debug(f"[THREAD] Monitor.update for track_id {det.track_id}, class_id {det.class_id}")
                        event = self.monitor.update(det.track_id, det.class_id)

                        if event == 'READY_TO_CLASSIFY':
                            logger.debug("[THREAD] READY_TO_CLASSIFY event is True")
                            h, w = frame.shape[:2]
                            x1, y1, x2, y2 = map(int, det.box)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            logger.debug(f"[THREAD] ROI coords: ({x1},{y1})-({x2},{y2})")
                            if x2 > x1 and y2 > y1:
                                roi = frame[y1:y2, x1:x2].copy()
                                logger.debug("[THREAD] Calling classifier_service.process...")
                                self.classifier_service.process(det.track_id, roi)
                    logger.debug(f"[THREAD] Calling monitor.cleanup...")
                    self.monitor.cleanup(active_ids)
                else:
                    logger.debug("[THREAD] No boxes found by detector.")

                t2 = cv2.getTickCount()
                latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
                fps = 1000 / latency if latency else 0

                # Send to Display (Leaky Bucket)
                if self.show_ui_screen:
                    logger.debug("[THREAD] In show_ui_screen block")
                    if self.result_queue.full():
                        logger.debug("[THREAD] result_queue is full, popping one item")
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            logger.debug("[THREAD] result_queue unexpectedly empty")
                            pass
                    logger.debug("[THREAD] Logic thread putting result to result_queue")
                    self.result_queue.put((frame, tracks, fps, self.ui_counts.copy()))
                else:
                    logger.debug(f"[THREAD] Not in UI mode: fps: {int(fps)} , latency: {latency:.2f} ms")
            except Exception as e:
                logger.error(f"[THREAD ERROR] Exception in logic: {e}")

    def run(self):
        logger.info("[RUN] Starting BagCounterApp.run()")
        self.is_running = True

        logic_thread = threading.Thread(target=self._logic_thread_loop, daemon=True)
        logic_thread.start()
        logger.info("[RUN] Logic thread started.")

        logger.info("[RUN] ✅ Main Loop Started (Display & Input)")
        self.config_watcher.start()
        try:
            for frame, latencyMs in self.frame_source.frames():
                logger.info(f"[RUN] Main loop got frame, latency: {latencyMs:.2f} ms")

                if self.input_queue.full():
                    logger.debug("[RUN] input_queue full, popping one frame")
                    try:
                        self.input_queue.get_nowait()
                    except queue.Empty:
                        logger.debug("[RUN] input_queue unexpectedly empty")
                        pass
                logger.debug("[RUN] Putting frame into input_queue")
                self.input_queue.put(frame)

                if self.ui_should_destroy:
                    self.ui_should_destroy = False
                    logger.debug("[RUN] UI should_destroy is True, destroying UI screen...")
                    cv2.destroyAllWindows()

                if self.show_ui_screen:
                    logger.debug("[RUN] In UI path.")
                    try:
                        logger.debug("[RUN] Waiting for result_queue.get()")
                        r_frame, r_tracks, r_fps, r_counts = self.result_queue.get(timeout=0.2)
                        logger.debug("[RUN] Got result from logic thread! Drawing and displaying...")
                        self.visualizer.draw_detections(r_frame, r_tracks)
                        self.visualizer.draw_stats(r_frame, r_counts)

                        cv2.putText(r_frame, f"FPS: {int(r_fps)}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        logger.debug("[RUN] Calling cv2.imshow")
                        r_frame = cv2.resize(r_frame, (1280, 720))
                        cv2.imshow("Bag Counter with DB", r_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.debug("[RUN] Quit key pressed, breaking.")
                            self.is_running = False  # Stop logic thread too!
                            cv2.destroyAllWindows()  # Ensure window closes right away
                            break
                    except queue.Empty:
                        logger.debug("[RUN] Result queue empty, skipping display")
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.debug("[RUN] Quit key pressed, breaking from empty loop.")
                            break
                else:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"[RUN ERROR] Exception in main loop: {e}")
        finally:
            logger.info("[RUN] Cleaning up...")
            self.is_running = False
            self.frame_source.cleanup()
            self.config_watcher.stop()
            cv2.destroyAllWindows()
            if logic_thread.is_alive():
                logic_thread.join()
            logger.info("[RUN] App Closed.")