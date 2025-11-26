import time
from typing import Dict, Any

import cv2

from src.counting.Visualizer import Visualizer
from src.classifier.AsyncClassificationService import AsyncClassificationService
from src.classifier.BaseClassifier import BaseClassifier
from src.counting.BagStateMonitor import BagStateMonitor
from src.detection.BaseDetection import BaseDetector
from src.logging.Database import DatabaseManager
from src.tracking.BaseTracker import BaseTracker
from src.utils.PerformanceChecker import run_with_timing
from src.utils.ThreadedCamera import ThreadedCamera


class BagCounterApp:
    def __init__(self,
                 video_path: str,
                 detector_engine: BaseDetector,
                 tracker: BaseTracker,
                 classifier_engine: BaseClassifier,
                 db_manager: DatabaseManager):

        self.db_manager = db_manager
        self.video_path = video_path
        self.detector = detector_engine
        self.tracker = tracker
        self.classifier_service = AsyncClassificationService(classifier_engine)

        self.show_ui_screen = False
        # ... (Rest of init remains the same) ...
        names = self.detector.class_names
        print(f'Found names: {names}')
        name_to_id = {v: k for k, v in names.items()}
        try:
            open_id = name_to_id['bread-bag-opened']
            closed_id = name_to_id['bread-bag-closed']
        except KeyError:
            raise ValueError("Model missing required classes")

        self.monitor = BagStateMonitor(open_id, closed_id)
        self.visualizer = Visualizer(names)

        # 3. CONNECT THEM via Callback
        # "When classification finishes, call self.handle_classification_result"
        self.classifier_service.register_callback(self.on_classification_result)
        self.ui_counts = {}

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        """
        This function is the Bridge.
        It takes pure data from the Service and applies Business Logic via the DB.
        """
        label = data['label']
        phash = data['phash']
        image_path = data['image_path']
        conf = data.get('confidence', 1.0)

        print(f"Received result for Track {track_id}: {label}")

        # 1. Resolve Bag Identity (The Magic Logic)
        # The DB decides: "Is this really unknown? Or is it unknown_bag_5?"
        bag_type_id = self.db_manager.get_or_create_bag_type(label, phash, image_path)

        # 2. Log the Event
        self.db_manager.log_event(bag_type_id, track_id, conf)

        # 3. Update UI Counters
        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1

    def run(self):
        # ... (Run loop logic remains largely the same) ...

        cap = run_with_timing("VideoCapture", ThreadedCamera,self.video_path)

        while cap.isOpened():
            t1 = cv2.getTickCount()
            success, frame = cap.read()
            if not success: break

            detections = run_with_timing("detector_predict", self.detector.predict, frame)

            if len(detections[0].boxes) > 0:
                xyxy = detections[0].boxes.xyxy.cpu().numpy()
                conf = detections[0].boxes.conf.cpu().numpy()
                cls_ids = detections[0].boxes.cls.cpu().numpy().astype(int)

                # B. TRACK
                tracks = run_with_timing("Tracker", self.tracker.update, xyxy, conf, cls_ids)

                active_ids = set()
                for det in tracks:
                    active_ids.add(det.track_id)
                    event = self.monitor.update(det.track_id, det.class_id)

                    if event == 'READY_TO_CLASSIFY':
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = map(int, det.box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            roi = frame[y1:y2, x1:x2].copy()
                            # Pass track_id so we can log it in DB
                            run_with_timing("Classify", self.classifier_service.process,det.track_id, roi)

                self.monitor.cleanup(active_ids)
                if self.show_ui_screen:
                    run_with_timing("draw_detection", self.visualizer.draw_detections, frame, tracks)
                    run_with_timing("draw_stats", self.visualizer.draw_stats, frame, self.ui_counts)


            t2 = cv2.getTickCount()
            latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
            fps = 1000 / latency if latency else 0
            print(f"FPS: {fps:.2f} ,  time_diff: {latency:.2f} ms")

            # Draw FPS on frame
            if self.show_ui_screen:
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.show_ui_screen:
                frame = cv2.resize(frame, (1280, 720))
                cv2.imshow("Bag Counter with DB", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.classifier_service.stop()
        cap.release()
        cv2.destroyAllWindows()
