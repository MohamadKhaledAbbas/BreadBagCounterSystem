import cv2

from src.counting.Visualizer import Visualizer
from src.classifier.AsyncClassificationService import AsyncClassificationService
from src.classifier.BaseClassifier import BaseClassifier
from src.counting.BagStateMonitor import BagStateMonitor
from src.detection.BaseDetection import BaseDetector
from src.logging.Database import DatabaseManager


class BagCounterApp:
    def __init__(self,
                 video_path: str,
                 tracker_engine: BaseDetector,
                 classifier_engine: BaseClassifier,
                 db_manager: DatabaseManager,  # New Parameter
                 tracker_config: str):

        self.video_path = video_path
        self.detector = tracker_engine
        # Pass DB manager to the service
        self.classifier_service = AsyncClassificationService(classifier_engine, db_manager)
        self.tracker_config = tracker_config

        # ... (Rest of init remains the same) ...
        names = self.detector.class_names
        name_to_id = {v: k for k, v in names.items()}
        try:
            open_id = name_to_id['bread-bag-opened']
            closed_id = name_to_id['bread-bag-closed']
        except KeyError:
            raise ValueError("Model missing required classes")

        self.monitor = BagStateMonitor(open_id, closed_id)
        self.visualizer = Visualizer(names)
        self.db_manager = DatabaseManager("bag_events.db")
        # 3. CONNECT THEM via Callback
        # "When classification finishes, call self.handle_classification_result"
        self.classifier_service.register_callback(self.handle_classification_result)
        self.ui_counts = {}

    def handle_classification_result(self, track_id: int, label: str):
        """
        This function is the ONLY place where Logic meets Persistence.
        It runs whenever the service finishes a task.
        """
        print(f"Event: Track {track_id} -> {label}")

        # Action A: Update UI Counters
        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1

        # Action B: Log to Database
        self.db_manager.log_event(track_id, label)
        # print("Logging to Database later")

    def run(self):
        # ... (Run loop logic remains largely the same) ...
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            detections = self.detector.track(frame, self.tracker_config)

            active_ids = set()
            for det in detections:
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
                        self.classifier_service.process(det.track_id, roi)

            self.monitor.cleanup(active_ids)
            self.visualizer.draw_detections(frame, detections)
            self.visualizer.draw_stats(frame, self.ui_counts)

            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("Bag Counter with DB", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.classifier_service.stop()
        cap.release()
        cv2.destroyAllWindows()
