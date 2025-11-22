import cv2

from src.counting.Visualizer import Visualizer
from src.classifier.AsyncClassificationService import AsyncClassificationService
from src.classifier.BaseClassifier import BaseClassifier
from src.counting.BagStateMonitor import BagStateMonitor
from src.detection.BaseDetection import BaseDetector


class BagCounterApp:
    def __init__(self,
                 video_path: str,
                 tracker_engine: BaseDetector,
                 classifier_engine: BaseClassifier,
                 tracker_config: str):

        self.video_path = video_path
        self.detector = tracker_engine
        self.classifier_service = AsyncClassificationService(classifier_engine)
        self.tracker_config = tracker_config

        # Setup Logic
        # We dynamically find IDs based on names provided by the engine
        names = self.detector.class_names
        name_to_id = {v: k for k, v in names.items()}

        try:
            open_id = name_to_id['bread-bag-opened']
            closed_id = name_to_id['bread-bag-closed']
        except KeyError:
            raise ValueError("Model missing required classes: bread-bag-opened/closed")

        self.monitor = BagStateMonitor(open_id, closed_id)
        self.visualizer = Visualizer(names)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 1. Detect & Track
            detections = self.detector.track(frame, self.tracker_config)

            # 2. Update Logic & Handle Events
            active_ids = set()
            for det in detections:
                active_ids.add(det.track_id)

                event = self.monitor.update(det.track_id, det.class_id)

                if event == 'READY_TO_CLASSIFY':
                    # Extract ROI safely
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = map(int, det.box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2].copy()
                        self.classifier_service.process(roi)
                        print(f"Processing Track {det.track_id}...")

            self.monitor.cleanup(active_ids)

            # 3. Visualize
            self.visualizer.draw_detections(frame, detections)
            self.visualizer.draw_stats(frame, self.classifier_service.get_counts())

            # 4. Display
            frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
            cv2.imshow("Modular Bag Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.classifier_service.stop()
        cap.release()
        cv2.destroyAllWindows()
