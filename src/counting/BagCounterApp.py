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
from src.tracking.BaseTracker import BaseTracker
from src.utils.PerformanceChecker import run_with_timing
from src.utils.ThreadedCamera import ThreadedCamera


class BagCounterApp:
    def __init__(self,
                 video_path: str,
                 detector_engine: BaseDetector,
                 tracker: BaseTracker,
                 classifier_engine: BaseClassifier,
                 db_manager: DatabaseManager,
                 show_ui_screen: bool = False):

        self.db_manager = db_manager
        self.video_path = video_path
        self.detector = detector_engine
        self.tracker = tracker
        self.classifier_service = ClassifierService(classifier_engine)

        # --- NEW: Threading Setup ---
        self.show_ui_screen = show_ui_screen
        self.is_running = False

        # 1. Input Queue: Frames from Camera -> Logic Thread
        self.input_queue = queue.Queue(maxsize=1)
        # 2. Result Queue: Processed Data -> Main Thread (Display)
        self.result_queue = queue.Queue(maxsize=1)

        # ... (Rest of init logic) ...
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
        self.classifier_service.register_callback(self.on_classification_result)
        self.ui_counts = {}

    def on_classification_result(self, track_id: int, data: Dict[str, Any]):
        label = data['label']
        phash = data['phash']
        image_path = data['image_path']
        conf = data.get('confidence', 1.0)
        print(f"Received result for Track {track_id}: {label}")
        bag_type_id = self.db_manager.get_or_create_bag_type(label, phash, image_path)
        self.db_manager.log_event(bag_type_id, track_id, conf)
        self.ui_counts[label] = self.ui_counts.get(label, 0) + 1

    # --- NEW: The Logic Worker ---
    def _logic_thread_loop(self):
        print("✅ Logic Thread Started")
        while self.is_running:
            try:
                # 1. Get Frame (Blocks here until camera sends data)
                # This prevents CPU spinning!
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 2. Run Full Logic
            t1 = cv2.getTickCount()
            detections = self.detector.predict(frame)

            tracks = []
            if len(detections[0].boxes) > 0:
                xyxy = detections[0].boxes.xyxy.cpu().numpy()
                conf = detections[0].boxes.conf.cpu().numpy()
                cls_ids = detections[0].boxes.cls.cpu().numpy().astype(int)

                tracks = self.tracker.update(xyxy, conf, cls_ids)

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
                            self.classifier_service.process(det.track_id, roi)
                self.monitor.cleanup(active_ids)

            t2 = cv2.getTickCount()
            latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
            fps = 1000 / latency if latency else 0

            # 3. Send to Display (Non-blocking 'Leaky Bucket')
            if self.show_ui_screen:
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Send tuple: (Original Frame, Track Results, FPS, UI Counts)
                self.result_queue.put((frame, tracks, fps, self.ui_counts.copy()))
            else:
                print(f"fps: {int(fps)} , latency: {latency:.2f} ms")
                # Just print status if no UI
                # print(f"Logic FPS: {fps:.2f}")
                pass

    def run(self):
        cap = ThreadedCamera(self.video_path)
        self.is_running = True

        # 1. Start the Logic Thread
        logic_thread = threading.Thread(target=self._logic_thread_loop, daemon=True)
        logic_thread.start()

        print("✅ Main Loop Started (Display & Input)")

        try:
            while cap.isOpened():
                # A. Get Camera Frame
                # Use blocking read to sync with camera speed
                success, frame = cap.read()
                if not success: break

                # B. Feed Logic Thread (Leaky Bucket)
                # If logic is slow, drop this frame so we don't lag
                if self.input_queue.full():
                    try:
                        self.input_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.input_queue.put(frame)


                # C. Display Handling (Main Thread)
                if self.show_ui_screen:
                    try:
                        r_frame, r_tracks, r_fps, r_counts = self.result_queue.get(timeout=0.005)
                        # Draw on the frame we just got back from logic
                        self.visualizer.draw_detections(r_frame, r_tracks)
                        self.visualizer.draw_stats(r_frame, r_counts)

                        cv2.putText(r_frame, f"FPS: {int(r_fps)}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        r_frame = cv2.resize(r_frame, (1280, 720))
                        cv2.imshow("Bag Counter with DB", r_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except queue.Empty:
                        # No result ready yet? That's fine.
                        # Just sleep a tiny bit to let logic run
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    # NO UI MODE:
                    # Vital Sleep. Since we aren't doing 'imshow', we must sleep
                    # to let the Logic Thread utilize the CPU.
                    time.sleep(0.1)

        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            print("App Closed.")