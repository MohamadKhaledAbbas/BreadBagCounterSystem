from src.classifier.BpuClassifyer import BpuClassifier
from src.counting.BagCounterApp import BagCounterApp
from src.detection.BpuDetector import BpuDetector
from src.logging.Database import DatabaseManager
from src.frame_source.FrameSourceFactory import FrameSourceFactory
from src.tracking.Tracker import ObjectTracker

if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "output2.mp4"
    DETECTION_MODEL = "data/model/best_detect_bayese_640x640_nv12.bin"
    CLASS_MODEL = "data/model/best_classify_bayese_224x224_nv12.bin"
    IS_PRODUCTION_MODE = False

    db_manager = DatabaseManager("data/db/bag_events.db")

    # Dependency Injection: We inject the concrete "Ultralytics" engines here.
    # Later, if you use RDK X5, you only change these two lines to "RDKX5Detector".
    detector = BpuDetector(DETECTION_MODEL, {0: 'bread-bag-closed', 1: 'bread-bag-opened'})
    tracker = ObjectTracker()
    classifier = BpuClassifier(CLASS_MODEL, { 0: 'Blue-Bag', 1: 'Brown-Bag', 2: 'Dark-Brown-Bag',
                                   3: 'Green-Bag', 4: 'Red-Bag', 5: 'Yellow-Bag'})

    if IS_PRODUCTION_MODE:
        frame_source = FrameSourceFactory.create("ros2")
    else:
        frame_source = FrameSourceFactory.create("opencv", source=VIDEO_PATH)

    app = BagCounterApp(
        video_path=VIDEO_PATH,
        detector_engine=detector,
        tracker=tracker,
        classifier_engine=classifier,
        db_manager=db_manager,
        frame_source=frame_source,
        show_ui_screen=True
    )
    app.run()