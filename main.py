from src.counting.BagCounterApp import BagCounterApp
from src.classifier.UltralyticsClassifier import UltralyticsClassifier
from src.detection.UltralyticsDetector import UltralyticsDetector
from src.logging.Database import DatabaseManager

if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "D:\\Recordings\\2025_11_08\\كعك_ch7_main_20251108070005_20251108080005.mp4"
    TRACK_MODEL = "best_detect.pt"
    CLASS_MODEL = "best_classify.pt"
    TRACK_CFG = "custom_bytetrack.yaml"

    db_manager = DatabaseManager("bag_events.db")

    # Dependency Injection: We inject the concrete "Ultralytics" engines here.
    # Later, if you use RDK X5, you only change these two lines to "RDKX5Detector".
    tracker = UltralyticsDetector(TRACK_MODEL)
    classifier = UltralyticsClassifier(CLASS_MODEL)

    app = BagCounterApp(
        video_path=VIDEO_PATH,
        tracker_engine=tracker,
        classifier_engine=classifier,
        db_manager=db_manager,
        tracker_config="custom_bytetrack.yaml"
    )
    app.run()