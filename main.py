from src.counting.BagCounterApp import BagCounterApp
from src.classifier.UltralyticsClassifier import UltralyticsClassifier
from src.detection.UltralyticsDetector import UltralyticsDetector

if __name__ == "__main__":
    # Configuration
    VIDEO = "D:\\Recordings\\2025_11_08\\20251108021447_20251108030005.mp4"
    TRACK_MODEL = "best_detect.pt"
    CLASS_MODEL = "best_classify.pt"
    TRACK_CFG = "custom_bytetrack.yaml"

    # Dependency Injection: We inject the concrete "Ultralytics" engines here.
    # Later, if you use RDK X5, you only change these two lines to "RDKX5Detector".
    tracker = UltralyticsDetector(TRACK_MODEL)
    classifier = UltralyticsClassifier(CLASS_MODEL)

    app = BagCounterApp(VIDEO, tracker, classifier, TRACK_CFG)
    app.run()