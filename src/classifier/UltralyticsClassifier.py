from ultralytics import YOLO

from src.classifier.BaseClassifier import BaseClassifier


class UltralyticsClassifier(BaseClassifier):
    """Concrete implementation using Ultralytics YOLO for classification."""
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def load(self, model_path: str):
        pass

    def predict(self, image) -> str:
        results = self.model(image, verbose=False)
        # Check if it's a classification model (probs) or detection model used as classifier (boxes)
        if results[0].probs is not None:
            return results[0].names[results[0].probs.top1]
        elif len(results[0].boxes) > 0:
            # Fallback if using a detection model for classification
            top_cls = int(results[0].boxes.cls[0])
            return results[0].names[top_cls]
        return "Unknown"