from typing import Any

from ultralytics import YOLO

from src.classifier.BaseClassifier import BaseClassifier


class UltralyticsClassifier(BaseClassifier):
    """Concrete implementation using Ultralytics YOLO for classification."""
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def load(self, model_path: str):
        pass

    def predict(self, image):
        results = self.model(image, verbose=False)

        label = "Unknown"
        conf = 0.0

        # Case A: It is a Classification Model (has .probs)
        if results[0].probs is not None:
            top1_idx = results[0].probs.top1
            conf = float(results[0].probs.top1conf)  # Extract confidence
            label = results[0].names[top1_idx]

        # Case B: It is a Detection Model used as Classifier (has .boxes)
        elif len(results[0].boxes) > 0:
            # Take the highest confidence box
            conf = float(results[0].boxes.conf[0])
            cls_id = int(results[0].boxes.cls[0])
            label = results[0].names[cls_id]

        return label, conf