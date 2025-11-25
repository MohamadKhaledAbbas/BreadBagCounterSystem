from typing import List, Dict

from ultralytics import YOLO

from src.detection.BaseDetection import BaseDetector

class UltralyticsDetector(BaseDetector):
    """Concrete implementation using Ultralytics YOLO."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame):
        # verbose=False prevents console spam
        return self.model.predict(frame, verbose=False)

    @property
    def class_names(self) -> Dict[int, str]:
        return self.model.names