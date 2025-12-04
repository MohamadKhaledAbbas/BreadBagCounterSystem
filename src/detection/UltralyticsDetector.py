from typing import List, Dict

from ultralytics import YOLO

from src.detection.BaseDetection import BaseDetector

class UltralyticsDetector(BaseDetector):
    """Concrete implementation using Ultralytics YOLO."""

    def __init__(self, model_path: str, class_names: Dict[int, str] = None):
        """Initialize Ultralytics YOLO detector.
        
        Args:
            model_path: Path to YOLO model (.pt, .onnx, or .engine)
            class_names: Optional custom class names. If not provided, uses model's built-in names.
        """
        self.model = YOLO(model_path)
        self._class_names = class_names

    def predict(self, frame):
        # verbose=False prevents console spam
        return self.model.predict(frame, verbose=False)

    @property
    def class_names(self) -> Dict[int, str]:
        # Use custom class names if provided, otherwise use model's names
        return self._class_names if self._class_names is not None else self.model.names