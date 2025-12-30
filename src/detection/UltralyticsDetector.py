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

    def predict_batch(self, frames):
        """
        V4: Batch inference for multiple frames using Ultralytics YOLO.
        
        Ultralytics YOLO natively supports batch inference by passing a list of frames.
        
        Args:
            frames: List of numpy arrays (frames) to process
            
        Returns:
            List of result objects, one per frame
        """
        if len(frames) == 0:
            return []
        
        if len(frames) == 1:
            return self.predict(frames[0])
        
        # Ultralytics YOLO supports passing a list of images for batch inference
        # This leverages the underlying batch processing capabilities
        results = self.model.predict(frames, verbose=False)
        
        return results