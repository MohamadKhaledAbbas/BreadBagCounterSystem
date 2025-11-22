from typing import List, Dict

from ultralytics import YOLO

from src.detection.BaseDetection import BaseDetector
from src.detection.DetectionResult import DetectionResult


class UltralyticsDetector(BaseDetector):
    """Concrete implementation using Ultralytics YOLO."""
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def load(self, model_path: str):
        # Already loaded in init for YOLO, but useful for other SDKs
        pass

    def track(self, frame, config=None) -> List[DetectionResult]:
        # Using verbose=False for performance
        results = self.model.track(frame, persist=True, tracker=config, verbose=False)
        detections = []

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()

            for i, c, b in zip(ids, clss, boxes):
                detections.append(DetectionResult(i, c, b))

        return detections

    @property
    def class_names(self) -> Dict[int, str]:
        return self.model.names