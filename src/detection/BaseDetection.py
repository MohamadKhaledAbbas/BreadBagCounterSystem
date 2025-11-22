from abc import ABC, abstractmethod
from typing import List, Dict
from src.detection.DetectionResult import DetectionResult


class BaseDetector(ABC):
    """Abstract base class for detection/tracking models."""
    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def track(self, frame, config=None) -> List[DetectionResult]:
        pass

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        pass