from abc import ABC, abstractmethod
from typing import List, Dict

class BaseDetector(ABC):
    """Abstract base class for detection/tracking models."""

    @abstractmethod
    def predict(self, frame):
        """Returns raw detections (boxes, conf, classes)"""
        pass

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        pass