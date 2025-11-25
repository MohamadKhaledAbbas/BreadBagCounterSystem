from abc import abstractmethod, ABC

import numpy as np


class BaseTracker(ABC):
    """Abstract base class for detection/tracking models."""

    @abstractmethod
    def update(self, detections_xyxy: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray):
        """Returns raw detections (boxes, conf, classes)"""
        pass
