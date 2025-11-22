from typing import List


class DetectionResult:
    """Standardized output for any detection model."""
    def __init__(self, track_id: int, class_id: int, box: List[float]):
        self.track_id = track_id
        self.class_id = class_id
        self.box = box # [x1, y1, x2, y2]