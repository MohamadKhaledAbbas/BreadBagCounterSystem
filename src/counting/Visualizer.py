from typing import Dict, List

import cv2

from src.detection.TrackedObject import TrackedObject


class Visualizer:
    """Handles all drawing operations."""
    def __init__(self, class_names: Dict[int, str]):
        self.names = class_names

    def draw_detections(self, frame, tracks: List[TrackedObject]):
        for det in tracks:
            x1, y1, x2, y2 = map(int, det.box)
            label = f"{det.track_id}: {self.names.get(det.class_id, 'Unknown')}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    def draw_stats(self, frame, counts: Dict[str, int]):
        y = 60
        cv2.putText(frame, "Counts:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        for label, count in counts.items():
            y += 70
            cv2.putText(frame, f"{label}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
