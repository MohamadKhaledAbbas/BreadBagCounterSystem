import numpy as np
import supervision as sv

from src.detection.TrackedObject import TrackedObject
from src.tracking.BaseTracker import BaseTracker

class ObjectTracker(BaseTracker):
    """
    Independent Tracker Wrapper.
    Does NOT rely on model.track(). It expects detections and returns tracked objects.
    """
    def __init__(self):
        # Using ByteTrack from supervision library (State of the Art)
        self.tracker = sv.ByteTrack()

    def update(self, detections_xyxy: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray):
        """
        Args:
            detections_xyxy: Array of shape (N, 4)
            confidences: Array of shape (N,)
            class_ids: Array of shape (N,)
        Returns:
            List of tracked objects with (track_id, x1, y1, x2, y2, class_id)
        """

        # Convert to supervision Detections format
        detections = sv.Detections(
            xyxy=detections_xyxy,
            confidence=confidences,
            class_id=class_ids
        )

        # Update tracker
        tracked_detections = self.tracker.update_with_detections(detections)

        # Format output as list of standard objects
        results = []
        if tracked_detections.tracker_id is not None:
            for xyxy, track_id, class_id in zip(
                    tracked_detections.xyxy,
                    tracked_detections.tracker_id,
                    tracked_detections.class_id
            ):
                results.append(TrackedObject(
                    track_id= int(track_id),
                    box= xyxy.tolist(),
                    class_id= int(class_id)
                ))
        return results