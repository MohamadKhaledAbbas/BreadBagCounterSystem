from typing import Dict, List, Union
import cv2
import numpy as np

from src.detection.TrackedObject import TrackedObject

class Visualizer:
    """Handles all drawing operations."""

    def __init__(self, class_names: Dict[int, str]):
        self.names = class_names

    @staticmethod
    def _compute_draw_params(box, frame_shape,
                            fixed_font_scale=None, fixed_thickness=None,
                            min_font_scale=0.7, max_font_scale=2.5, thickness_ratio=0.03):
        x1, y1, x2, y2 = map(int, box)
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        if fixed_thickness is not None:
            rect_thickness = max(2, int(fixed_thickness))
        else:
            rect_thickness = max(2, int(round(min(box_w, box_h) * thickness_ratio)))
        if fixed_font_scale is not None:
            font_scale = float(fixed_font_scale)
        else:
            # scale relative to box height; tune divisor as needed
            font_scale = max(min_font_scale, min(max_font_scale, box_h / 240.0))
        text_thickness = max(1, rect_thickness // 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize("Ag", font, font_scale, text_thickness)
        pad = max(2, int(round(0.35 * th)))
        return rect_thickness, font_scale, text_thickness, pad, baseline

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Union[TrackedObject, Dict]],
                        show_conf: bool = True,
                        fixed_font_scale: float = None,
                        fixed_thickness: int = None,
                        bg_label: bool = True):
        """
        Draw tracked detections or raw detection dicts with class-colored boxes and large, readable labels.
        Supports both TrackedObject and dict ({box, class_id, conf, track_id}) entries.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = frame.shape[:2]
        for det in detections:
            # Support either dict or object
            if isinstance(det, dict):
                box = det['box']
                class_id = det.get('class_id', None)
                conf = det.get('conf', None)
                track_id = det.get('track_id', None)
            else:
                box = det.box
                class_id = getattr(det, "class_id", None)
                conf = getattr(det, "conf", None)
                track_id = getattr(det, "track_id", None)
            x1, y1, x2, y2 = map(int, box)
            thickness, font_scale, text_thickness, pad, baseline = self._compute_draw_params(
                box, frame.shape, fixed_font_scale, fixed_thickness
            )
            # Color per class (customizable)
            if class_id == 0: color = (255, 0, 0)
            elif class_id == 1: color = (0, 255, 0)
            else: color = (0, 0, 255)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Build label
            name = self.names.get(class_id, "Unknown")
            if track_id is not None:
                main_label = f"{track_id}: {name}"
            else:
                main_label = name
            if show_conf and conf is not None:
                label = f"{main_label} {conf:.2f}"
            else:
                label = main_label
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            label_x1 = x1
            label_x2 = x1 + text_w + 2 * pad
            label_y2 = y1 - 5
            label_y1 = label_y2 - (text_h + 2 * pad)
            # If not enough space, place below
            if label_y1 < 0:
                label_y1 = y2 + 5
                label_y2 = label_y1 + (text_h + 2 * pad)
                if label_y2 > h:  # Clamp inside frame
                    label_y1 = max(2, h - (text_h + 2 * pad) - 2)
                    label_y2 = label_y1 + (text_h + 2 * pad)
            # Filled background for label
            if bg_label:
                cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
            else:
                cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, max(1, thickness // 2))
            # Contrast-aware text color
            b, g, r = color
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            text_x = label_x1 + pad
            text_y = label_y2 - pad - (baseline // 2)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    def draw_active_events(self, frame: np.ndarray,
                          active_events: List,
                          fixed_font_scale: float = None,
                          fixed_thickness: int = None):
        """
        Draw active bag events (ID and state) with consistent formatting near the box.
        Expects active_events as a list of objects with .id, .state, .box
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = frame.shape[:2]
        for event in active_events:
            x1, y1, x2, y2 = map(int, event.box)
            thickness, font_scale, text_thickness, pad, baseline = self._compute_draw_params(
                event.box, frame.shape, fixed_font_scale, fixed_thickness
            )

            color = (0, 255, 255)  # Cyan/yellow for events
            event_label = f"ID:{event.id} {event.state}"
            (text_w, text_h), _ = cv2.getTextSize(event_label, font, font_scale, text_thickness)
            label_x1 = x1
            label_x2 = x1 + text_w + 2 * pad
            label_y1 = y2 + 5
            label_y2 = label_y1 + (text_h + 2 * pad)
            # Clamp if off-screen
            if label_y2 > h:
                label_y2 = y1 - 5
                label_y1 = label_y2 - (text_h + 2 * pad)
                if label_y1 < 0:
                    label_y1 = max(2, h - (text_h + 2 * pad) - 2)
                    label_y2 = label_y1 + (text_h + 2 * pad)
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
            b, g, r = color
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            text_x = label_x1 + pad
            text_y = label_y2 - pad - (baseline // 2)
            cv2.putText(frame, event_label, (text_x, text_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    def draw_stats(self, frame: np.ndarray, counts: Dict[str, int]):
        y = 60
        cv2.putText(frame, "Counts:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        for label, count in counts.items():
            y += 70
            cv2.putText(frame, f"{label}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def draw_fps(self, frame: np.ndarray, fps: float):
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    def render_all(self, frame: np.ndarray,
                   detections: List[Union[TrackedObject, Dict]],
                   active_events: List,
                   counts: Dict[str, int] = None,
                   fps: float = None):
        """
        Full pass: draws detections, events, stats, fps in one call.
        """
        if detections:
            self.draw_detections(frame, detections)
        if active_events:
            self.draw_active_events(frame, active_events)
        if counts is not None:
            self.draw_stats(frame, counts)
        if fps is not None:
            self.draw_fps(frame, fps)