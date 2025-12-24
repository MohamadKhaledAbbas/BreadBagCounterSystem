from typing import Dict, List, Union
import cv2
import numpy as np

from src.detection.TrackedObject import TrackedObject

# State colors for event-centric tracking visualization
STATE_COLORS = {
    'detecting_open': (0, 255, 0),      # Green - bag is open
    'detecting_closed': (0, 165, 255),  # Orange - transitioning to closed
    'OPEN': (0, 255, 0),                # Green - bag is open
    'CLOSING': (0, 165, 255),           # Orange - transitioning to closed  
    'CLOSED': (0, 255, 255),            # Yellow - bag is closed, collecting ROIs
    'COMMITTED': (255, 0, 255),         # Magenta - counted
    'counted': (255, 0, 255),           # Magenta - counted (legacy)
}

class Visualizer:
    """Handles all drawing operations."""

    def __init__(self, class_names: Dict[int, str]):
        self.names = class_names
        self.exit_margin = 50  # Default exit boundary margin
        
        # V3 Performance: Cache for exit boundary overlay
        self._exit_boundary_cache = None
        self._exit_boundary_cache_shape = None
        
        # V3 Performance: Frame counter for conditional rendering
        self._frame_counter = 0
        self._legend_redraw_interval = 30  # Redraw legend every N frames
        self._last_state_hash = None  # Track state changes for legend updates
        
        # Phase 1 Optimization: Cache legend as static image
        self._legend_cache = None
        self._legend_cache_size = None  # (width, height) of frame when legend was created

    def set_exit_margin(self, margin: int):
        """Set the exit boundary margin for visualization."""
        self.exit_margin = margin

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
        Draw active bag events with state-colored boxes, centroids, and detailed info.
        Expects active_events as a list of objects with .id, .state, .box, and optionally
        .open_hits, .closed_hits, .last_centroid, .roi_count
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = frame.shape[:2]
        
        for event in active_events:
            x1, y1, x2, y2 = map(int, event.box)
            thickness, font_scale, text_thickness, pad, baseline = self._compute_draw_params(
                event.box, frame.shape, fixed_font_scale, fixed_thickness
            )

            # Get state-specific color
            state_str = str(event.state) if not isinstance(event.state, str) else event.state
            color = STATE_COLORS.get(state_str, (0, 255, 255))  # Default cyan
            
            # Draw bounding box with state color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness + 1)
            
            # Draw centroid marker
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if hasattr(event, 'last_centroid') and event.last_centroid:
                cx, cy = int(event.last_centroid[0]), int(event.last_centroid[1])
            cv2.circle(frame, (cx, cy), 8, color, -1)  # Filled circle at centroid
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)  # White outline
            
            # Build detailed label
            event_id_short = event.id % 10000  # Last 4 digits for readability
            event_label = f"E{event_id_short}: {state_str}"
            
            # Add open/closed counts if available
            open_count = getattr(event, 'open_hits', 0)
            closed_count = getattr(event, 'closed_hits', 0)
            if open_count or closed_count:
                event_label += f" O:{open_count} C:{closed_count}"
            
            # Add ROI count if available and in CLOSED state
            roi_count = getattr(event, 'roi_count', 0)
            if roi_count and ('CLOSED' in state_str.upper() or 'OPEN' in state_str.upper()):
                event_label += f" ROI:{roi_count}"
            
            (text_w, text_h), _ = cv2.getTextSize(event_label, font, font_scale * 0.8, text_thickness)
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
            cv2.putText(frame, event_label, (text_x, text_y), font, font_scale * 0.8, text_color, text_thickness, cv2.LINE_AA)

    def draw_exit_boundary(self, frame: np.ndarray, margin: int = None):
        """
        Draw the exit boundary zone around the frame edges.
        Bags must reach this zone to be counted.
        
        V3 Performance: Uses caching to avoid recreating overlay every frame.
        Only redraws when frame size changes.
        
        Args:
            frame: Frame to draw on
            margin: Exit boundary margin in pixels (uses self.exit_margin if None)
        """
        h, w = frame.shape[:2]
        h = 650
        if margin is None:
            margin = self.exit_margin
        
        current_shape = (h, w, margin)
        
        # V3 Performance: Check if we need to regenerate the cache
        if self._exit_boundary_cache is None or self._exit_boundary_cache_shape != current_shape:
            # Create new cached overlay
            self._exit_boundary_cache = np.zeros((h, w, 3), dtype=np.uint8)
            exit_color = (0, 100, 0)  # Dark green for exit zone
            
            # Top edge
            cv2.rectangle(self._exit_boundary_cache, (0, 0), (w, margin), exit_color, -1)
            # Bottom edge
            cv2.rectangle(self._exit_boundary_cache, (0, h - margin), (w, h), exit_color, -1)
            # Left edge
            cv2.rectangle(self._exit_boundary_cache, (0, 0), (margin, h), exit_color, -1)
            # Right edge
            cv2.rectangle(self._exit_boundary_cache, (w - margin, 0), (w, h), exit_color, -1)
            
            self._exit_boundary_cache_shape = current_shape
        
        # Blend cached overlay with frame (in-place to avoid extra copy)
        cv2.addWeighted(self._exit_boundary_cache, 0.2, frame, 0.8, 0, frame)
        
        # Draw boundary lines (these are cheap)
        line_color = (0, 200, 0)  # Brighter green for lines
        thickness = 2
        # Inner rectangle showing work zone
        cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), line_color, thickness)
        
        # Add label
        cv2.putText(frame, "EXIT ZONE", (margin + 5, margin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)

    def draw_stats(self, frame: np.ndarray, counts: Dict[str, int]):
        y = 60
        cv2.putText(frame, "Counts:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        for label, count in counts.items():
            y += 70
            cv2.putText(frame, f"{label}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def draw_fps(self, frame: np.ndarray, fps: float):
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    def draw_state_legend(self, frame: np.ndarray):
        """
        Draw a legend showing what each state color means.
        Phase 1 Optimization: Returns pre-rendered legend image for blitting.
        """
        h, w = frame.shape[:2]
        
        # Check if we need to regenerate the legend (frame size changed)
        if self._legend_cache is None or self._legend_cache_size != (w, h):
            # Create legend as a separate image
            legend_w, legend_h = 220, 150
            self._legend_cache = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
            
            legend_x = 0
            legend_y = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_height = 25
            
            # Background for legend
            cv2.rectangle(self._legend_cache, (legend_x, legend_y - 20), 
                         (legend_w, legend_y + len(STATE_COLORS) * line_height), 
                         (40, 40, 40), -1)
            
            cv2.putText(self._legend_cache, "Event States:", (legend_x, legend_y), 
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            for i, (state, color) in enumerate(STATE_COLORS.items()):
                if state in ['detecting_open', 'detecting_closed', 'counted']:  # Skip legacy names
                    continue
                y = legend_y + (i + 1) * line_height
                # Draw color box
                cv2.rectangle(self._legend_cache, (legend_x, y - 12), (legend_x + 15, y + 3), color, -1)
                # Draw state name
                cv2.putText(self._legend_cache, state, (legend_x + 20, y), 
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            self._legend_cache_size = (w, h)
        
        # Blit cached legend onto frame (fast operation)
        legend_h, legend_w = self._legend_cache.shape[:2]
        x_pos = w - legend_w - 10
        y_pos = 30
        
        # Ensure we don't go out of bounds
        if x_pos >= 0 and y_pos + legend_h <= h:
            frame[y_pos:y_pos+legend_h, x_pos:x_pos+legend_w] = self._legend_cache

    def render_all(self, frame: np.ndarray,
                   detections: List[Union[TrackedObject, Dict]],
                   active_events: List,
                   counts: Dict[str, int] = None,
                   fps: float = None,
                   show_exit_boundary: bool = True,
                   show_legend: bool = True):
        """
        Full pass: draws detections, events, stats, fps, and visual guides in one call.
        
        V3 Performance: Legend is only redrawn every 30 frames or when state changes.
        
        Args:
            frame: Frame to draw on
            detections: List of detections to draw
            active_events: List of active events to draw
            counts: Dictionary of counts per class
            fps: Current FPS to display
            show_exit_boundary: Whether to show the exit boundary zone
            show_legend: Whether to show the state color legend
        """
        # Increment frame counter
        self._frame_counter += 1
        
        # Draw exit boundary first (background layer)
        if show_exit_boundary:
            self.draw_exit_boundary(frame)
        
        if detections:
            self.draw_detections(frame, detections)
        if active_events:
            self.draw_active_events(frame, active_events)
        if counts is not None:
            self.draw_stats(frame, counts)
        if fps is not None:
            self.draw_fps(frame, fps)
        
        # V3 Performance: Draw legend conditionally
        # - Every 30th frame (to reduce overhead)
        # - Or when active event states change (detected via efficient hash)
        if show_legend:
            # Efficient state change detection: count and hash of state values
            # Guard against missing 'state' attribute
            try:
                states = tuple(e.state for e in active_events if hasattr(e, 'state'))
                current_state_hash = (len(active_events), hash(states))
            except (TypeError, AttributeError):
                # Fallback if hashing fails
                current_state_hash = (len(active_events), 0)
            
            state_changed = current_state_hash != self._last_state_hash
            
            # Redraw on every Nth frame or when state changes (skip frame 0)
            should_redraw = (
                (self._frame_counter > 0 and self._frame_counter % self._legend_redraw_interval == 0) or 
                state_changed
            )
            
            if should_redraw:
                self.draw_state_legend(frame)
                self._last_state_hash = current_state_hash