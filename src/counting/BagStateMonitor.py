import cv2
import uuid
import math
import numpy as np
from typing import List, Tuple, Optional

from src.utils.AppLogging import logger, structured_logger
from src.config.tracking_config import tracking_config
from src.utils.PipelineMetrics import pipeline_metrics


class BagEvent:
    """
    Represents a tracked bag event through its lifecycle.
    
    State Machine: detecting_open -> detecting_closed -> counted
    
    V4 Enhancements:
    - ROI collection deferred to track end for classification
    - Frame index and temporal metadata tracked for each ROI
    - Sharpness-based candidate selection at track end
    - Evidence-based classification (replaces entropy/EMA/Dirichlet)
    """
    
    # V2: Temporal smoothing parameters
    BOX_SMOOTHING_ALPHA = 0.3  # EMA smoothing factor (lower = more smoothing)
    
    # V2: Aspect ratio validation
    MIN_ASPECT_RATIO = 0.3  # Minimum width/height ratio
    MAX_ASPECT_RATIO = 3.0  # Maximum width/height ratio
    
    def __init__(self, box, frame_img, open_id, closed_id, frame_index: int = 0):
        self.id = int(uuid.uuid4().int >> 96)
        self.box = np.array(box, dtype=float)  # V2: Use numpy for smoothing
        self.smoothed_box = self.box.copy()  # V2: Smoothed box for stable tracking
        self.previous_box = box  # For motion tracking
        
        # State Machine: detecting_open -> detecting_closed -> counted
        self.state = 'detecting_open'

        # Threshold Counters
        self.open_hits = 1
        self.closed_hits = 0

        self.frames_since_update = 0
        self.total_frames_tracked = 1  # Track lifetime
        
        # V4: Track lifecycle timestamps
        self.start_frame_index = frame_index
        self.current_frame_index = frame_index

        # Buffer settings (from centralized config)
        self.max_open_samples = tracking_config.max_open_samples
        self.max_closed_samples = tracking_config.max_closed_samples

        self.open_id = open_id
        self.closed_id = closed_id

    # V4: Enhanced ROI storage with full metadata
        # Each entry: (sharpness, roi, frame_index, bbox_area, confidence)
        # roi is np.ndarray (BGR image)
        self.open_rois: List[Tuple[float, np.ndarray, int, float, float]] = []
        self.closed_rois: List[Tuple[float, np.ndarray, int, float, float]] = []
        
        # Motion tracking for adaptive suppression
        self.motion_history: List[float] = []  # Recent motion magnitudes
        self.max_motion_history = 10
        
        # V2: Confidence tracking for detection quality
        self.confidence_history: List[float] = []
        self.max_confidence_history = 10

        # Add first frame
        self._add_roi(box, frame_img, is_open=True, frame_index=frame_index, confidence=1.0)
        
        # Record metrics
        pipeline_metrics.record_event_created()

        # Structured logging for event creation
        try:
            box_list = box.tolist() if hasattr(box, 'tolist') else (list(box) if box is not None else [0, 0, 0, 0])
        except (TypeError, AttributeError):
            box_list = [0, 0, 0, 0]
        
        structured_logger.event_created(
            event_id=self.id,
            confidence=confidence,
            box=box_list,
            frame_index=frame_index,
            state=self.state
        )
    
    def _calculate_motion(self, new_box) -> float:
        """Calculate motion magnitude between current and new box."""
        if self.previous_box is None:
            return 0.0
        
        # Calculate center displacement
        old_cx = (self.previous_box[0] + self.previous_box[2]) / 2
        old_cy = (self.previous_box[1] + self.previous_box[3]) / 2
        new_cx = (new_box[0] + new_box[2]) / 2
        new_cy = (new_box[1] + new_box[3]) / 2
        
        motion = math.sqrt((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2)
        return motion
    
    def _smooth_box(self, new_box) -> np.ndarray:
        """
        V2: Apply temporal smoothing to bounding box coordinates.
        
        Uses exponential moving average (EMA) to reduce detection jitter
        and stabilize bounding box positions across frames.
        
        Returns:
            np.ndarray: Smoothed bounding box [x1, y1, x2, y2]
        """
        new_box_array = np.array(new_box, dtype=float)
        
        # EMA smoothing: smoothed = alpha * new + (1 - alpha) * old
        self.smoothed_box = (
            self.BOX_SMOOTHING_ALPHA * new_box_array + 
            (1 - self.BOX_SMOOTHING_ALPHA) * self.smoothed_box
        )
        
        return self.smoothed_box
    
    def _validate_aspect_ratio(self, box) -> bool:
        """
        V2: Validate that bounding box has a reasonable aspect ratio.
        
        Rejects boxes that are too narrow or too wide, which likely
        indicate detection errors.
        
        Returns:
            bool: True if aspect ratio is valid
        """
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        if height <= 0:
            return False
        
        aspect_ratio = width / height
        
        is_valid = self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO
        
        if not is_valid:
            logger.debug(
                f"[BagEvent:{self.id}] Invalid aspect ratio: {aspect_ratio:.2f} "
                f"(valid range: [{self.MIN_ASPECT_RATIO}, {self.MAX_ASPECT_RATIO}])"
            )
        
        return is_valid
    
    def _update_motion(self, new_box, confidence: float = 1.0):
        """Update motion history with new box position and apply smoothing."""
        motion = self._calculate_motion(new_box)
        self.motion_history.append(motion)
        if len(self.motion_history) > self.max_motion_history:
            self.motion_history = self.motion_history[-self.max_motion_history:]
        
        # V2: Track confidence history
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.max_confidence_history:
            self.confidence_history = self.confidence_history[-self.max_confidence_history:]
        
        self.previous_box = self.box.copy()
        
        # V2: Apply temporal smoothing
        self.box = self._smooth_box(new_box)
    
    def get_avg_confidence(self) -> float:
        """V2: Get average confidence over recent detections."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def get_avg_motion(self) -> float:
        """Get average motion over recent frames."""
        if not self.motion_history:
            return 0.0
        return sum(self.motion_history) / len(self.motion_history)
    
    def is_stationary(self, threshold: float = 5.0) -> bool:
        """Check if the event has been relatively stationary."""
        return self.get_avg_motion() < threshold

    def _add_roi(self, box, frame_img, is_open: bool, frame_index: int = 0, confidence: float = 1.0):
        """
        V4: Extract ROI and add to appropriate buffer with full metadata.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            frame_img: Full frame image for ROI extraction
            is_open: True if this is an open bag detection
            frame_index: Current frame index in the video/stream
            confidence: Detection confidence score
        """
        h, w = frame_img.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            structured_logger.roi_rejected(
                event_id=self.id,
                reason="invalid_dimensions",
                dimensions=(x2-x1, y2-y1),
                sharpness=0.0
            )
            return False

        roi = frame_img[y1:y2, x1:x2].copy()
        
        # Calculate bbox area for stability tracking
        bbox_area = (x2 - x1) * (y2 - y1)

        # Quality check (size and brightness)
        is_valid, sharpness, reject_reason = self._is_valid_roi_with_reason(roi)
        
        if not is_valid:
            # Record rejection for size or brightness
            pipeline_metrics.record_roi_quality(False, sharpness, reject_reason)
            return False
        
        # Check sharpness threshold (separate from basic validation)
        if sharpness < tracking_config.min_roi_sharpness:
            structured_logger.roi_rejected(
                event_id=self.id,
                reason="sharpness",
                sharpness=sharpness,
                dimensions=(roi.shape[1], roi.shape[0]),
                min_required=tracking_config.min_roi_sharpness
            )
            pipeline_metrics.record_roi_quality(False, sharpness, "sharpness")
            return False
        
        # ROI passed all quality checks - log then record metrics
        pipeline_metrics.record_roi_quality(True, sharpness, None)
        
        # V4: Store ROI with full metadata (sharpness, roi, frame_index, bbox_area, confidence)
        roi_entry = (sharpness, roi, frame_index, bbox_area, confidence)

        if is_open:
            self.open_rois.append(roi_entry)
            # Keep top N sharpest
            self.open_rois.sort(key=lambda x: x[0], reverse=True)
            if len(self.open_rois) > self.max_open_samples:
                self.open_rois = self.open_rois[:self.max_open_samples]
            
            # Structured logging for ROI addition
            structured_logger.roi_added(
                event_id=self.id,
                is_open=True,
                sharpness=sharpness,
                frame_index=frame_index,
                confidence=confidence,
                total_rois=len(self.open_rois),
                bbox_area=bbox_area
            )
        else:
            self.closed_rois.append(roi_entry)
            # Keep top N sharpest
            self.closed_rois.sort(key=lambda x: x[0], reverse=True)
            if len(self.closed_rois) > self.max_closed_samples:
                self.closed_rois = self.closed_rois[:self.max_closed_samples]
            
            # Structured logging for ROI addition
            structured_logger.roi_added(
                event_id=self.id,
                is_open=False,
                sharpness=sharpness,
                frame_index=frame_index,
                confidence=confidence,
                total_rois=len(self.closed_rois),
                bbox_area=bbox_area
            )

        return True
    
    def _is_valid_roi_with_reason(self, roi, min_size=None, min_sharpness=None):
        """
        Quality gate with detailed rejection reason tracking.
        
        Returns: (is_valid, sharpness, reject_reason)
        """
        # Use config values if not explicitly provided
        if min_size is None:
            min_size = tracking_config.min_roi_size
        if min_sharpness is None:
            min_sharpness = tracking_config.min_roi_sharpness
        
        reject_reason = None

        h, w = roi.shape[:2]
        if h < min_size or w < min_size:
            # Structured logging for ROI rejection
            structured_logger.roi_rejected(
                event_id=self.id,
                reason="size",
                dimensions=(w, h),
                sharpness=0.0,
                min_required=min_size
            )
            return (False, 0.0, "size")

        # Brightness check
        mean_brightness = roi.mean()
        if not (tracking_config.min_mean_brightness <= mean_brightness <= tracking_config.max_mean_brightness):
            # Structured logging for ROI rejection
            structured_logger.roi_rejected(
                event_id=self.id,
                reason="brightness",
                brightness=mean_brightness,
                dimensions=(w, h),
                min_brightness=tracking_config.min_mean_brightness,
                max_brightness=tracking_config.max_mean_brightness
            )
            return (False, 0.0, "brightness")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return (True, sharpness, None)

    def add_open_frame(self, box, frame_img, confidence: float = 1.0, frame_index: int = 0) -> bool:
        """
        Add ROI from open detection with motion tracking.
        
        V4: Added frame_index for temporal tracking.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            frame_img: Full frame image for ROI extraction
            confidence: Detection confidence score
            frame_index: Current frame index
            
        Returns:
            bool: True if frame was added successfully, False if validation failed
        """
        # V2: Validate aspect ratio
        if not self._validate_aspect_ratio(box):
            return False
        
        self._update_motion(box, confidence)
        self.frames_since_update = 0
        self.total_frames_tracked += 1
        self.current_frame_index = frame_index
        return self._add_roi(self.smoothed_box, frame_img, is_open=True, frame_index=frame_index, confidence=confidence)

    def add_closed_frame(self, box, frame_img, confidence: float = 1.0, frame_index: int = 0) -> bool:
        """
        Add ROI from closed detection with motion tracking.
        
        V4: Added frame_index for temporal tracking.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            frame_img: Full frame image for ROI extraction
            confidence: Detection confidence score
            frame_index: Current frame index
            
        Returns:
            bool: True if frame was added successfully, False if validation failed
        """
        # V2: Validate aspect ratio
        if not self._validate_aspect_ratio(box):
            return False
        
        self._update_motion(box, confidence)
        self.frames_since_update = 0
        self.total_frames_tracked += 1
        self.current_frame_index = frame_index
        return self._add_roi(self.smoothed_box, frame_img, is_open=False, frame_index=frame_index, confidence=confidence)

    def _is_valid_roi(self, roi, min_size=None, min_sharpness=None):
        """Basic quality gate (backward compatible wrapper)."""
        is_valid, sharpness, _ = self._is_valid_roi_with_reason(roi, min_size, min_sharpness)
        return [is_valid, sharpness]

    def get_all_candidates(self) -> List[dict]:
        """
        V4: Return all collected ROIs with full metadata for evidence-based classification.
        
        Returns:
            List of candidate dictionaries, each containing:
            - roi: The ROI image
            - sharpness: Laplacian variance score
            - frame_index: Frame number when captured
            - bbox_area: Bounding box area in pixels
            - confidence: Detection confidence
            - relative_time: Position in track lifecycle (0.0 = start, 1.0 = end)
        """
        # Combine both buffers
        all_rois = self.open_rois + self.closed_rois
        
        if not all_rois:
            return []
        
        # V4: Calculate relative time for each ROI
        track_duration = max(1, self.current_frame_index - self.start_frame_index)
        
        # Convert to list of dictionaries with full metadata
        candidates = []
        for sharpness, roi, frame_index, bbox_area, confidence in all_rois:
            # Ensure no division by zero with additional safety check
            relative_time = (frame_index - self.start_frame_index) / track_duration if track_duration > 0 else 0.5
            candidates.append({
                'roi': roi,
                'sharpness': sharpness,
                'frame_index': frame_index,
                'bbox_area': bbox_area,
                'confidence': confidence,
                'relative_time': relative_time,
            })
        
        # Sort by sharpness (highest first), then by frame_index (later first) for ties
        candidates.sort(key=lambda x: (x['sharpness'], x['frame_index']), reverse=True)
        
        # Select top-K candidates
        top_k = tracking_config.top_k_candidates
        selected_candidates = candidates[:top_k]
        return selected_candidates

    def get_stats(self) -> dict:
        """
        V4: Return comprehensive stats about collected ROIs and event lifecycle.
        
        Includes track duration for determining if track is long enough for classification.
        """
        track_duration = self.current_frame_index - self.start_frame_index
        
        # Calculate average sharpness of all ROIs
        all_sharpness = [s for s, _, _, _, _ in self.open_rois + self.closed_rois]
        avg_sharpness = sum(all_sharpness) / len(all_sharpness) if all_sharpness else 0.0
        
        return {
            "open_count": len(self.open_rois),
            "closed_count": len(self.closed_rois),
            "total": len(self.open_rois) + len(self.closed_rois),
            "open_hits": self.open_hits,
            "closed_hits": self.closed_hits,
            "total_frames_tracked": self.total_frames_tracked,
            "avg_motion": self.get_avg_motion(),
            "is_stationary": self.is_stationary(),
            "avg_confidence": self.get_avg_confidence(),
            # V4: New metrics for evidence-based classification
            "track_duration_frames": track_duration,
            "start_frame": self.start_frame_index,
            "end_frame": self.current_frame_index,
            "avg_sharpness": avg_sharpness,
        }


class BagStateMonitor:
    """
    Monitors bag detection events through their lifecycle.
    
    Enhanced with adaptive suppression based on motion and improved
    event lifecycle tracking for 99.9% accuracy target.
    """
    
    # Adaptive suppression parameters
    STATIONARY_LOCKOUT_MULTIPLIER = 1.5  # Increase lockout for stationary objects
    MOTION_LOCKOUT_REDUCTION = 0.7  # Reduce lockout for moving objects
    
    def __init__(self, open_cls_id, closed_cls_id):

        self.open_id = open_cls_id
        self.closed_id = closed_cls_id

        self.iou_threshold = tracking_config.iou_threshold  # IoU threshold for suppressing duplicate events
        self.lockout_window = tracking_config.lockout_window  # Number of frames to suppress new events
        self.recently_counted = []  # To store recently counted events and suppress duplicates

        # Use config values if not explicitly provided
        self.min_open_frames = tracking_config.min_open_frames
        self.min_closed_frames = tracking_config.min_closed_frames

        self.active_events = []
        
        # Additional metrics tracking
        self.total_events_created = 0
        self.total_events_counted = 0
        self.total_events_expired = 0
        self.total_events_suppressed = 0

        self.HIGH_MOTION_THRESHOLD = 15.0  # tune for your data
        self.STATIONARY_THRESHOLD = 5.0
        self.LOW_IOU_FOR_FLIPPING = 0.2  # for flipping/spinning, allow more difference
        self.SUPPRESSION_CENTER_DIST_PX = 40  # Tune for your image scaling

        logger.info(
            f"[BagStateMonitor] Initialized: open_id={open_cls_id}, "
            f"closed_id={closed_cls_id}, iou={self.iou_threshold}, "
            f"min_open={self.min_open_frames}, min_closed={self.min_closed_frames}, "
            f"lockout_window={self.lockout_window}"
        )
    
    def _get_adaptive_lockout(self, event: Optional[BagEvent] = None) -> int:
        """
        Calculate adaptive lockout window based on motion patterns.
        
        For stationary objects (likely conveyor stopped), use longer lockout.
        For moving objects, use shorter lockout to not miss fast sequences.
        """
        return self.lockout_window
        # base_lockout = self.lockout_window
        #
        # if event is None:
        #     return base_lockout

        # avg_motion = event.get_avg_motion()

        # if avg_motion > self.HIGH_MOTION_THRESHOLD:
        #     # Flipping/spinning: give a "normal" lockout (because we use low IoU for them)
        #     return base_lockout
        # elif avg_motion < self.STATIONARY_THRESHOLD:
        #     return int(base_lockout * self.STATIONARY_LOCKOUT_MULTIPLIER)
        # else:
        #     # Simple moving (not stationary, not flipping): give long lockout (even longer if you wish)
        #     return int(base_lockout * 2.0)  # or any value >1.0 you prefer

    def _center_distance(self, boxA, boxB):
        ax = (boxA[0] + boxA[2]) / 2
        ay = (boxA[1] + boxA[3]) / 2
        bx = (boxB[0] + boxB[2]) / 2
        by = (boxB[1] + boxB[3]) / 2
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def compute_iou(self, boxA, boxB):
        # Sanity checks for box coordinates
        if (boxA[2] <= boxA[0] or boxA[3] <= boxA[1] or
            boxB[2] <= boxB[0] or boxB[3] <= boxB[1]):
            return 0.0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections, frame_dict):
        ready_to_classify = []

        open_dets = [d for d in detections if d['class_id'] == self.open_id]
        closed_dets = [d for d in detections if d['class_id'] == self.closed_id]

        used_open_indices = set()
        used_closed_indices = set()
        matched_event_ids = set()  # Prevent same event matching twice

        # ---------------------------------------------------
        # *** Addition: Maintain Recently Counted Memory ***
        # ---------------------------------------------------
        # Clean up 'recently_counted' memory based on adaptive expiry
        current_frame = frame_dict['frame_count']
        updated_recently_counted = []
        for event in self.recently_counted:
            adaptive_lockout = event.get('adaptive_lockout', self.lockout_window)
            if current_frame - event['frame_count'] <= adaptive_lockout:
                updated_recently_counted.append(event)
        self.recently_counted = updated_recently_counted

        # ---------------------------------------------------
        # 1. Match OPEN detections to existing events
        # ---------------------------------------------------
        for i, det in enumerate(open_dets):
            best_iou = 0
            best_event = None
            for event in self.active_events:
                if event.id in matched_event_ids:
                    continue
                iou = self.compute_iou(event.box, det['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_event = event

            if best_event:
                # V4: Pass confidence and frame_index to add_open_frame
                det_conf = det.get('conf', 1.0)
                best_event.add_open_frame(det['box'], frame_dict['frame'], confidence=det_conf, frame_index=current_frame)
                used_open_indices.add(i)
                matched_event_ids.add(best_event.id)

                if best_event.state != 'counted':
                    best_event.open_hits += 1

                    if best_event.state == 'detecting_closed':
                        # Structured logging for state transition (reopened)
                        structured_logger.event_state_transition(
                            event_id=best_event.id,
                            old_state='detecting_closed',
                            new_state='detecting_open',
                            trigger='open_detection_after_closed',
                            open_hits=best_event.open_hits,
                            closed_hits=best_event.closed_hits,
                            iou=best_iou,
                            frame_index=current_frame
                        )
                        best_event.closed_hits = 0
                        best_event.state = 'detecting_open'

        # ---------------------------------------------------
        # 2. Match CLOSED detections to existing events
        # ---------------------------------------------------
        for j, det in enumerate(closed_dets):
            best_iou = 0
            best_event = None
            for event in self.active_events:
                if event.id in matched_event_ids:
                    continue
                iou = self.compute_iou(event.box, det['box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_event = event

            if best_event:
                # V4: Pass confidence and frame_index to add_closed_frame
                det_conf = det.get('conf', 1.0)
                best_event.add_closed_frame(det['box'], frame_dict['frame'], confidence=det_conf, frame_index=current_frame)
                used_closed_indices.add(j)
                matched_event_ids.add(best_event.id)

                if best_event.state != 'counted':
                    if best_event.open_hits >= self.min_open_frames:
                        old_state = best_event.state
                        best_event.closed_hits += 1
                        best_event.state = 'detecting_closed'
                        
                        # Structured logging for state transition (if state changed)
                        if old_state != 'detecting_closed':
                            structured_logger.event_state_transition(
                                event_id=best_event.id,
                                old_state=old_state,
                                new_state='detecting_closed',
                                trigger='min_open_frames_reached',
                                open_hits=best_event.open_hits,
                                closed_hits=best_event.closed_hits,
                                iou=best_iou,
                                frame_index=current_frame
                            )

        # ---------------------------------------------------
        # 3. Create NEW events for unmatched open detections
        # ---------------------------------------------------
        for i, det in enumerate(open_dets):
            if i not in used_open_indices:
                if det.get('conf', 1.0) < tracking_config.min_conf_threshold:
                    pipeline_metrics.record_detection_filtered("low_confidence")
                    continue

                # Prevent excessive memory usage with too many active events
                if len(self.active_events) >= tracking_config.max_active_events:
                    structured_logger.pipeline_error(
                        component="BagStateMonitor",
                        operation="event_creation",
                        error_type="ResourceLimit",
                        error_message=f"Max active events reached ({tracking_config.max_active_events})",
                        affected_ids=None,
                        context={
                            "active_events_count": len(self.active_events),
                            "max_allowed": tracking_config.max_active_events
                        }
                    )
                    break

                # ---------------------------------------------------
                # *** Addition: Suppress Events Using IoU ***
                # ---------------------------------------------------
                suppress_event = False
                for counted_event in self.recently_counted:
                    iou = self.compute_iou(det['box'], counted_event['box'])
                    avg_motion = counted_event['avg_motion']

                    # If the counted event had very high motion (flipping/spinning), reduce IoU threshold
                    if avg_motion > self.HIGH_MOTION_THRESHOLD:
                        iou_thresh = self.LOW_IOU_FOR_FLIPPING

                    center_dist = self._center_distance(det['box'], counted_event['box'])
                    if iou > self.iou_threshold or center_dist < self.SUPPRESSION_CENTER_DIST_PX:
                        suppress_event = True
                        self.total_events_suppressed += 1
                        pipeline_metrics.record_event_suppressed()
                        
                        # Structured logging for event suppression
                        structured_logger.event_suppressed(
                            event_id=-1,  # New event not created yet
                            reason='duplicate_spatial_overlap',
                            iou=iou,
                            conflicting_event_id=counted_event['id'],
                            center_distance=center_dist,
                            frame_index=current_frame,
                            detection_confidence=det.get('conf', 1.0)
                        )
                        break

                if suppress_event:
                    continue

                # Create a new event with frame_index
                new_event = BagEvent(det['box'], frame_dict['frame'], self.open_id, self.closed_id, frame_index=current_frame)
                self.active_events.append(new_event)
                self.total_events_created += 1

        # ---------------------------------------------------
        # 4. Check triggers & cleanup
        # ---------------------------------------------------
        active_next_frame = []
        expired_count = 0

        for event in self.active_events:
            event.frames_since_update += 1

            # Trigger classification when closed threshold reached
            if (event.state == 'detecting_closed' and
                    event.closed_hits >= self.min_closed_frames and
                    event.state != 'counted'):

                candidates = event.get_all_candidates()
                stats = event.get_stats()

                if candidates:
                    ready_to_classify.append((event.id, candidates, event.box, stats))
                else:
                    # Log error if event reached counted state without candidates
                    structured_logger.pipeline_error(
                        component="BagStateMonitor",
                        operation="event_classification_ready",
                        error_type="NoCandidates",
                        error_message=f"Event {event.id} triggered but no candidates collected",
                        affected_ids=[event.id],
                        context=stats
                    )

                old_state = event.state
                event.state = 'counted'  # Transition to counted
                self.total_events_counted += 1
                
                # Structured logging for state transition to counted
                structured_logger.event_state_transition(
                    event_id=event.id,
                    old_state=old_state,
                    new_state='counted',
                    trigger='classification_ready',
                    open_hits=event.open_hits,
                    closed_hits=event.closed_hits,
                    total_frames=event.total_frames_tracked,
                    candidates_count=len(candidates)
                )
                
                # Record metrics with event details
                pipeline_metrics.record_event_counted(event.open_hits, event.closed_hits)

                # ---------------------------------------------------
                # *** Addition: Add to Recently Counted Memory ***
                # Use adaptive lockout based on motion patterns
                # ---------------------------------------------------
                adaptive_lockout = self._get_adaptive_lockout(event)
                avg_motion = event.get_avg_motion()
                self.recently_counted.append({
                    'frame_count': frame_dict['frame_count'],
                    'box': event.box,
                    'id': event.id,
                    'adaptive_lockout': adaptive_lockout,
                    'is_stationary': event.is_stationary(),
                    'avg_motion': avg_motion,
                })
                logger.debug(
                    f"[BagStateMonitor] Event {event.id} state -> counted, "
                    f"adaptive_lockout={adaptive_lockout}"
                )

            # Handle state expiry
            if event.state == 'detecting_open':
                expiry_threshold = tracking_config.expiry_detecting_open
            elif event.state == 'detecting_closed':
                expiry_threshold = tracking_config.expiry_detecting_closed
            else:  # 'counted'
                expiry_threshold = tracking_config.expiry_counted

            if event.frames_since_update < expiry_threshold:
                active_next_frame.append(event)
            else:
                expired_count += 1
                self.total_events_expired += 1
                pipeline_metrics.record_event_expired(event.state)
                
                # Structured logging for event expiration
                structured_logger.event_expired(
                    event_id=event.id,
                    state=event.state,
                    frames_tracked=event.total_frames_tracked,
                    open_hits=event.open_hits,
                    closed_hits=event.closed_hits,
                    frames_since_update=event.frames_since_update,
                    avg_motion=event.get_avg_motion(),
                    avg_confidence=event.get_avg_confidence()
                )

        self.active_events = active_next_frame

        return ready_to_classify
    
    def get_monitor_stats(self) -> dict:
        """Return overall monitor statistics for monitoring."""
        return {
            "total_events_created": self.total_events_created,
            "total_events_counted": self.total_events_counted,
            "total_events_expired": self.total_events_expired,
            "total_events_suppressed": self.total_events_suppressed,
            "active_events": len(self.active_events),
            "recently_counted": len(self.recently_counted),
            "completion_rate": (
                self.total_events_counted / self.total_events_created
                if self.total_events_created > 0 else 0.0
            ),
        }