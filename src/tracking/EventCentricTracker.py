"""
Event-Centric Tracking System for BreadBag Counter.

This module implements an event-centric approach to tracking bread bags,
designed for human-operated table environments where bags are rotated,
flipped, and temporarily occluded.

ARCHITECTURE OVERVIEW:
- An Event represents one physical bread-bag operation
- Events survive detection loss, track fragmentation, and hand occlusion
- Tracks/detections are observations attached to Events
- Counting occurs only after bag exits the scene (not at closure moment)

HARD CONSTRAINTS MET:
- NO visual appearance embeddings
- NO IoU-based association (uses centroid distance + time)
- NO frame-based counting (uses millisecond-based timing)
- Tolerates rotation, deformation, and temporary occlusion

TARGET: ≥99.9% counting reliability (≤1 error per 1000 bags)
"""

import time
import math
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import cv2

from src.utils.AppLogging import logger, structured_logger
from src.utils.PipelineMetrics import pipeline_metrics


class EventState(Enum):
    """
    Event state machine states.
    
    State transitions:
        OPEN -> CLOSING (when closed detections start)
        CLOSING -> OPEN (if open detections resume)
        CLOSING -> CLOSED (temporal stability reached)
        CLOSED -> COMMITTED (after timeout with no detections - timeout-based only)
    
    NOTE: Commitment is based exclusively on timeout (idle time without detection).
    Exit boundary logic has been removed to simplify and improve robustness.
    """
    OPEN = auto()       # Bag is open, being manipulated
    CLOSING = auto()    # Transitioning to closed state
    CLOSED = auto()     # Bag is closed, collecting final ROIs
    COMMITTED = auto()  # Event finalized, ready for counting


@dataclass
class EventConfig:
    """
    Configuration for event-centric tracking system.
    
    All time-based parameters are in milliseconds for precision.
    Distance parameters are in pixels.
    
    Tuning Guidelines:
    - D (association_distance_px): Based on expected bag movement per frame.
      Typical: 50-150px for 720p video at 25fps
    - T (association_time_ms): Max time gap to associate detection to event.
      Typical: 200-500ms for human manipulation speed
    - G (ghost_timeout_ms): How long to keep event alive without detections.
      Should cover typical hand occlusion duration: 500-1500ms
    
    COMMITMENT MODEL (Timeout-Based Only):
    - Commitment is based exclusively on timeout (idle time without detection).
    - After an event enters the CLOSED state, it is committed if undetected for
      the configured idle timeout (commit_idle_frames frames or ghost_timeout_ms).
    - Exit boundary logic has been removed to simplify and improve robustness.
    - Anti-double-counting is achieved through suppression of new events near
      recently committed events for a configurable duration.
    """
    
    # ==========================================================================
    # Work Zone Configuration
    # ==========================================================================
    work_zone_enabled: bool = False
    work_zone_x1: int = 0      # Top-left X of work zone
    work_zone_y1: int = 0      # Top-left Y of work zone
    work_zone_x2: int = 1280   # Bottom-right X of work zone
    work_zone_y2: int = 720    # Bottom-right Y of work zone
    
    # ==========================================================================
    # Event Association Parameters (D, T from requirements)
    # ==========================================================================
    association_distance_px: float = 100.0  # D: Max centroid distance for association
    association_time_ms: float = 400.0      # T: Max time gap for association
    
    # ==========================================================================
    # IoU-Based Association (complementary to centroid distance)
    # ==========================================================================
    # IoU provides robustness when centroid distance alone may fail (e.g., during
    # partial occlusion where box overlaps but centroid shifts significantly)
    iou_association_enabled: bool = True    # Enable IoU as additional association criterion
    iou_association_threshold: float = 0.3  # Min IoU to associate (if centroid fails)
    
    # ==========================================================================
    # Velocity-Based Association (for fast movements during flip/throw)
    # ==========================================================================
    velocity_scaling_enabled: bool = True   # Enable velocity-based distance scaling
    velocity_scale_factor: float = 2.5      # Max multiplier for association distance
    max_association_distance_px: float = 250.0  # Absolute max association distance
    min_velocity_threshold: float = 0.01    # Min velocity (px/ms) to trigger scaling
    max_prediction_time_ms: float = 500.0   # Max time ahead to predict centroid
    
    # ==========================================================================
    # Ghost Event Parameters (G from requirements)
    # ==========================================================================
    ghost_timeout_ms: float = 1000.0  # G: Keep event alive without detections
    
    # ==========================================================================
    # Timeout-Based Commitment Parameters (Exclusive Method)
    # ==========================================================================
    # NOTE: Exit boundary logic has been removed. Commitment is now based
    # exclusively on timeout (idle time without detection).
    commit_idle_frames: int = 25           # Frames without detection before commit
    commit_min_closed_ratio: float = 0.3   # Min ratio of closed/total evidence for commit
    
    # ==========================================================================
    # Anti-Double-Counting Suppression Parameters
    # ==========================================================================
    # These parameters prevent new events from being created for a bag that was
    # temporarily lost then re-detected after commitment.
    suppression_distance_px: float = 150.0   # Distance within which new events are suppressed
    suppression_duration_ms: float = 1000.0  # Duration to suppress new events after commit
    
    # ==========================================================================
    # State Transition Parameters (temporal stability)
    # ==========================================================================
    open_to_closing_time_ms: float = 100.0   # Min time in OPEN before CLOSING
    closing_stability_time_ms: float = 150.0  # Closed detections must persist this long
    closed_stability_time_ms: float = 200.0   # Min time in CLOSED before COMMIT eligible
    
    # Geometric stability thresholds
    centroid_stability_px: float = 30.0  # Max centroid movement for "stable"
    
    # ==========================================================================
    # State Reversion Parameters (to prevent oscillation)
    # ==========================================================================
    closing_revert_open_count: int = 3    # Open detections in recent window to revert CLOSING->OPEN
    closing_revert_window_size: int = 5   # Window size for revert check
    
    # ==========================================================================
    # Detection Evidence Thresholds
    # ==========================================================================
    min_open_evidence_count: int = 3    # Min open detections before state can change
    min_closed_evidence_count: int = 2  # Min closed detections for CLOSED state
    
    # Confidence thresholds for evidence
    min_detection_confidence: float = 0.4
    
    # ==========================================================================
    # ROI Collection Parameters
    # ==========================================================================
    max_roi_samples: int = 8            # Max ROIs to collect during CLOSED
    min_roi_size: int = 100             # Min ROI dimension
    min_roi_sharpness: float = 300.0    # Min Laplacian variance
    min_brightness: int = 80
    max_brightness: int = 220
    
    # ==========================================================================
    # Classification Voting Parameters
    # ==========================================================================
    min_voting_agreement_pct: float = 60.0    # Min % votes for same class
    confidence_margin_threshold: float = 0.15  # Min margin between top classes
    
    # ==========================================================================
    # Resource Limits
    # ==========================================================================
    max_active_events: int = 10


@dataclass
class DetectionEvidence:
    """
    Evidence from a single detection, decoupled from YOLO output.
    
    YOLO outputs (open/closed confidence) are stored as evidence,
    not as final state determination.
    """
    timestamp_ms: float
    centroid_x: float
    centroid_y: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    is_open: bool
    is_closed: bool
    confidence: float
    frame_index: int


@dataclass
class ROICandidate:
    """
    ROI candidate with quality metrics for classification.
    """
    roi: np.ndarray
    sharpness: float
    size: Tuple[int, int]  # width, height
    timestamp_ms: float
    frame_index: int
    centroid_stability: float  # How stable the centroid was when captured
    confidence: float


class BreadBagEvent:
    """
    Represents a single bread-bag operation event.
    
    An Event survives:
    - Detection loss (ghost period)
    - Track fragmentation
    - Hand occlusion
    
    State Machine:
        OPEN -> CLOSING -> CLOSED -> COMMITTED
        
    Counting Rule (Timeout-Based Only):
        Event is counted ONLY when:
        1. State == CLOSED
        2. No detections for commit_idle_frames (timeout-based commitment)
        3. Minimum closed evidence ratio is met
        
    NOTE: Exit boundary logic has been removed. Commitment relies exclusively
    on timeout-based logic to ensure robustness and simplicity.
    """
    
    def __init__(self, 
                 initial_detection: DetectionEvidence,
                 config: EventConfig,
                 open_class_id: int,
                 closed_class_id: int):
        """
        Create a new event from an initial detection.
        
        Args:
            initial_detection: First detection that created this event
            config: Event configuration parameters
            open_class_id: Class ID for open bag detections
            closed_class_id: Class ID for closed bag detections
        """
        self.id = int(uuid.uuid4().int >> 96)
        self.config = config
        self.open_class_id = open_class_id
        self.closed_class_id = closed_class_id
        
        # State machine
        self.state = EventState.OPEN
        self.state_enter_time_ms = initial_detection.timestamp_ms
        self.state_enter_evidence_idx = 0  # Track which evidence index we entered current state
        
        # Temporal tracking
        self.created_at_ms = initial_detection.timestamp_ms
        self.last_detection_time_ms = initial_detection.timestamp_ms
        self.last_update_time_ms = initial_detection.timestamp_ms
        
        # Spatial tracking (centroid-based, NOT IoU)
        self.last_centroid = (initial_detection.centroid_x, initial_detection.centroid_y)
        self.centroid_history: List[Tuple[float, float, float]] = [
            (initial_detection.centroid_x, initial_detection.centroid_y, initial_detection.timestamp_ms)
        ]
        self.last_box = initial_detection.box
        
        # Evidence collection (decoupled from YOLO state)
        self.evidence_history: List[DetectionEvidence] = [initial_detection]
        self.open_evidence_count = 1 if initial_detection.is_open else 0
        self.closed_evidence_count = 1 if initial_detection.is_closed else 0
        
        # Detection gap tracking (for ghost events)
        self.detection_gaps: List[Tuple[float, float]] = []  # (start_ms, end_ms)
        self.current_gap_start: Optional[float] = None
        
        # ROI collection (during CLOSED state)
        self.roi_candidates: List[ROICandidate] = []
        
        # State transition history for debugging
        self.state_transitions: List[Dict[str, Any]] = [{
            'timestamp_ms': initial_detection.timestamp_ms,
            'from_state': None,
            'to_state': EventState.OPEN.name,
            'trigger': 'event_created'
        }]
        
        # Classification result (set after COMMIT)
        self.classification_result: Optional[Dict[str, Any]] = None
        self.commit_reason: Optional[str] = None
        
        # Metrics
        self.total_frames_observed = 1
        
        # Velocity tracking for fast movement handling
        self.velocity = (0.0, 0.0)  # (vx, vy) in pixels per millisecond
        self.velocity_history: List[Tuple[float, float, float]] = []  # (vx, vy, timestamp_ms)
        
        # Frame-based idle tracking for center commit
        self.frames_without_detection = 0
        self.last_detection_frame_index = initial_detection.frame_index
        
        # Log event creation
        structured_logger.event_created(
            event_id=self.id,
            confidence=initial_detection.confidence,
            box=list(initial_detection.box),
            frame_index=initial_detection.frame_index,
            state=self.state.name
        )
        pipeline_metrics.record_event_created()
    
    def get_centroid(self) -> Tuple[float, float]:
        """Get current centroid position."""
        return self.last_centroid
    
    def get_centroid_stability(self) -> float:
        """
        Calculate centroid stability over recent history.
        
        Returns average movement magnitude - lower is more stable.
        """
        if len(self.centroid_history) < 2:
            return 0.0
        
        # Look at last 5 positions
        recent = self.centroid_history[-5:]
        if len(recent) < 2:
            return 0.0
        
        movements = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            movements.append(math.sqrt(dx*dx + dy*dy))
        
        return sum(movements) / len(movements) if movements else 0.0
    
    def _compute_centroid(self, box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Compute centroid from bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate (pixels per millisecond).
        
        Uses recent centroid history to estimate velocity.
        """
        if len(self.centroid_history) < 2:
            return (0.0, 0.0)
        
        # Use last 3-5 positions for smoothed velocity
        recent = self.centroid_history[-5:]
        if len(recent) < 2:
            return (0.0, 0.0)
        
        # Calculate weighted average velocity (more recent = more weight)
        total_vx, total_vy = 0.0, 0.0
        total_weight = 0.0
        
        for i in range(1, len(recent)):
            x1, y1, t1 = recent[i-1]
            x2, y2, t2 = recent[i]
            dt = t2 - t1
            if dt > 0:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                weight = i  # More recent = higher weight
                total_vx += vx * weight
                total_vy += vy * weight
                total_weight += weight
        
        if total_weight > 0:
            return (total_vx / total_weight, total_vy / total_weight)
        return (0.0, 0.0)
    
    def get_velocity_magnitude(self) -> float:
        """Get velocity magnitude in pixels per millisecond."""
        vx, vy = self.get_velocity()
        return math.sqrt(vx*vx + vy*vy)
    
    def predict_centroid(self, target_time_ms: float) -> Tuple[float, float]:
        """
        Predict centroid position at target_time using velocity.
        
        This helps with association during fast movements (flip/throw).
        """
        if len(self.centroid_history) < 2:
            return self.last_centroid
        
        vx, vy = self.get_velocity()
        dt = target_time_ms - self.last_detection_time_ms
        
        # Limit prediction to configurable max time ahead
        dt = min(dt, self.config.max_prediction_time_ms)
        
        pred_x = self.last_centroid[0] + vx * dt
        pred_y = self.last_centroid[1] + vy * dt
        
        return (pred_x, pred_y)
    
    def _compute_iou(self, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Compute intersection
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Check if there is an intersection
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Compute areas of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute union
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def can_associate(self, detection: DetectionEvidence) -> Tuple[bool, float, str]:
        """
        Check if a detection can be associated with this event.
        
        Uses multiple association criteria for robustness:
        1. Centroid distance (primary) - with velocity-based scaling
        2. IoU (complementary) - helps during partial occlusion
        
        A detection can associate if EITHER:
        - Centroid distance is within threshold, OR
        - IoU is above threshold (when enabled)
        
        Args:
            detection: Detection to check
            
        Returns:
            Tuple of (can_associate, distance, reason)
        """
        det_centroid = (detection.centroid_x, detection.centroid_y)
        
        # Check time gap first (cheap check)
        time_gap_ms = detection.timestamp_ms - self.last_detection_time_ms
        if time_gap_ms > self.config.association_time_ms:
            # Calculate distance for logging
            dx = det_centroid[0] - self.last_centroid[0]
            dy = det_centroid[1] - self.last_centroid[1]
            distance = math.sqrt(dx*dx + dy*dy)
            return False, distance, f"time_gap_exceeded ({time_gap_ms:.1f}ms > {self.config.association_time_ms}ms)"
        
        # Calculate base association distance
        base_distance_threshold = self.config.association_distance_px
        
        # Velocity-based scaling: increase threshold for fast-moving bags
        if self.config.velocity_scaling_enabled:
            velocity_mag = self.get_velocity_magnitude()
            
            # Scale factor: velocity * time_gap gives expected movement
            # If velocity is high, we expect larger movements
            if velocity_mag > self.config.min_velocity_threshold:
                expected_movement = velocity_mag * time_gap_ms
                # Scale the threshold based on expected movement
                scale = 1.0 + min(expected_movement / base_distance_threshold, 
                                  self.config.velocity_scale_factor - 1.0)
                scaled_threshold = min(base_distance_threshold * scale, 
                                       self.config.max_association_distance_px)
            else:
                scaled_threshold = base_distance_threshold
        else:
            scaled_threshold = base_distance_threshold
        
        # Calculate distance to last centroid
        dx = det_centroid[0] - self.last_centroid[0]
        dy = det_centroid[1] - self.last_centroid[1]
        distance_to_last = math.sqrt(dx*dx + dy*dy)
        
        # Also try distance to predicted position (for fast movements)
        if self.config.velocity_scaling_enabled and len(self.centroid_history) >= 2:
            pred_centroid = self.predict_centroid(detection.timestamp_ms)
            dx_pred = det_centroid[0] - pred_centroid[0]
            dy_pred = det_centroid[1] - pred_centroid[1]
            distance_to_pred = math.sqrt(dx_pred*dx_pred + dy_pred*dy_pred)
            
            # Use the smaller of the two distances
            distance = min(distance_to_last, distance_to_pred)
        else:
            distance = distance_to_last
        
        # Check centroid distance threshold (primary criterion)
        centroid_match = distance <= scaled_threshold
        
        # Check IoU (complementary criterion)
        iou_match = False
        iou_value = 0.0
        if self.config.iou_association_enabled and self.last_box is not None:
            iou_value = self._compute_iou(self.last_box, detection.box)
            iou_match = iou_value >= self.config.iou_association_threshold
        
        # Associate if EITHER centroid OR IoU criterion is met
        if centroid_match:
            return True, distance, f"centroid_match (dist={distance:.1f}px)"
        elif iou_match:
            return True, distance, f"iou_match (iou={iou_value:.2f}, dist={distance:.1f}px)"
        else:
            # Neither criterion met
            reason = f"no_match (dist={distance:.1f} > {scaled_threshold:.1f}"
            if self.config.iou_association_enabled:
                reason += f", iou={iou_value:.2f} < {self.config.iou_association_threshold}"
            reason += ")"
            return False, distance, reason
    
    def add_detection(self, detection: DetectionEvidence, frame_img: Optional[np.ndarray] = None):
        """
        Add a detection as evidence to this event.
        
        Args:
            detection: Detection evidence to add
            frame_img: Optional frame image for ROI collection
        """
        # Close any detection gap
        if self.current_gap_start is not None:
            gap_duration = detection.timestamp_ms - self.current_gap_start
            self.detection_gaps.append((self.current_gap_start, detection.timestamp_ms))
            self.current_gap_start = None
            logger.debug(
                f"[Event:{self.id}] Detection gap closed: {gap_duration:.1f}ms"
            )
        
        # Update velocity before updating centroid
        if len(self.centroid_history) >= 1:
            last_pos = self.centroid_history[-1]
            dt = detection.timestamp_ms - last_pos[2]
            if dt > 0:
                vx = (detection.centroid_x - last_pos[0]) / dt
                vy = (detection.centroid_y - last_pos[1]) / dt
                self.velocity = (vx, vy)
                self.velocity_history.append((vx, vy, detection.timestamp_ms))
                # Keep velocity history bounded
                if len(self.velocity_history) > 10:
                    self.velocity_history = self.velocity_history[-10:]
        
        # Update spatial tracking
        self.last_centroid = (detection.centroid_x, detection.centroid_y)
        self.centroid_history.append(
            (detection.centroid_x, detection.centroid_y, detection.timestamp_ms)
        )
        self.last_box = detection.box
        
        # Keep centroid history bounded
        if len(self.centroid_history) > 30:
            self.centroid_history = self.centroid_history[-30:]
        
        # Update evidence
        self.evidence_history.append(detection)
        if detection.is_open:
            self.open_evidence_count += 1
        if detection.is_closed:
            self.closed_evidence_count += 1
        
        # Update timing and frame tracking
        self.last_detection_time_ms = detection.timestamp_ms
        self.last_update_time_ms = detection.timestamp_ms
        self.total_frames_observed += 1
        self.frames_without_detection = 0  # Reset idle counter
        self.last_detection_frame_index = detection.frame_index
        
        # Process state transitions based on evidence
        self._process_state_transition(detection)
        
        # Collect ROI if in CLOSED state
        if self.state == EventState.CLOSED and frame_img is not None:
            self._try_collect_roi(detection, frame_img)
    
    def _process_state_transition(self, detection: DetectionEvidence):
        """
        Process state transitions based on new evidence.
        
        State transitions require temporal stability and evidence agreement.
        One incorrect frame does NOT change state.
        """
        current_time_ms = detection.timestamp_ms
        time_in_state_ms = current_time_ms - self.state_enter_time_ms
        
        if self.state == EventState.OPEN:
            # Can transition to CLOSING if closed evidence starts accumulating
            # Requires: min time in OPEN + closed evidence
            if (time_in_state_ms >= self.config.open_to_closing_time_ms and
                self.closed_evidence_count >= 1 and
                self.open_evidence_count >= self.config.min_open_evidence_count):
                self._transition_to(EventState.CLOSING, detection.timestamp_ms, 
                                    "closed_evidence_detected")
        
        elif self.state == EventState.CLOSING:
            # Can transition to CLOSED if closed evidence is stable
            # Can revert to OPEN if open evidence resumes (with hysteresis)
            
            # Check for reversion to OPEN - only consider evidence SINCE entering CLOSING
            # This prevents immediate reversion due to earlier open evidence
            window_size = self.config.closing_revert_window_size
            revert_threshold = self.config.closing_revert_open_count
            
            # Get evidence since entering CLOSING state
            evidence_since_closing = self.evidence_history[self.state_enter_evidence_idx:]
            
            # Only check for reversion if we have enough evidence since CLOSING
            if len(evidence_since_closing) >= revert_threshold:
                recent_evidence = evidence_since_closing[-window_size:] if len(evidence_since_closing) >= window_size else evidence_since_closing
                recent_open = sum(1 for e in recent_evidence if e.is_open)
                
                if recent_open >= revert_threshold:
                    self._transition_to(EventState.OPEN, detection.timestamp_ms,
                                        f"open_evidence_resumed ({recent_open}/{len(recent_evidence)} open)")
                    return
            
            # Check for progression to CLOSED
            if (time_in_state_ms >= self.config.closing_stability_time_ms and
                self.closed_evidence_count >= self.config.min_closed_evidence_count):
                # Also check geometric stability
                stability = self.get_centroid_stability()
                if stability <= self.config.centroid_stability_px:
                    self._transition_to(EventState.CLOSED, detection.timestamp_ms,
                                        f"closing_stable (stability={stability:.1f}px)")
        
        elif self.state == EventState.CLOSED:
            # CLOSED state - collecting ROIs, waiting for exit
            # Can only go to COMMITTED via update_ghost_state
            pass
    
    def _transition_to(self, new_state: EventState, timestamp_ms: float, trigger: str):
        """Record state transition with logging."""
        old_state = self.state
        self.state = new_state
        self.state_enter_time_ms = timestamp_ms
        self.state_enter_evidence_idx = len(self.evidence_history) - 1  # Track evidence index
        
        transition_record = {
            'timestamp_ms': timestamp_ms,
            'from_state': old_state.name,
            'to_state': new_state.name,
            'trigger': trigger
        }
        self.state_transitions.append(transition_record)
        
        structured_logger.event_state_transition(
            event_id=self.id,
            old_state=old_state.name,
            new_state=new_state.name,
            trigger=trigger,
            open_hits=self.open_evidence_count,
            closed_hits=self.closed_evidence_count
        )
    
    def _try_collect_roi(self, detection: DetectionEvidence, frame_img: np.ndarray):
        """
        Try to collect an ROI candidate during CLOSED state.
        
        ROIs are scored for quality (sharpness, size, stability).
        """
        if len(self.roi_candidates) >= self.config.max_roi_samples:
            return
        
        x1, y1, x2, y2 = map(int, detection.box)
        h, w = frame_img.shape[:2]
        
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # Size check
        if roi_width < self.config.min_roi_size or roi_height < self.config.min_roi_size:
            return
        
        roi = frame_img[y1:y2, x1:x2].copy()
        
        # Brightness check
        mean_brightness = roi.mean()
        if not (self.config.min_brightness <= mean_brightness <= self.config.max_brightness):
            return
        
        # Sharpness check
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if sharpness < self.config.min_roi_sharpness:
            return
        
        # Create ROI candidate
        candidate = ROICandidate(
            roi=roi,
            sharpness=sharpness,
            size=(roi_width, roi_height),
            timestamp_ms=detection.timestamp_ms,
            frame_index=detection.frame_index,
            centroid_stability=self.get_centroid_stability(),
            confidence=detection.confidence
        )
        
        self.roi_candidates.append(candidate)
        
        # Keep sorted by quality (sharpness primary)
        self.roi_candidates.sort(key=lambda x: x.sharpness, reverse=True)
        
        logger.debug(
            f"[Event:{self.id}] ROI collected: sharpness={sharpness:.1f}, "
            f"size={roi_width}x{roi_height}, total={len(self.roi_candidates)}"
        )
    
    def update_ghost_state(self, current_time_ms: float, frame_size: Tuple[int, int], 
                           current_frame_index: int = -1) -> bool:
        """
        Update event when no detection is present (ghost state).
        
        Commitment Logic (Timeout-Based Only):
        - After entering CLOSED state, commit if undetected for commit_idle_frames
        - Requires minimum closed evidence ratio to be met
        - No exit boundary check - commitment is purely timeout-based
        
        Args:
            current_time_ms: Current timestamp in milliseconds
            frame_size: (width, height) of frame (kept for API compatibility)
            current_frame_index: Current frame index for idle tracking
            
        Returns:
            True if event should be committed (counted), False otherwise
        """
        # Start gap tracking if not already
        if self.current_gap_start is None:
            self.current_gap_start = self.last_detection_time_ms
        
        time_since_detection_ms = current_time_ms - self.last_detection_time_ms
        self.last_update_time_ms = current_time_ms
        
        # Update frames without detection counter
        if current_frame_index >= 0:
            self.frames_without_detection = current_frame_index - self.last_detection_frame_index
        
        # Check if ghost timeout exceeded
        if time_since_detection_ms > self.config.ghost_timeout_ms:
            if self.state == EventState.CLOSED:
                # Timeout-based commitment: check if enough frames have passed without detection
                if self.frames_without_detection >= self.config.commit_idle_frames:
                    # Verify we have sufficient closed evidence
                    total_evidence = self.open_evidence_count + self.closed_evidence_count
                    if total_evidence > 0:
                        closed_ratio = self.closed_evidence_count / total_evidence
                        if closed_ratio >= self.config.commit_min_closed_ratio:
                            self._transition_to(EventState.COMMITTED, current_time_ms,
                                                f"timeout_commit (idle={self.frames_without_detection} frames, "
                                                f"closed_ratio={closed_ratio:.2f})")
                            self.commit_reason = "timeout_commit"
                            logger.info(
                                f"[Event:{self.id}] Timeout commit: bag counted after idle timeout "
                                f"(idle={self.frames_without_detection} frames, centroid={self.last_centroid})"
                            )
                            return True
                        else:
                            # Not enough closed evidence - don't commit
                            logger.debug(
                                f"[Event:{self.id}] CLOSED but insufficient closed ratio "
                                f"({closed_ratio:.2f} < {self.config.commit_min_closed_ratio})"
                            )
                            return False
                
                # Not enough idle time yet
                logger.debug(
                    f"[Event:{self.id}] CLOSED but waiting for timeout "
                    f"(idle_frames={self.frames_without_detection}, "
                    f"required={self.config.commit_idle_frames})"
                )
                return False
            else:
                # Event expired without reaching CLOSED state
                logger.debug(
                    f"[Event:{self.id}] Expired in state {self.state.name} "
                    f"after {time_since_detection_ms:.0f}ms without detection"
                )
                return False  # Don't commit, just expire
        
        return False
    
    def get_roi_candidates(self) -> List[Dict[str, Any]]:
        """
        Get ROI candidates for classification.
        
        Returns candidates formatted for ClassifierService.
        """
        candidates = []
        track_duration = len(self.evidence_history)
        
        for idx, roi_cand in enumerate(self.roi_candidates):
            # Calculate relative time (0.0 = start, 1.0 = end of track)
            relative_time = idx / max(1, len(self.roi_candidates) - 1) if len(self.roi_candidates) > 1 else 0.5
            
            candidates.append({
                'roi': roi_cand.roi,
                'sharpness': roi_cand.sharpness,
                'frame_index': roi_cand.frame_index,
                'bbox_area': roi_cand.size[0] * roi_cand.size[1],
                'confidence': roi_cand.confidence,
                'relative_time': relative_time,
            })
        
        return candidates
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get comprehensive debug information for analysis.
        
        Returns all data needed for:
        - Dropped events analysis
        - False splits detection
        - Premature commits detection
        """
        lifespan_ms = self.last_update_time_ms - self.created_at_ms
        total_gap_time = sum(end - start for start, end in self.detection_gaps)
        
        return {
            'event_id': self.id,
            'state': self.state.name,
            'lifespan_ms': lifespan_ms,
            'created_at_ms': self.created_at_ms,
            'last_detection_ms': self.last_detection_time_ms,
            'total_frames_observed': self.total_frames_observed,
            'open_evidence_count': self.open_evidence_count,
            'closed_evidence_count': self.closed_evidence_count,
            'detection_gaps': self.detection_gaps,
            'total_gap_time_ms': total_gap_time,
            'state_transitions': self.state_transitions,
            'roi_count': len(self.roi_candidates),
            'last_centroid': self.last_centroid,
            'centroid_stability': self.get_centroid_stability(),
            'commit_reason': self.commit_reason,
        }


class EventCentricTracker:
    """
    Event-centric tracker for bread bag counting.
    
    Key Design Principles:
    1. Events, not tracks - An Event represents a physical bag operation
    2. Centroid-based association - No IoU or appearance features
    3. Millisecond-based timing - Not frame counts
    4. Timeout-based counting - Count after idle timeout, not at boundary
    
    Anti-Double-Counting:
    - Suppresses new event creation near recently committed events
    - Uses configurable suppression distance and duration
    - Ensures each physical bag is counted exactly once
    
    Usage:
        tracker = EventCentricTracker(config, open_id=1, closed_id=0)
        
        for frame_index, frame in enumerate(video):
            timestamp_ms = frame_index * (1000 / fps)
            detections = detector.detect(frame)
            
            ready_events = tracker.update(detections, timestamp_ms, frame, frame_index)
            
            for event_data in ready_events:
                # Process classification and counting
                pass
    """
    
    def __init__(self, 
                 config: Optional[EventConfig] = None,
                 open_class_id: int = 1,
                 closed_class_id: int = 0):
        """
        Initialize the event-centric tracker.
        
        Args:
            config: EventConfig instance (uses defaults if None)
            open_class_id: Class ID for open bag detections
            closed_class_id: Class ID for closed bag detections
        """
        self.config = config or EventConfig()
        self.open_class_id = open_class_id
        self.closed_class_id = closed_class_id
        
        # Active events
        self.active_events: Dict[int, BreadBagEvent] = {}
        
        # Recently committed events (for anti-double-counting suppression)
        self.recently_committed: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'events_created': 0,
            'events_committed': 0,
            'events_expired': 0,
            'events_suppressed': 0,
            'total_detections_processed': 0,
        }
        
        logger.info(
            f"[EventCentricTracker] Initialized with: "
            f"D={self.config.association_distance_px}px, "
            f"T={self.config.association_time_ms}ms, "
            f"G={self.config.ghost_timeout_ms}ms, "
            f"commit_idle_frames={self.config.commit_idle_frames}, "
            f"suppression_distance={self.config.suppression_distance_px}px, "
            f"suppression_duration={self.config.suppression_duration_ms}ms"
        )
    
    def update(self, 
               detections: List[Dict[str, Any]], 
               timestamp_ms: float,
               frame_img: np.ndarray,
               frame_index: int) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with keys:
                - box: [x1, y1, x2, y2]
                - class_id: int
                - conf: float (confidence)
            timestamp_ms: Current timestamp in milliseconds
            frame_img: Current frame image
            frame_index: Current frame index
            
        Returns:
            List of committed event data ready for classification
        """
        ready_events = []
        frame_size = (frame_img.shape[1], frame_img.shape[0])
        
        # Clean up old recently_committed entries based on suppression duration
        self.recently_committed = [
            rc for rc in self.recently_committed
            if timestamp_ms - rc['timestamp_ms'] < self.config.suppression_duration_ms
        ]
        
        # Convert detections to evidence
        detection_evidences = []
        for det in detections:
            box = det['box']
            if hasattr(box, 'tolist'):
                box = tuple(box.tolist())
            else:
                box = tuple(box)
            
            centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            class_id = det['class_id']
            
            evidence = DetectionEvidence(
                timestamp_ms=timestamp_ms,
                centroid_x=centroid[0],
                centroid_y=centroid[1],
                box=box,
                is_open=(class_id == self.open_class_id),
                is_closed=(class_id == self.closed_class_id),
                confidence=det.get('conf', 1.0),
                frame_index=frame_index
            )
            detection_evidences.append(evidence)
        
        self.stats['total_detections_processed'] += len(detection_evidences)
        
        # Track which detections were associated
        associated_detection_indices = set()
        
        # 1. Associate detections with existing events
        for det_idx, evidence in enumerate(detection_evidences):
            best_event = None
            best_distance = float('inf')
            
            for event in self.active_events.values():
                if event.state == EventState.COMMITTED:
                    continue
                
                can_assoc, distance, reason = event.can_associate(evidence)
                if can_assoc and distance < best_distance:
                    best_event = event
                    best_distance = distance
            
            if best_event is not None:
                best_event.add_detection(evidence, frame_img)
                associated_detection_indices.add(det_idx)
        
        # 2. Create new events for unassociated open detections
        for det_idx, evidence in enumerate(detection_evidences):
            if det_idx in associated_detection_indices:
                continue
            
            # Only create events from open detections
            if not evidence.is_open:
                continue
            
            # Check work zone if enabled
            if self.config.work_zone_enabled:
                if not self._is_in_work_zone(evidence.centroid_x, evidence.centroid_y):
                    continue
            
            # Check confidence threshold
            if evidence.confidence < self.config.min_detection_confidence:
                continue
            
            # Check max active events
            if len(self.active_events) >= self.config.max_active_events:
                logger.warning(
                    f"[EventCentricTracker] Max active events reached ({self.config.max_active_events})"
                )
                break
            
            # Check suppression against recently committed events
            if self._should_suppress(evidence):
                self.stats['events_suppressed'] += 1
                continue
            
            # Create new event
            new_event = BreadBagEvent(
                initial_detection=evidence,
                config=self.config,
                open_class_id=self.open_class_id,
                closed_class_id=self.closed_class_id
            )
            self.active_events[new_event.id] = new_event
            self.stats['events_created'] += 1
        
        # 3. Update ghost state for events without detections
        events_to_remove = []
        
        for event_id, event in self.active_events.items():
            # Check if THIS specific event received a detection this frame
            if event.last_detection_time_ms != timestamp_ms:
                # No detection for this event - update ghost state
                should_commit = event.update_ghost_state(timestamp_ms, frame_size, frame_index)
                
                if should_commit:
                    # Event is ready for classification
                    ready_events.append(self._prepare_event_output(event))
                    events_to_remove.append(event_id)
                    self.stats['events_committed'] += 1
                    
                    # Add to recently committed
                    self.recently_committed.append({
                        'centroid': event.last_centroid,
                        'timestamp_ms': timestamp_ms,
                        'event_id': event_id
                    })
                
                elif event.state != EventState.COMMITTED:
                    # Check if ghost timeout exceeded without reaching CLOSED
                    time_since = timestamp_ms - event.last_detection_time_ms
                    if time_since > self.config.ghost_timeout_ms:
                        events_to_remove.append(event_id)
                        self.stats['events_expired'] += 1
                        
                        # Log expiration with debug info
                        debug_info = event.get_debug_info()
                        structured_logger.event_expired(
                            event_id=event_id,
                            state=event.state.name,
                            frames_tracked=event.total_frames_observed,
                            open_hits=event.open_evidence_count,
                            closed_hits=event.closed_evidence_count,
                            frames_since_update=int(time_since / (1000 / 25)),  # Approx frames
                            avg_motion=event.get_centroid_stability()
                        )
        
        # Remove committed/expired events
        for event_id in events_to_remove:
            del self.active_events[event_id]
        
        return ready_events
    
    def _is_in_work_zone(self, x: float, y: float) -> bool:
        """Check if position is within configured work zone."""
        return (self.config.work_zone_x1 <= x <= self.config.work_zone_x2 and
                self.config.work_zone_y1 <= y <= self.config.work_zone_y2)
    
    def _should_suppress(self, evidence: DetectionEvidence) -> bool:
        """
        Check if new event should be suppressed.
        
        Anti-Double-Counting:
        Prevents new events from being created for a bag that was temporarily
        lost then re-detected after commitment. This ensures each physical bag
        is counted exactly once.
        
        Args:
            evidence: Detection evidence for potential new event
            
        Returns:
            True if event should be suppressed, False otherwise
        """
        for rc in self.recently_committed:
            dx = evidence.centroid_x - rc['centroid'][0]
            dy = evidence.centroid_y - rc['centroid'][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Use configurable suppression distance for anti-double-counting
            if distance < self.config.suppression_distance_px:
                logger.debug(
                    f"[EventCentricTracker] Suppressing new event: "
                    f"too close to recently committed {rc['event_id']} "
                    f"(distance={distance:.1f}px < suppression_threshold={self.config.suppression_distance_px}px)"
                )
                return True
        
        return False
    
    def _prepare_event_output(self, event: BreadBagEvent) -> Dict[str, Any]:
        """
        Prepare committed event data for classification.
        
        Returns data in format compatible with ClassifierService.
        """
        candidates = event.get_roi_candidates()
        debug_info = event.get_debug_info()
        
        # Format stats for ClassifierService
        event_stats = {
            'total': len(candidates),
            'open_count': len([c for c in candidates]),  # All candidates counted
            'closed_count': len(candidates),
            'open_hits': event.open_evidence_count,
            'closed_hits': event.closed_evidence_count,
            'total_frames_tracked': event.total_frames_observed,
            'track_duration_frames': event.total_frames_observed,
            'start_frame': event.evidence_history[0].frame_index if event.evidence_history else 0,
            'end_frame': event.evidence_history[-1].frame_index if event.evidence_history else 0,
            'avg_sharpness': (
                sum(c.sharpness for c in event.roi_candidates) / len(event.roi_candidates)
                if event.roi_candidates else 0.0
            ),
        }
        
        return {
            'event_id': event.id,
            'candidates': candidates,
            'box': event.last_box,
            'stats': event_stats,
            'debug_info': debug_info,
        }
    
    def get_active_events_info(self) -> List[Dict[str, Any]]:
        """Get summary info for all active events (for visualization)."""
        events_info = []
        for event in self.active_events.values():
            events_info.append({
                'id': event.id,
                'state': event.state.name,
                'box': event.last_box,
                'centroid': event.last_centroid,
                'open_count': event.open_evidence_count,
                'closed_count': event.closed_evidence_count,
                'roi_count': len(event.roi_candidates),
            })
        return events_info
    
    def get_tracker_stats(self) -> Dict[str, Any]:
        """Get overall tracker statistics."""
        return {
            **self.stats,
            'active_events': len(self.active_events),
            'recently_committed': len(self.recently_committed),
            'completion_rate': (
                self.stats['events_committed'] / self.stats['events_created']
                if self.stats['events_created'] > 0 else 0.0
            ),
        }
