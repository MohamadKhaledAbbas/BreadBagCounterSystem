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
        CLOSED -> COMMITTED (after exit timeout with no detections)
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
    - exit_timeout_ms: Time after CLOSED before COMMIT.
      Should ensure bag has left scene: 500-1000ms
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
    # Ghost Event Parameters (G from requirements)
    # ==========================================================================
    ghost_timeout_ms: float = 1000.0  # G: Keep event alive without detections
    
    # ==========================================================================
    # Exit and Counting Parameters
    # ==========================================================================
    exit_timeout_ms: float = 800.0    # Time after CLOSED before COMMIT
    exit_boundary_margin_px: int = 50  # Margin from frame edge for exit detection
    
    # ==========================================================================
    # State Transition Parameters (temporal stability)
    # ==========================================================================
    open_to_closing_time_ms: float = 100.0   # Min time in OPEN before CLOSING
    closing_stability_time_ms: float = 150.0  # Closed detections must persist this long
    closed_stability_time_ms: float = 200.0   # Min time in CLOSED before COMMIT eligible
    
    # Geometric stability thresholds
    centroid_stability_px: float = 30.0  # Max centroid movement for "stable"
    
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
        
    Counting Rule:
        Event is counted ONLY when:
        1. State == CLOSED
        2. No detections for exit_timeout_ms
        3. Last centroid near scene exit boundary
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
    
    def can_associate(self, detection: DetectionEvidence) -> Tuple[bool, float, str]:
        """
        Check if a detection can be associated with this event.
        
        Uses centroid distance and time gap - NO IoU or appearance features.
        
        Args:
            detection: Detection to check
            
        Returns:
            Tuple of (can_associate, distance, reason)
        """
        # Calculate centroid distance
        det_centroid = (detection.centroid_x, detection.centroid_y)
        dx = det_centroid[0] - self.last_centroid[0]
        dy = det_centroid[1] - self.last_centroid[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check distance threshold
        if distance > self.config.association_distance_px:
            return False, distance, f"distance_exceeded ({distance:.1f} > {self.config.association_distance_px})"
        
        # Check time gap
        time_gap_ms = detection.timestamp_ms - self.last_detection_time_ms
        if time_gap_ms > self.config.association_time_ms:
            return False, distance, f"time_gap_exceeded ({time_gap_ms:.1f}ms > {self.config.association_time_ms}ms)"
        
        return True, distance, "associated"
    
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
        
        # Update timing
        self.last_detection_time_ms = detection.timestamp_ms
        self.last_update_time_ms = detection.timestamp_ms
        self.total_frames_observed += 1
        
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
            # Can revert to OPEN if open evidence resumes
            
            # Check for reversion to OPEN
            recent_open = sum(1 for e in self.evidence_history[-3:] if e.is_open)
            if recent_open >= 2:
                self._transition_to(EventState.OPEN, detection.timestamp_ms,
                                    "open_evidence_resumed")
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
    
    def update_ghost_state(self, current_time_ms: float, frame_size: Tuple[int, int]) -> bool:
        """
        Update event when no detection is present (ghost state).
        
        Args:
            current_time_ms: Current timestamp in milliseconds
            frame_size: (width, height) of frame for exit boundary check
            
        Returns:
            True if event should be committed (counted), False otherwise
        """
        # Start gap tracking if not already
        if self.current_gap_start is None:
            self.current_gap_start = self.last_detection_time_ms
        
        time_since_detection_ms = current_time_ms - self.last_detection_time_ms
        self.last_update_time_ms = current_time_ms
        
        # Check if ghost timeout exceeded
        if time_since_detection_ms > self.config.ghost_timeout_ms:
            if self.state == EventState.CLOSED:
                # Check if near exit boundary
                if self._is_near_exit_boundary(frame_size):
                    self._transition_to(EventState.COMMITTED, current_time_ms,
                                        f"exit_timeout_near_boundary ({time_since_detection_ms:.0f}ms)")
                    self.commit_reason = "exit_boundary"
                    return True
                elif time_since_detection_ms > self.config.exit_timeout_ms:
                    # Still commit if exit timeout exceeded
                    self._transition_to(EventState.COMMITTED, current_time_ms,
                                        f"exit_timeout ({time_since_detection_ms:.0f}ms)")
                    self.commit_reason = "exit_timeout"
                    return True
            else:
                # Event expired without reaching CLOSED state
                logger.debug(
                    f"[Event:{self.id}] Expired in state {self.state.name} "
                    f"after {time_since_detection_ms:.0f}ms without detection"
                )
                return False  # Don't commit, just expire
        
        return False
    
    def _is_near_exit_boundary(self, frame_size: Tuple[int, int]) -> bool:
        """
        Check if last centroid is near frame exit boundary.
        
        Args:
            frame_size: (width, height) of frame
            
        Returns:
            True if centroid is within margin of frame boundary
        """
        width, height = frame_size
        margin = self.config.exit_boundary_margin_px
        cx, cy = self.last_centroid
        
        # Check if near any edge
        near_left = cx < margin
        near_right = cx > (width - margin)
        near_top = cy < margin
        near_bottom = cy > (height - margin)
        
        return near_left or near_right or near_top or near_bottom
    
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
    4. Exit-based counting - Count when bag leaves, not when it closes
    
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
        
        # Recently committed events (for suppression)
        self.recently_committed: List[Dict[str, Any]] = []
        self.commit_suppression_time_ms: float = 500.0  # Suppress new events near recent commits
        
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
            f"exit_timeout={self.config.exit_timeout_ms}ms"
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
        
        # Clean up old recently_committed entries
        self.recently_committed = [
            rc for rc in self.recently_committed
            if timestamp_ms - rc['timestamp_ms'] < self.commit_suppression_time_ms
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
            # Skip events that received a detection this frame
            received_detection = any(
                event.last_detection_time_ms == timestamp_ms
                for event in self.active_events.values()
            )
            
            if event.last_detection_time_ms != timestamp_ms:
                # No detection for this event - update ghost state
                should_commit = event.update_ghost_state(timestamp_ms, frame_size)
                
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
        
        Prevents duplicate events near recently committed ones.
        """
        for rc in self.recently_committed:
            dx = evidence.centroid_x - rc['centroid'][0]
            dy = evidence.centroid_y - rc['centroid'][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.config.association_distance_px:
                logger.debug(
                    f"[EventCentricTracker] Suppressing new event: "
                    f"too close to recently committed {rc['event_id']} "
                    f"(distance={distance:.1f}px)"
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
