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

PARALLEL HYBRID ASSOCIATION:
Detection-to-event association uses a parallel hybrid approach:
- Both centroid distance AND IoU are ALWAYS computed for every association attempt
- Association succeeds if EITHER criterion is met
- This provides robustness during:
  * Bag flips/spins: centroid may jump but IoU remains high
  * Fast slides: IoU may drop but centroid distance stays close
  * Partial occlusions: one metric may fail while the other succeeds

EVENT EXPIRATION:
- Events have a maximum lifetime (default 10 seconds)
- After max lifetime, events are automatically expired and counted
- This prevents events from staying active indefinitely when bags aren't removed
- Useful when workers don't remove bags fast enough from the work area

HARD CONSTRAINTS MET:
- NO visual appearance embeddings
- NO frame-based counting (uses millisecond-based timing)
- Tolerates rotation, deformation, and temporary occlusion

TARGET: ≥99.9% counting reliability (≤1 error per 1000 bags)
"""

import time
import math
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, replace
import logging  # Only for DEBUG level constant
import numpy as np
import cv2

from src.utils.AppLogging import logger, structured_logger
from src.utils.PipelineMetrics import pipeline_metrics
from src.config.tracking_config import tracking_config  # V4 Phase 3: For lazy_roi_cropping_enabled


# =============================================================================
# V4 Phase 3: Vectorized IoU Computation
# =============================================================================

def compute_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    V4 Phase 3: Vectorized IoU computation for multiple boxes.
    
    Computes IoU between each box in boxes1 and each box in boxes2 using
    numpy vectorization. Replaces O(n*m) loops with O(1) vectorized ops.
    
    Expected speedup: 2-3x faster for multiple boxes (10+ boxes)
    
    Args:
        boxes1: Array of shape (N, 4) with boxes in format (x1, y1, x2, y2)
        boxes2: Array of shape (M, 4) with boxes in format (x1, y1, x2, y2)
        
    Returns:
        IoU matrix of shape (N, M) where element [i, j] is IoU between boxes1[i] and boxes2[j]
        
    Example:
        >>> boxes1 = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
        >>> boxes2 = np.array([[15, 15, 25, 25], [35, 35, 45, 45]])
        >>> iou_matrix = compute_iou_batch(boxes1, boxes2)
        >>> # iou_matrix[0, 0] is IoU between boxes1[0] and boxes2[0]
    """
    # Reshape for broadcasting: (N, 1, 4) and (1, M, 4)
    boxes1 = boxes1[:, np.newaxis, :]  # Shape: (N, 1, 4)
    boxes2 = boxes2[np.newaxis, :, :]  # Shape: (1, M, 4)
    
    # Compute intersection coordinates
    x1_inter = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1_inter = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2_inter = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2_inter = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    # Compute intersection area
    inter_width = np.maximum(0.0, x2_inter - x1_inter)
    inter_height = np.maximum(0.0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Compute areas of boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Compute union area
    union_area = area1 + area2 - inter_area
    
    # Compute IoU (avoid division by zero)
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    
    return iou


# =============================================================================

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
    work_zone_enabled: bool = True
    work_zone_x1: int = 0      # Top-left X of work zone
    work_zone_y1: int = 0      # Top-left Y of work zone
    work_zone_x2: int = 1280   # Bottom-right X of work zone
    work_zone_y2: int = 650    # Bottom-right Y of work zone (moved up from 720)
    
    # Work zone enforcement during association (Issue #2 fix)
    enforce_work_zone_associations: bool = True
    """Prevent associations for detections outside work zone even for active events"""
    
    out_of_zone_grace_frames: int = 5
    """Number of frames an event can remain outside work zone before faster expiration"""

    fast_commit_on_out_of_zone: bool = True  # Commit (count) instead of expire if CLOSED
    
    # ==========================================================================
    # Event Association Parameters (D, T from requirements)
    # ==========================================================================
    association_distance_px: float = 80.0   # D: Max centroid distance for association (reduced from 100.0)
    association_time_ms: float = 400.0      # T: Max time gap for association
    
    # ==========================================================================
    # IoU-Based Association (complementary to centroid distance)
    # ==========================================================================
    # IoU provides robustness when centroid distance alone may fail (e.g., during
    # partial occlusion where box overlaps but centroid shifts significantly)
    iou_association_enabled: bool = True    # Enable IoU as additional association criterion
    iou_association_threshold: float = 0.4  # Min IoU to associate (increased from 0.3)
    
    # ==========================================================================
    # IoU Box Margin Expansion (for flip/spin scenarios)
    # ==========================================================================
    # During flip/spin, the bounding box may shift significantly but the bag is
    # still nearby. Expanded box IoU helps maintain association in these cases.
    iou_box_margin_enabled: bool = True     # Enable expanded box for IoU computation
    iou_box_margin_ratio: float = 0.25      # Expansion ratio (0.25 = 25% per side)
    iou_expanded_threshold: float = 0.15    # Min IoU with expanded box to associate
    
    # ==========================================================================
    # Velocity-Based Association (for fast movements during flip/throw)
    # ==========================================================================
    velocity_scaling_enabled: bool = True   # Enable velocity-based distance scaling
    velocity_scale_factor: float = 2.5      # Max multiplier for association distance
    max_association_distance_px: float = 180.0  # Absolute max association distance (reduced from 250.0)
    min_velocity_threshold: float = 0.01    # Min velocity (px/ms) to trigger scaling
    max_prediction_time_ms: float = 500.0   # Max time ahead to predict centroid
    
    # Hard constraints for preventing teleportation (Issue #1 fix)
    max_jump_distance_px: float = 200.0
    """Hard cap on centroid jump distance, even if IoU/expanded IoU passes.
    Prevents events from teleporting to distant detections during crowded scenes."""
    
    require_centroid_proximity_for_expanded_iou: bool = True
    """When True, expanded IoU associations still require reasonable centroid distance.
    Prevents expanded IoU from matching detections that are too far away."""
    
    max_centroid_distance_for_expanded_iou: float = 250.0
    """Maximum centroid distance allowed for expanded IoU associations (px)."""
    
    # ==========================================================================
    # Ghost Event Parameters (G from requirements)
    # ==========================================================================
    ghost_timeout_ms: Optional[float] = None  # G: Keep event alive without detections (deprecated - use frames)
    ghost_timeout_frames: int = 25  # Frame-based ghost timeout (default 25 frames @ 25fps = 1000ms)
    
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
    suppression_distance_px: float = 100.0   # Distance within which new events are suppressed (reduced from 150.0)
    suppression_duration_ms: Optional[float] = None  # Duration to suppress (deprecated - use frames)
    suppression_duration_frames: int = 38  # Frame-based suppression duration (38 frames @ 25fps = 1500ms)
    
    # Conditional suppression (Issue #3 fix)
    suppression_require_box_overlap: bool = True
    """When True, suppression requires IoU overlap with last committed box.
    Allows new bags to start immediately at same location if no box overlap."""
    
    suppression_iou_threshold: float = 0.10
    """Minimum IoU with last committed box to trigger suppression.
    Lower values = more aggressive suppression (reduced from 0.15)."""
    
    # ==========================================================================
    # Temporal Cooldown for New Event Creation
    # ==========================================================================
    # Prevents rapid event creation at the same location after commitment
    min_event_creation_interval_ms: Optional[float] = None  # Deprecated - use frames
    temporal_cooldown_frames: int = 10  # Frame-based cooldown (10 frames @ 25fps = 400ms)
    """Minimum frames before allowing new event creation at same location after commit."""
    
    temporal_cooldown_distance_px: float = 120.0
    """Distance within which temporal cooldown applies."""
    
    # ==========================================================================
    # Active Event Spatial Exclusion
    # ==========================================================================
    # Prevents creating new events when another active event already covers the area
    active_event_exclusion_distance_px: float = 60.0
    """Distance within which new events are blocked if an active event exists."""
    
    active_event_exclusion_iou: float = 0.25
    """IoU threshold - if detection overlaps active event by this much, don't create new event."""
    
    # ==========================================================================
    # Detection Clustering Parameters
    # ==========================================================================
    # Groups nearby unassociated detections to prevent duplicate event creation
    detection_cluster_distance_px: float = 80.0
    """Distance within which detections are clustered together."""
    
    # ==========================================================================
    # State Transition Parameters (temporal stability)
    # ==========================================================================
    open_to_closing_time_ms: Optional[float] = None  # Deprecated - use frames
    open_to_closing_frames: int = 3  # Min frames in OPEN before CLOSING (3 frames @ 25fps = 120ms)
    closing_stability_time_ms: Optional[float] = None  # Deprecated - use frames  
    closing_stability_frames: int = 4  # Closed detections must persist (4 frames @ 25fps = 160ms)
    closed_stability_time_ms: Optional[float] = None  # Deprecated - use frames
    closed_stability_frames: int = 5  # Min frames in CLOSED before COMMIT eligible (5 frames @ 25fps = 200ms)
    
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
    max_open_roi_samples: int = 15        # Max ROIs to collect while open
    max_closed_roi_samples: int = 5       # Max ROIs to collect while closed

    # ==========================================================================
    # Disambiguate Parameters
    # ==========================================================================
    disambiguation_small_threshold: float = 8200
    disambiguation_regular_threshold: float = 10_000
    penalty_for_roi_in_gray_zone: float = 0.2

    # ==========================================================================
    # Classification Voting Parameters
    # ==========================================================================
    min_voting_agreement_pct: float = 60.0    # Min % votes for same class
    confidence_margin_threshold: float = 0.15  # Min margin between top classes
    
    # ==========================================================================
    # Resource Limits
    # ==========================================================================
    max_active_events: int = 4
    
    # ==========================================================================
    # Max Event Lifetime (Force Expiration) - Stuck Event Fail-safe
    # ==========================================================================
    max_event_lifetime_ms: Optional[float] = None  # Deprecated - use frames
    max_event_lifetime_frames: int = 250  # Max frames event can exist (250 @ 25fps = 10 seconds)
    """
    Maximum lifetime for an event in frames.
    
    After this duration, the event will be expired and counted regardless of
    whether it's still on screen. This prevents events from staying active
    indefinitely when workers don't remove bags fast enough.
    
    Range: 125 - 750 frames (5-30 seconds @ 25fps)
    - Lower values: More aggressive cleanup, may count prematurely
    - Higher values: More patient, but events may accumulate
    
    Default: 250 frames (10 seconds @ 25fps)
    """
    
    # State-specific maximum lifetimes (stuck event fail-safes)
    max_open_state_frames: int = 150  # Max frames in OPEN state (150 @ 25fps = 6 seconds)
    max_closing_state_frames: int = 75  # Max frames in CLOSING state (75 @ 25fps = 3 seconds)
    max_closed_state_frames: int = 100  # Max frames in CLOSED state (100 @ 25fps = 4 seconds)
    
    # ==========================================================================
    # Logging Control Parameters
    # ==========================================================================
    min_gap_duration_for_logging_ms: float = 500.0
    """Minimum detection gap duration to log (reduces log flooding)"""
    
    min_candidates_for_logging: int = 3
    """Minimum candidate count to log association candidates (only log ambiguous cases)"""
    
    low_score_threshold: float = 0.4
    """Score threshold below which associations are logged (focus on low-confidence matches)"""
    
    # Match types that are always logged (noteworthy cases)
    noteworthy_match_types: tuple = (
    )

    # noteworthy_match_types: tuple = (
    #     'ghost_iou_match', 'ghost_centroid_match', 'ghost_both_match',
    #     'expanded_iou_match', 'ghost_expanded_iou_match'
    # )
    """Match types that are always logged as they indicate special recovery cases"""

    use_frame_timestamps: bool = False
    
    # ==========================================================================
    # Testing Mode Time Scaling (for slower processing environments)
    # ==========================================================================
    testing_time_scale_factor: float = 1.0
    """
    Multiplier for all time-based thresholds when running in testing/development mode.
    
    When processing speed is slower than real-time (e.g., Windows without BPU acceleration),
    this factor scales all millisecond-based timeouts to maintain equivalent behavior.
    
    For example, if testing processes at 5fps (200ms/frame) instead of production's 
    25fps (40ms/frame), set this to 5.0 to scale all timeouts accordingly.
    
    Default: 1.0 (no scaling, production behavior)
    """
    
    enable_auto_time_scaling: bool = False
    """
    Automatically calculate time scaling factor based on measured processing speed.
    Enabled by default on Windows, disabled on RDK.
    """
    
    # Auto-scaling configuration parameters
    auto_scaling_target_frame_time_ms: float = 40.0
    """
    Target frame time in milliseconds for auto-scaling calculation (default: 40ms = 25fps).
    The auto-scaling factor is calculated as: measured_frame_time / target_frame_time.
    """
    
    auto_scaling_warmup_frames: int = 100
    """
    Number of frames to process before calculating auto-scaling factor.
    Ensures stable measurements by allowing system to reach steady state.
    """
    
    auto_scaling_activation_threshold: float = 1.2
    """
    Minimum scale factor to activate auto-scaling.
    If calculated factor is below this threshold (processing close to real-time),
    no scaling is applied. Range: 1.1 - 2.0, default: 1.2
    """
    
    # Target FPS for ms-to-frames conversion
    target_fps: float = 25.0
    """Target FPS for converting millisecond thresholds to frame-based thresholds."""
    
    # ==========================================================================
    # V6 Performance & Reliability Optimization Parameters
    # ==========================================================================
    
    # Adaptive Ghost Timeout - scales with object velocity
    adaptive_ghost_timeout_enabled: bool = True
    """Enable velocity-based ghost timeout scaling."""
    
    adaptive_ghost_velocity_factor: float = 2.0
    """Velocity scaling factor (k) for adaptive ghost timeout."""
    
    adaptive_ghost_min_timeout_frames: int = 15
    """Minimum ghost timeout frames (floor for adaptive scaling)."""
    
    adaptive_ghost_max_timeout_frames: int = 75
    """Maximum ghost timeout frames (ceiling for adaptive scaling)."""
    
    # Temporal Decimation - skip redundant monitor updates
    temporal_decimation_enabled: bool = True
    """Enable temporal decimation to skip redundant monitor updates."""
    
    temporal_decimation_area_epsilon: float = 0.05
    """Area change threshold for temporal decimation (5%)."""
    
    temporal_decimation_centroid_delta_px: float = 5.0
    """Centroid shift threshold (pixels) for temporal decimation."""
    
    temporal_decimation_confidence_epsilon: float = 0.05
    """Confidence change threshold for temporal decimation."""
    
    temporal_decimation_max_skip_frames: int = 3
    """Maximum consecutive frames to skip before forcing an update."""
    
    # Multi-Stage Matching Early Rejection
    early_rejection_enabled: bool = True
    """Enable early rejection gates before IOU computation."""
    
    early_rejection_area_ratio_min: float = 0.4
    """Minimum area ratio for early rejection."""
    
    early_rejection_area_ratio_max: float = 2.5
    """Maximum area ratio for early rejection."""
    
    # Spatial Zones
    spatial_zones_enabled: bool = True
    """Enable explicit spatial zone definitions."""
    
    entry_zone_margin_px: int = 50
    """Margin from frame edges for entry zone constraint."""
    
    exit_zone_margin_px: int = 80
    """Margin from edges defining the exit zone."""
    
    # Retention Safety
    retention_safety_enabled: bool = True
    """Enable retention safety rule."""
    
    # ==========================================================================
    # Velocity Stability Gate for ROI Collection
    # ==========================================================================
    velocity_stability_gate_enabled: bool = True
    """Enable velocity stability gating for ROI collection."""
    
    velocity_stability_threshold: float = 0.15
    """Maximum velocity (pixels per millisecond) to consider position stable."""
    
    velocity_stability_min_duration_ms: float = 150.0
    """Minimum time (ms) the bag must remain stable before collecting ROIs."""
    
    def __post_init__(self):
        """
        Post-initialization to handle migration compatibility.
        
        1. If any _ms parameters are provided (not None), convert them to frame-based
           equivalents and log the conversion for transparency.
        2. If any _ms parameters are None (deprecated), compute them from frame-based
           equivalents to ensure all code paths have valid values.
        """
        ms_per_frame = 1000.0 / self.target_fps
        conversions = []
        
        # Ghost timeout conversion (bidirectional)
        if self.ghost_timeout_ms is not None:
            frames = int(round(self.ghost_timeout_ms / ms_per_frame))
            self.ghost_timeout_frames = frames
            conversions.append(f"ghost_timeout: {self.ghost_timeout_ms}ms → {frames} frames")
        else:
            # Compute ms from frames (for backward compatibility)
            self.ghost_timeout_ms = self.ghost_timeout_frames * ms_per_frame
        
        # Suppression duration conversion (bidirectional)
        if self.suppression_duration_ms is not None:
            frames = int(round(self.suppression_duration_ms / ms_per_frame))
            self.suppression_duration_frames = frames
            conversions.append(f"suppression_duration: {self.suppression_duration_ms}ms → {frames} frames")
        else:
            self.suppression_duration_ms = self.suppression_duration_frames * ms_per_frame
        
        # Temporal cooldown conversion (bidirectional)
        if self.min_event_creation_interval_ms is not None:
            frames = int(round(self.min_event_creation_interval_ms / ms_per_frame))
            self.temporal_cooldown_frames = frames
            conversions.append(f"temporal_cooldown: {self.min_event_creation_interval_ms}ms → {frames} frames")
        else:
            self.min_event_creation_interval_ms = self.temporal_cooldown_frames * ms_per_frame
        
        # State transition timeouts (bidirectional)
        if self.open_to_closing_time_ms is not None:
            frames = int(round(self.open_to_closing_time_ms / ms_per_frame))
            self.open_to_closing_frames = frames
            conversions.append(f"open_to_closing: {self.open_to_closing_time_ms}ms → {frames} frames")
        else:
            self.open_to_closing_time_ms = self.open_to_closing_frames * ms_per_frame
        
        if self.closing_stability_time_ms is not None:
            frames = int(round(self.closing_stability_time_ms / ms_per_frame))
            self.closing_stability_frames = frames
            conversions.append(f"closing_stability: {self.closing_stability_time_ms}ms → {frames} frames")
        else:
            self.closing_stability_time_ms = self.closing_stability_frames * ms_per_frame
        
        if self.closed_stability_time_ms is not None:
            frames = int(round(self.closed_stability_time_ms / ms_per_frame))
            self.closed_stability_frames = frames
            conversions.append(f"closed_stability: {self.closed_stability_time_ms}ms → {frames} frames")
        else:
            self.closed_stability_time_ms = self.closed_stability_frames * ms_per_frame
        
        # Max event lifetime conversion (bidirectional)
        if self.max_event_lifetime_ms is not None:
            frames = int(round(self.max_event_lifetime_ms / ms_per_frame))
            self.max_event_lifetime_frames = frames
            conversions.append(f"max_event_lifetime: {self.max_event_lifetime_ms}ms → {frames} frames")
        else:
            self.max_event_lifetime_ms = self.max_event_lifetime_frames * ms_per_frame
        
        # Log conversions if any were performed
        if conversions:
            logger.info(
                f"[EventConfig] Converted time-based thresholds to frame-based (target_fps={self.target_fps}):"
            )
            for conv in conversions:
                logger.info(f"  - {conv}")


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
@dataclass
class ROICandidate:
    """
    ROI candidate with quality metrics for classification.
    
    V4 Phase 3: Supports lazy ROI cropping for memory and CPU efficiency.
    When lazy_roi_cropping_enabled=True, ROI is not cropped immediately but stored
    as metadata (box + frame reference) and cropped on-demand when needed for classification.
    """
    # V4 Phase 3: Optional ROI (None when lazy cropping enabled)
    roi: Optional[np.ndarray]
    sharpness: float
    quality: float  # composite quality score (lightweight to compute)
    size: Tuple[int, int]  # width, height
    timestamp_ms: float
    frame_index: int
    centroid_stability: float  # How stable the centroid was when captured
    confidence: float
    is_open: bool  # label the ROI as coming from open evidence
    is_closed: bool  # label the ROI as coming from closed evidence
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2) for disambiguation
    
    # V4 Phase 3: Lazy cropping metadata
    frame_ref: Optional[np.ndarray] = None  # Reference to frame (only when lazy=True)
    lazy: bool = False  # True if ROI needs to be cropped on-demand
    
    def get_roi(self) -> Optional[np.ndarray]:
        """
        V4 Phase 3: Get ROI, cropping on-demand if lazy=True.
        
        Returns:
            The ROI as numpy array, or None if cropping failed
        """
        if not self.lazy:
            # Already cropped, return immediately
            return self.roi
        
        # Lazy cropping: crop now from frame_ref
        if self.frame_ref is None or self.bbox is None:
            logger.warning(f"[LazyROI] Cannot crop: frame_ref={self.frame_ref is not None}, bbox={self.bbox is not None}")
            return None
        
        try:
            x1, y1, x2, y2 = map(int, self.bbox)
            h, w = self.frame_ref.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop and cache
            self.roi = self.frame_ref[y1:y2, x1:x2].copy()
            self.frame_ref = None  # Release frame reference
            self.lazy = False  # No longer lazy
            
            return self.roi
        except Exception as e:
            logger.error(f"[LazyROI] Crop failed: {e}")
            return None

    def __str__(self):
        lazy_str = " (lazy)" if self.lazy else ""
        return (f"Candidate{lazy_str}: sharpness = {self.sharpness}, quality = {self.quality:.3f}, "
                f"size = {self.size}, confidence = {self.confidence}, is_open = {self.is_open},"
                f" is_closed = {self.is_closed}, bbox = {self.bbox}")


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
        self.state_enter_frame_index = initial_detection.frame_index  # Frame-based state tracking
        self.state_enter_evidence_idx = 0  # Track which evidence index we entered current state
        
        # Temporal tracking
        self.created_at_ms = initial_detection.timestamp_ms
        self.created_at_frame_index = initial_detection.frame_index  # Frame-based creation tracking
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
        
        # ROI collection (during OPEN or CLOSED state)
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
        
        # Out-of-zone tracking (Issue #2 fix)
        self.frames_out_of_zone = 0
        self.out_of_zone_since_ms: Optional[float] = None
        
        # Velocity Stability Gate tracking
        # Tracks how long the bag has been "stable" (velocity below threshold)
        self.stability_duration_ms: float = 0.0  # Time spent below velocity threshold
        self.last_stability_check_time_ms: float = initial_detection.timestamp_ms
        self.is_velocity_stable: bool = True  # Start as stable (no movement yet)
        
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
    
    def _update_velocity_stability(self, current_time_ms: float) -> None:
        """
        Update velocity stability tracking.
        
        This implements the "Time-To-Live" (TTL) gate for ROI collection:
        - If velocity > threshold, reset stability_duration_ms to 0
        - If velocity < threshold, increment stability_duration_ms by time_delta
        
        Only ROIs collected when stability_duration_ms > min_duration_ms.
        """
        if not self.config.velocity_stability_gate_enabled:
            return
        
        velocity_mag = self.get_velocity_magnitude()
        time_delta = current_time_ms - self.last_stability_check_time_ms
        
        if velocity_mag > self.config.velocity_stability_threshold:
            # Velocity exceeds threshold - reset stability tracking
            self.stability_duration_ms = 0.0
            self.is_velocity_stable = False
        else:
            # Velocity below threshold - accumulate stability time
            self.stability_duration_ms += max(0, time_delta)
            # Check if we've been stable long enough
            if self.stability_duration_ms >= self.config.velocity_stability_min_duration_ms:
                self.is_velocity_stable = True
        
        self.last_stability_check_time_ms = current_time_ms
    
    def is_stable_for_roi_collection(self) -> bool:
        """
        Check if the event is stable enough for ROI collection.
        
        Returns True if:
        - Velocity stability gate is disabled, OR
        - Velocity has been below threshold for >= min_duration_ms
        
        This ensures bags have truly settled before collecting ROIs,
        preventing blurry images from vibrating or moving bags.
        """
        if not self.config.velocity_stability_gate_enabled:
            return True
        return self.is_velocity_stable
    
    def get_adaptive_ghost_timeout_frames(self) -> int:
        """
        V6: Get adaptive ghost timeout based on recent velocity.
        
        Formula: ghost_timeout = base_timeout + k * velocity_magnitude
        
        Benefits:
        - Spinning/rotating objects get longer timeout to survive occlusions
        - Thrown/fast objects terminate quickly to prevent stale events
        - More responsive to object motion dynamics
        
        Returns:
            Ghost timeout in frames (clamped to min/max bounds)
        """
        if not self.config.adaptive_ghost_timeout_enabled:
            return self.config.ghost_timeout_frames
        
        base_timeout = self.config.ghost_timeout_frames
        velocity_mag = self.get_velocity_magnitude()
        k = self.config.adaptive_ghost_velocity_factor
        
        # Scale factor: higher velocity = longer timeout (up to a point)
        # velocity_mag is in px/ms, convert to more meaningful scale
        # A typical fast movement might be 0.5 px/ms (500 px/s)
        # We want velocity of 0.5 px/ms to add roughly 10-15 frames
        velocity_scale = velocity_mag * 1000.0  # Convert to px/s
        additional_frames = int(k * velocity_scale / 50.0)  # 50 px/s = 1 frame addition
        
        adaptive_timeout = base_timeout + additional_frames
        
        # Clamp to configured bounds
        adaptive_timeout = max(
            self.config.adaptive_ghost_min_timeout_frames,
            min(adaptive_timeout, self.config.adaptive_ghost_max_timeout_frames)
        )
        
        return adaptive_timeout
    
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
    
    def _expand_box(self, box: Tuple[float, float, float, float], 
                    ratio: float) -> Tuple[float, float, float, float]:
        """
        Expand a bounding box by a given ratio on all sides.
        
        This is used for expanded-box IoU computation during flip/spin scenarios
        where the bounding box may shift significantly but the bag is still nearby.
        
        Args:
            box: Original box (x1, y1, x2, y2)
            ratio: Expansion ratio (e.g., 0.25 = 25% expansion on each side)
            
        Returns:
            Expanded box (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Calculate expansion amount
        expand_w = width * ratio
        expand_h = height * ratio
        
        # Expand the box uniformly on all sides
        new_x1 = x1 - expand_w
        new_y1 = y1 - expand_h
        new_x2 = x2 + expand_w
        new_y2 = y2 + expand_h
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    def can_associate(self, detection: DetectionEvidence) -> Tuple[bool, float, str, float]:
        """
        Check if a detection can be associated with this event using multi-stage matching.
        
        V6 MULTI-STAGE MATCHING PIPELINE (in order):
        1. Ghost timeout check (instant rejection - cheapest)
        2. Centroid distance gate (cheap)
        3. Area ratio gate (cheap) 
        4. IOU computation (expensive - only if above pass)
        
        This approach provides:
        - Early rejection of most candidates (cheap checks first)
        - IOU only computed on viable candidates
        - Significant CPU cost reduction
        
        Args:
            detection: Detection to check
            
        Returns:
            Tuple of (can_associate, distance, reason, iou_value)
            - can_associate: True if detection can be associated with this event
            - distance: Centroid distance in pixels
            - reason: Human-readable reason for the decision
            - iou_value: Computed IoU value (0.0-1.0) for association scoring
        """
        det_centroid = (detection.centroid_x, detection.centroid_y)
        
        # Calculate time gap
        time_gap_ms = detection.timestamp_ms - self.last_detection_time_ms
        
        # ==========================================================================
        # STAGE 1: Ghost timeout check (cheapest - instant rejection)
        # ==========================================================================
        # Use adaptive ghost timeout if enabled
        effective_ghost_timeout_ms = self.config.ghost_timeout_ms
        if self.config.adaptive_ghost_timeout_enabled:
            effective_ghost_timeout_frames = self.get_adaptive_ghost_timeout_frames()
            effective_ghost_timeout_ms = effective_ghost_timeout_frames * (1000.0 / self.config.target_fps)
        
        if time_gap_ms > effective_ghost_timeout_ms:
            # Time gap exceeds ghost timeout - cannot associate
            # Return 0.0 for distance since we didn't compute it (early rejection)
            reason = f"time_gap_exceeded ({time_gap_ms:.1f}ms > {effective_ghost_timeout_ms:.1f}ms)"
            return False, 0.0, reason, 0.0
        
        # ==========================================================================
        # STAGE 2: Centroid distance gate (cheap)
        # ==========================================================================
        # Calculate base association distance threshold
        base_distance_threshold = self.config.association_distance_px
        scaled_threshold = base_distance_threshold
        velocity_mag = 0.0
        
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
        
        # Calculate distance to last centroid
        dx = det_centroid[0] - self.last_centroid[0]
        dy = det_centroid[1] - self.last_centroid[1]
        distance_to_last = math.sqrt(dx*dx + dy*dy)
        distance = distance_to_last
        
        # Also try distance to predicted position (for fast movements)
        if self.config.velocity_scaling_enabled and len(self.centroid_history) >= 2:
            pred_centroid = self.predict_centroid(detection.timestamp_ms)
            dx_pred = det_centroid[0] - pred_centroid[0]
            dy_pred = det_centroid[1] - pred_centroid[1]
            distance_to_pred = math.sqrt(dx_pred*dx_pred + dy_pred*dy_pred)
            # Use the smaller of the two distances
            distance = min(distance_to_last, distance_to_pred)
        
        # Check both criteria (centroid first - cheap)
        centroid_match = distance <= scaled_threshold
        
        # ==========================================================================
        # STAGE 3: Area ratio gate (cheap - before expensive IOU)
        # ==========================================================================
        if self.config.early_rejection_enabled and self.last_box is not None:
            # Compute areas
            last_area = (self.last_box[2] - self.last_box[0]) * (self.last_box[3] - self.last_box[1])
            det_area = (detection.box[2] - detection.box[0]) * (detection.box[3] - detection.box[1])
            
            if last_area > 0 and det_area > 0:
                area_ratio = min(last_area, det_area) / max(last_area, det_area)
                area_ratio_inverse = max(last_area, det_area) / min(last_area, det_area)
                
                # Early rejection if area ratio is too extreme (both min and max checks)
                area_too_different = (
                    area_ratio < self.config.early_rejection_area_ratio_min or
                    area_ratio_inverse > self.config.early_rejection_area_ratio_max
                )
                
                if area_too_different:
                    # Areas are too different - likely different objects
                    # Reject regardless of centroid match (true early rejection)
                    reason = f"area_ratio_rejected (ratio={area_ratio:.2f}, inverse={area_ratio_inverse:.2f})"
                    return False, distance_to_last, reason, 0.0
        
        # ==========================================================================
        # STAGE 4: IOU computation (expensive - only if above checks pass)
        # ==========================================================================
        iou_value = 0.0
        iou_expanded = 0.0
        expanded_iou_match = False
        
        if self.last_box is not None:
            # Compute standard IoU
            iou_value = self._compute_iou(self.last_box, detection.box)
            
            # Compute expanded-box IoU if enabled (for flip/spin scenarios)
            # This helps maintain tracking when the box shifts significantly during rotation
            if self.config.iou_box_margin_enabled:
                expanded_box = self._expand_box(self.last_box, self.config.iou_box_margin_ratio)
                iou_expanded = self._compute_iou(expanded_box, detection.box)
                expanded_iou_match = iou_expanded >= self.config.iou_expanded_threshold
            
            # Note: Detailed IoU calculation debug removed to reduce log flooding
            # IoU values are logged in the hybrid_association_attempt call below
        
        # ISSUE #1 FIX: Hard cap on centroid jump distance
        # This prevents teleportation even if IoU/expanded IoU allows it
        if distance_to_last > self.config.max_jump_distance_px:
            match_type = "jump_distance_exceeded"
            metrics_detail = (
                f"dist={distance_to_last:.1f}px > max_jump={self.config.max_jump_distance_px}px"
            )
            reason = f"{match_type} ({metrics_detail})"
            return False, distance_to_last, reason, iou_value
        
        # Check IOU match
        iou_match = self.config.iou_association_enabled and iou_value >= self.config.iou_association_threshold
        
        # ISSUE #1 FIX: Expanded IoU still requires reasonable centroid proximity
        if self.config.require_centroid_proximity_for_expanded_iou and expanded_iou_match:
            if distance_to_last > self.config.max_centroid_distance_for_expanded_iou:
                # Expanded IoU passed but centroid is too far - reject to prevent teleportation
                expanded_iou_match = False
        
        # Determine time windows for association
        # - within_association_window: normal matching (higher reliability)
        # - within_ghost_window: ghost reattachment allowed (event is "alive but lost")
        within_association_window = time_gap_ms <= self.config.association_time_ms
        within_ghost_window = time_gap_ms <= effective_ghost_timeout_ms
        
        # Determine match type for structured logging
        # Priority:
        # 1. Check time windows (reject if beyond ghost_timeout)
        # 2. Normal association within association_time_ms
        # 3. Ghost reattachment within ghost_timeout_ms (if IoU or centroid match)
        # 4. Expanded IoU as fallback
        
        if not within_ghost_window:
            # Time gap exceeds ghost timeout - cannot associate
            match_type = "time_exceeded"
            associated = False
        elif within_association_window:
            # Normal association window - all match types allowed
            if centroid_match and iou_match:
                match_type = "both_match"
                associated = True
            elif centroid_match:
                match_type = "centroid_match"
                associated = True
            elif iou_match:
                match_type = "iou_match"
                associated = True
            elif expanded_iou_match:
                # Expanded IoU match is a fallback for flip/spin scenarios
                match_type = "expanded_iou_match"
                associated = True
                iou_value = iou_expanded
            else:
                match_type = "no_match"
                associated = False
        else:
            # Ghost reattachment window (association_time_ms < gap <= ghost_timeout_ms)
            # Allow reattachment if IoU or centroid matches - event was alive but lost
            if centroid_match and iou_match:
                match_type = "ghost_both_match"
                associated = True
            elif centroid_match:
                match_type = "ghost_centroid_match"
                associated = True
            elif iou_match:
                match_type = "ghost_iou_match"
                associated = True
            elif expanded_iou_match:
                match_type = "ghost_expanded_iou_match"
                associated = True
                iou_value = iou_expanded
            else:
                match_type = "no_match"
                associated = False
        
        # Only log association attempts that fail or are noteworthy (not every successful match)
        # This reduces log flooding while keeping important debug information
        # if not associated or match_type in self.config.noteworthy_match_types:
        #     structured_logger.hybrid_association_attempt(
        #         event_id=self.id,
        #         detection_centroid=det_centroid,
        #         event_centroid=self.last_centroid,
        #         distance_px=distance,
        #         distance_threshold=scaled_threshold,
        #         iou_value=iou_value,
        #         iou_threshold=self.config.iou_association_threshold,
        #         time_gap_ms=time_gap_ms,
        #         centroid_match=centroid_match,
        #         iou_match=iou_match,
        #         associated=associated,
        #         match_type=match_type,
        #         velocity_mag=velocity_mag if velocity_mag > self.config.min_velocity_threshold else None,
        #         base_threshold=base_distance_threshold,
        #     )
        
        # Build detailed reason string for return value
        metrics_detail = (
            f"dist={distance:.1f}px (thresh={scaled_threshold:.1f}px), "
            f"iou={iou_value:.2f} (thresh={self.config.iou_association_threshold}), "
            f"time_gap={time_gap_ms:.1f}ms (assoc_thresh={self.config.association_time_ms}ms, "
            f"ghost_thresh={self.config.ghost_timeout_ms}ms)"
        )
        
        # Include expanded IoU info if relevant
        if self.config.iou_box_margin_enabled and iou_expanded > 0:
            metrics_detail += f", iou_expanded={iou_expanded:.2f} (thresh={self.config.iou_expanded_threshold})"
        
        if velocity_mag > self.config.min_velocity_threshold:
            metrics_detail += f", velocity={velocity_mag:.3f}px/ms"
        
        if match_type == "time_exceeded":
            reason = f"time_gap_exceeded ({time_gap_ms:.1f}ms > {self.config.ghost_timeout_ms}ms) | {metrics_detail}"
        else:
            reason = f"{match_type} ({metrics_detail})"
        
        return associated, distance, reason, iou_value
    
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
            # Only log significant detection gaps to reduce log flooding
            if gap_duration > self.config.min_gap_duration_for_logging_ms:
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
                
                # Update velocity stability tracking
                self._update_velocity_stability(detection.timestamp_ms)
        
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
        
        # ISSUE #2 FIX: Reset out-of-zone tracking when detection is received
        # (Detection was associated, so event is being tracked)
        self.out_of_zone_since_ms = None
        self.frames_out_of_zone = 0
        
        # Process state transitions based on evidence
        self._process_state_transition(detection)

        # Collect ROI if in OPEN or CLOSED state
        if self.state in (EventState.OPEN, EventState.CLOSED) and frame_img is not None:
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
            # Requires: min frames in OPEN + closed evidence
            frames_in_state = detection.frame_index - self.state_enter_frame_index
            if (frames_in_state >= self.config.open_to_closing_frames and
                self.closed_evidence_count >= 1 and
                self.open_evidence_count >= self.config.min_open_evidence_count):
                self._transition_to(EventState.CLOSING, detection.timestamp_ms, 
                                    "closed_evidence_detected", detection.frame_index)
        
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
                                        f"open_evidence_resumed ({recent_open}/{len(recent_evidence)} open)",
                                        detection.frame_index)
                    return
            
            # Check for progression to CLOSED using frames
            frames_in_state = detection.frame_index - self.state_enter_frame_index
            if (frames_in_state >= self.config.closing_stability_frames and
                self.closed_evidence_count >= self.config.min_closed_evidence_count):
                # Also check geometric stability
                stability = self.get_centroid_stability()
                if stability <= self.config.centroid_stability_px:
                    self._transition_to(EventState.CLOSED, detection.timestamp_ms,
                                        f"closing_stable (stability={stability:.1f}px)",
                                        detection.frame_index)
        
        elif self.state == EventState.CLOSED:
            # CLOSED state - collecting ROIs, waiting for exit
            # Can only go to COMMITTED via update_ghost_state
            pass
    
    def _transition_to(self, new_state: EventState, timestamp_ms: float, trigger: str, frame_index: int = -1):
        """Record state transition with logging."""
        old_state = self.state
        self.state = new_state
        self.state_enter_time_ms = timestamp_ms
        self.state_enter_frame_index = frame_index if frame_index >= 0 else self.last_detection_frame_index
        self.state_enter_evidence_idx = len(self.evidence_history) - 1  # Track evidence index
        
        transition_record = {
            'timestamp_ms': timestamp_ms,
            'frame_index': self.state_enter_frame_index,
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

    def _compute_roi_quality(self, roi: np.ndarray, gray: np.ndarray, sharpness: float, raw_area: float) -> Tuple[float, float, float, float, float]:
        """
        Lightweight ROI quality score:
         - sharpness (variance of Laplacian)               -> focus / blur
        - edge density (mean abs Sobel)                   -> text/texture presence
        - entropy (histogram entropy, 32 bins)            -> richness/texture
        - contrast (stddev)                               -> usable dynamic range
        - glare penalty (% of near-white pixels)          -> reduce specular highlights
        All ops are O(pixels) on the already-cropped ROI.
        """

        # Edge density (Sobel)
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        edge_density = float(np.mean(np.abs(sobel_x)) + np.mean(np.abs(sobel_y)))

        # Entropy (coarse 32-bin histogram to stay light)
        hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
        p = hist + 1e-12
        entropy = float(-1 * np.sum(p * np.log2(p)))

        contrast = float(gray.std())
        glare_ratio = float(np.mean(gray > 245))  # near-white pixels

        # image is BGR, so convert to HSV for better color perception
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_metric = float(np.std(hsv[..., 1]))  # Standard deviation of Saturation channel as a simple color diversity proxy

        # Normalize to [0,1] with soft reference values tuned for typical 720p crops
        sharp_norm = min(sharpness / (self.config.min_roi_sharpness * 1.5), 1.0)
        edge_norm = min(edge_density / 25.0, 1.0)
        entropy_norm = min(entropy / 5.0, 1.0)  # entropy of 5 is already rich
        contrast_norm = min(contrast / 60.0, 1.0)
        color_norm = min(color_metric / 20.0, 1.0)  # Adjust denominator to match your empirical range
        glare_penalty = min(glare_ratio * 2.0, 0.3)  # cap penalty

        # === Area penalty for gray zone ===
        small_thresh = self.config.disambiguation_small_threshold
        regular_thresh = self.config.disambiguation_regular_threshold
        fixed_penalty = self.config.penalty_for_roi_in_gray_zone  # Select/tune based on logs/experiments

        if small_thresh <= raw_area <= regular_thresh:
            area_penalty = fixed_penalty
        else:
            area_penalty = 0.0

        quality = (
            0.40 * sharp_norm +
            0.18 * edge_norm +
            0.17 * entropy_norm +
            0.12 * contrast_norm +
            0.13 * color_norm -  # <- new colorfulness component
            glare_penalty -
            area_penalty
        )

        return quality, edge_density, entropy, contrast, glare_ratio

    def _try_collect_roi(self, detection: DetectionEvidence, frame_img: np.ndarray):
        """
        V4 Phase 3: Supports lazy ROI cropping for memory and CPU efficiency.
        V7.3: Enhanced validation for invalid crops, aspect ratios, and glare detection.
        V8: Velocity stability gate - only collect ROIs when bag has settled.
        
        When lazy_roi_cropping_enabled=True:
        - ROI is not cropped immediately
        - Only metadata (box, frame reference, quality) is stored
        - Actual cropping happens on-demand when event is ready for classification
        
        Benefits:
        - Reduces memory bandwidth (no immediate cropping)
        - Reduces CPU overhead (only crop what's needed)
        - Events that expire never trigger cropping
        - Expected 30-50% reduction in monitor processing time
        
        V7.3 Validation Improvements:
        - Minimum width/height checks after clamping
        - Aspect ratio validation (reject extreme ratios)
        - Glare/overexposure detection
        - Empty crop detection
        - Frame reference validation in lazy mode
        
        V8 Velocity Stability Gate:
        - Only collect ROIs when bag has been stable for >= min_duration_ms
        - Prevents blurry ROIs from vibrating or moving bags
        """
        # V8: Velocity Stability Gate - check if bag has settled before collecting ROI
        if not self.is_stable_for_roi_collection():
            # Bag is still moving/vibrating - skip ROI collection
            pipeline_metrics.record_roi_quality(False, 0.0, "velocity_unstable")
            return
        
        # Determine class-specific caps (fallback to legacy max_roi_samples)
        max_open_cap = getattr(self.config, "max_open_roi_samples", self.config.max_roi_samples)
        max_closed_cap = getattr(self.config, "max_closed_roi_samples", self.config.max_roi_samples)

        # Decide ROI class; skip if neither open nor closed
        roi_is_open = detection.is_open
        roi_is_closed = detection.is_closed
        if not (roi_is_open or roi_is_closed):
            return

        # V7.3: Clamp bbox and validate
        x1, y1, x2, y2 = map(int, detection.box)
        h, w = frame_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # V7.3: Guard against invalid crops (empty or too small after clamping)
        MIN_WIDTH = 20  # Minimum acceptable width
        MIN_HEIGHT = 20  # Minimum acceptable height
        
        if roi_width < MIN_WIDTH or roi_height < MIN_HEIGHT:
            logger.debug(
                f"[ROI] Rejected: invalid crop dimensions after clamping "
                f"(w={roi_width}, h={roi_height}, min={MIN_WIDTH})"
            )
            pipeline_metrics.record_roi_quality(False, 0.0, "invalid_dimensions")
            return
        
        # V7.3: Check aspect ratio (reject extreme ratios)
        MAX_ASPECT_RATIO = 4.0  # Maximum ratio (width/height or height/width)
        aspect_ratio = max(roi_width, roi_height) / min(roi_width, roi_height)
        
        if aspect_ratio > MAX_ASPECT_RATIO:
            logger.debug(
                f"[ROI] Rejected: extreme aspect ratio "
                f"(ratio={aspect_ratio:.2f}, max={MAX_ASPECT_RATIO}, w={roi_width}, h={roi_height})"
            )
            pipeline_metrics.record_roi_quality(False, 0.0, "aspect_ratio")
            return

        # Size check (standard minimum)
        if roi_width < self.config.min_roi_size or roi_height < self.config.min_roi_size:
            pipeline_metrics.record_roi_quality(False, 0.0, "size")
            return

        # V4 Phase 3: Check if lazy cropping is enabled
        lazy_cropping = tracking_config.lazy_roi_cropping_enabled
        
        if lazy_cropping:
            # V7.3: Validate frame reference in lazy mode
            if frame_img is None:
                logger.warning("[ROI] CRITICAL: frame_ref is None in lazy mode, cannot crop ROI")
                pipeline_metrics.record_roi_quality(False, 0.0, "null_frame_ref")
                return
            
            # Lazy mode: Store metadata only, compute quality from small sample
            # Sample a small region for quality estimation (center 50% of ROI for better estimates)
            sample_x1 = max(x1, x1 + roi_width // 4)
            sample_y1 = max(y1, y1 + roi_height // 4)
            sample_x2 = min(x2, x2 - roi_width // 4)
            sample_y2 = min(y2, y2 - roi_height // 4)
            
            # Validate sample bounds
            if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
                logger.debug(
                    f"[ROI] Rejected: invalid sample bounds "
                    f"(sample_x=[{sample_x1}, {sample_x2}], sample_y=[{sample_y1}, {sample_y2}])"
                )
                pipeline_metrics.record_roi_quality(False, 0.0, "invalid_sample")
                return
            
            try:
                roi_sample = frame_img[sample_y1:sample_y2, sample_x1:sample_x2].copy()
            except Exception as e:
                logger.error(f"[ROI] Failed to sample crop: {e}")
                pipeline_metrics.record_roi_quality(False, 0.0, "sample_error")
                return
            
            # Check if sample is empty
            if roi_sample.size == 0:
                logger.debug("[ROI] Rejected: empty sample crop")
                pipeline_metrics.record_roi_quality(False, 0.0, "empty_sample")
                return
            
            # Quick quality checks on sample
            mean_brightness = roi_sample.mean()
            if not (self.config.min_brightness <= mean_brightness <= self.config.max_brightness):
                pipeline_metrics.record_roi_quality(False, 0.0, "brightness")
                return
            
            # V7.3: Quick glare/overexposure check
            OVEREXPOSURE_THRESHOLD = 0.3  # 30% of pixels near white
            glare_pct = np.mean(roi_sample > 240)  # Count near-white pixels
            if glare_pct > OVEREXPOSURE_THRESHOLD:
                logger.debug(
                    f"[ROI] Rejected: overexposed/glare detected "
                    f"(glare_pct={glare_pct:.2f}, threshold={OVEREXPOSURE_THRESHOLD})"
                )
                pipeline_metrics.record_roi_quality(False, 0.0, "overexposure")
                return

            # Sharpness check on sample
            try:
                gray_sample = cv2.cvtColor(roi_sample, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray_sample, cv2.CV_64F).var()
            except Exception as e:
                logger.error(f"[ROI] Failed to compute sharpness: {e}")
                pipeline_metrics.record_roi_quality(False, 0.0, "sharpness_error")
                return

            if sharpness < self.config.min_roi_sharpness:
                pipeline_metrics.record_roi_quality(False, sharpness, "sharpness")
                return

            # Estimate quality from sample (will be refined when actually cropped)
            quality, edge_density, entropy, contrast, glare_ratio = self._compute_roi_quality(
                roi_sample, gray_sample, sharpness, (roi_width * roi_height)
            )
            pipeline_metrics.record_roi_quality(True, sharpness, None)
            
            # V7.3: Deduplication check - skip if similar ROI already exists
            if self._is_duplicate_roi(detection.box, quality):
                logger.debug(
                    f"[ROI] Rejected: duplicate ROI detected "
                    f"(quality={quality:.3f})"
                )
                pipeline_metrics.record_roi_quality(False, quality, "duplicate")
                return

            # Create lazy candidate (roi=None, frame_ref=frame_img)
            candidate = ROICandidate(
                roi=None,  # Not cropped yet
                sharpness=sharpness,
                quality=quality,
                size=(roi_width, roi_height),
                timestamp_ms=detection.timestamp_ms,
                frame_index=detection.frame_index,
                centroid_stability=self.get_centroid_stability(),
                confidence=detection.confidence,
                is_open=roi_is_open,
                is_closed=roi_is_closed,
                bbox=detection.box,
                frame_ref=frame_img,  # Store frame reference
                lazy=True  # Mark as lazy
            )
        else:
            # Legacy mode: Crop immediately
            try:
                roi = frame_img[y1:y2, x1:x2].copy()
            except Exception as e:
                logger.error(f"[ROI] Failed to crop: {e}")
                pipeline_metrics.record_roi_quality(False, 0.0, "crop_error")
                return
            
            # Check if crop is empty
            if roi.size == 0:
                logger.debug("[ROI] Rejected: empty crop")
                pipeline_metrics.record_roi_quality(False, 0.0, "empty_crop")
                return

            # Brightness check
            mean_brightness = roi.mean()
            if not (self.config.min_brightness <= mean_brightness <= self.config.max_brightness):
                pipeline_metrics.record_roi_quality(False, 0.0, "brightness")
                return
            
            # V7.3: Quick glare/overexposure check
            OVEREXPOSURE_THRESHOLD = 0.3  # 30% of pixels near white
            glare_pct = np.mean(roi > 240)
            if glare_pct > OVEREXPOSURE_THRESHOLD:
                logger.debug(
                    f"[ROI] Rejected: overexposed/glare detected "
                    f"(glare_pct={glare_pct:.2f}, threshold={OVEREXPOSURE_THRESHOLD})"
                )
                pipeline_metrics.record_roi_quality(False, 0.0, "overexposure")
                return

            # Sharpness check (variance of Laplacian)
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            except Exception as e:
                logger.error(f"[ROI] Failed to compute sharpness: {e}")
                pipeline_metrics.record_roi_quality(False, 0.0, "sharpness_error")
                return

            if sharpness < self.config.min_roi_sharpness:
                pipeline_metrics.record_roi_quality(False, sharpness, "sharpness")
                return

            # Lightweight composite quality
            quality, edge_density, entropy, contrast, glare_ratio = self._compute_roi_quality(
                roi, gray, sharpness, (roi_width * roi_height)
            )
            pipeline_metrics.record_roi_quality(True, sharpness, None)
            
            # V7.3: Deduplication check
            if self._is_duplicate_roi(detection.box, quality):
                logger.debug(
                    f"[ROI] Rejected: duplicate ROI detected "
                    f"(quality={quality:.3f})"
                )
                pipeline_metrics.record_roi_quality(False, quality, "duplicate")
                return

            candidate = ROICandidate(
                roi=roi,
                sharpness=sharpness,
            quality=quality,
            size=(roi_width, roi_height),
            timestamp_ms=detection.timestamp_ms,
            frame_index=detection.frame_index,
            centroid_stability=self.get_centroid_stability(),
            confidence=detection.confidence,
            is_open=roi_is_open,
            is_closed=roi_is_closed,
            bbox=detection.box  # Pass bbox for disambiguation
        )

        logger.debug(f"[ROI_CANDIDATE] added candidate = {candidate}")

        # Insert and keep best-per-class by composite quality
        self.roi_candidates.append(candidate)
        self.roi_candidates.sort(key=lambda x: x.quality, reverse=True)

        # Recompute counts
        self.open_roi_count = sum(1 for c in self.roi_candidates if c.is_open)
        self.closed_roi_count = sum(1 for c in self.roi_candidates if c.is_closed)

        # Enforce class caps by dropping the lowest-quality ROI of that class if over cap
        if roi_is_open and self.open_roi_count > max_open_cap:
            # drop worst open ROI
            logger.debug(f"[ROI_CANDIDATE] open_roi_count({self.open_roi_count}) > max_open_cap({max_open_cap})")
            worst_idx = min(
                (i for i, c in enumerate(self.roi_candidates) if c.is_open),
                key = lambda i: self.roi_candidates[i].quality)
            removed = self.roi_candidates.pop(worst_idx)
            logger.debug(f"[ROI_CANDIDATE] removed worst open candidate = {removed}")
            self.open_roi_count -= 1

        if roi_is_closed and self.closed_roi_count > max_closed_cap:
            # drop worst closed ROI
            logger.debug(f"[ROI_CANDIDATE] closed_roi_count({self.closed_roi_count}) > max_closed_cap({max_closed_cap})")
            worst_idx = min(
                (i for i, c in enumerate(self.roi_candidates) if c.is_closed),
                key = lambda i: self.roi_candidates[i].quality)
            removed = self.roi_candidates.pop(worst_idx)
            logger.debug(f"[ROI_CANDIDATE] removed worst closed candidate = {removed}")
            self.closed_roi_count -= 1
    
    def _is_duplicate_roi(self, new_bbox: Tuple[float, float, float, float], new_quality: float) -> bool:
        """
        V7.3: Check if new ROI candidate is a duplicate of existing ones.
        
        Deduplication criteria:
        - High IoU with existing ROI (>= 0.7)
        - Quality gain must exceed epsilon (0.05) to replace existing
        
        Args:
            new_bbox: Bounding box of new ROI candidate (x1, y1, x2, y2)
            new_quality: Quality score of new ROI candidate
            
        Returns:
            True if ROI should be rejected as duplicate, False otherwise
        """
        DUPLICATE_IOU_THRESHOLD = 0.7
        QUALITY_GAIN_EPSILON = 0.05
        
        for existing in self.roi_candidates:
            if existing.bbox is None:
                continue
            
            # Compute IoU with existing ROI
            iou = self._compute_iou_static(new_bbox, existing.bbox)
            
            if iou >= DUPLICATE_IOU_THRESHOLD:
                # High overlap - check if quality gain is significant
                quality_gain = new_quality - existing.quality
                if quality_gain < QUALITY_GAIN_EPSILON:
                    # Not enough quality gain - reject as duplicate
                    return True
        
        return False
    
    def _compute_iou_static(self, box1: Tuple[float, float, float, float], 
                           box2: Tuple[float, float, float, float]) -> float:
        """
        Compute IoU between two bounding boxes.
        
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
    
    def update_ghost_state(self, current_time_ms: float, frame_size: Tuple[int, int], 
                           current_frame_index: int = -1) -> Tuple[bool, str]:
        """
        Update event when no detection is present (ghost state).
        
        Commitment Logic (decoupled from ghost timeout):
        - Commit eligibility is evaluated as soon as:
          1. State is CLOSED
          2. frames_without_detection >= commit_idle_frames
          3. closed_ratio >= commit_min_closed_ratio
          4. (Optional) time_in_closed >= closed_stability_time_ms
        - Ghost timeout is used separately to expire non-CLOSED events
        
        Args:
            current_time_ms: Current timestamp in milliseconds
            frame_size: (width, height) of frame (kept for API compatibility)
            current_frame_index: Current frame index for idle tracking
            
        Returns:
            Tuple of (should_commit, status):
            - should_commit: True if event should be committed (counted)
            - status: 'commit', 'keep_alive', 'expire', or 'waiting'
        """
        # Start gap tracking if not already
        if self.current_gap_start is None:
            self.current_gap_start = self.last_detection_time_ms
        
        time_since_detection_ms = current_time_ms - self.last_detection_time_ms
        self.last_update_time_ms = current_time_ms
        
        # Update frames without detection counter
        if current_frame_index >= 0:
            self.frames_without_detection = current_frame_index - self.last_detection_frame_index
        
        # Calculate time in current state
        time_in_state_ms = current_time_ms - self.state_enter_time_ms
        
        # PRIORITY CHECK: Max event lifetime exceeded - force commit/expire (stuck event fail-safe)
        # This prevents events from staying active indefinitely when bags aren't removed
        event_lifetime_frames = current_frame_index - self.created_at_frame_index if current_frame_index >= 0 else 0
        if event_lifetime_frames > self.config.max_event_lifetime_frames:
            # Force commit if we have reasonable evidence, otherwise expire
            if self.state == EventState.CLOSED or self.closed_evidence_count >= self.config.min_closed_evidence_count:
                # Has closed evidence - commit it
                self._transition_to(EventState.COMMITTED, current_time_ms,
                                    f"max_lifetime_exceeded (lifetime={event_lifetime_frames} frames)")
                self.commit_reason = "max_lifetime"
                structured_logger.event_forced_close(
                    event_id=self.id,
                    state=self.state.name,
                    reason="max_lifetime",
                    lifetime_frames=event_lifetime_frames,
                    max_allowed_frames=self.config.max_event_lifetime_frames,
                    evidence={'open': self.open_evidence_count, 'closed': self.closed_evidence_count}
                )
                logger.info(
                    f"[Event:{self.id}] Max lifetime commit: bag counted after max lifetime "
                    f"(lifetime={event_lifetime_frames} frames, max={self.config.max_event_lifetime_frames} frames, "
                    f"state={self.state.name})"
                )
                return True, 'commit'
            else:
                # No closed evidence - just expire
                structured_logger.event_forced_close(
                    event_id=self.id,
                    state=self.state.name,
                    reason="max_lifetime_no_evidence",
                    lifetime_frames=event_lifetime_frames,
                    max_allowed_frames=self.config.max_event_lifetime_frames,
                    evidence={'open': self.open_evidence_count, 'closed': self.closed_evidence_count}
                )
                logger.info(
                    f"[Event:{self.id}] Max lifetime expired: no closed evidence "
                    f"(lifetime={event_lifetime_frames} frames, max={self.config.max_event_lifetime_frames} frames)"
                )
                return False, 'expire'
        
        # CHECK: State-specific stuck event fail-safes
        # Force transitions if event is stuck in a state for too long
        frames_in_state = current_frame_index - self.state_enter_frame_index if current_frame_index >= 0 else 0
        
        if self.state == EventState.OPEN and frames_in_state > self.config.max_open_state_frames:
            # Stuck in OPEN - force expire (unlikely to be a valid bag)
            structured_logger.event_forced_close(
                event_id=self.id,
                state="OPEN",
                reason="max_open_state_exceeded",
                frames_in_state=frames_in_state,
                max_allowed_frames=self.config.max_open_state_frames,
                evidence={'open': self.open_evidence_count, 'closed': self.closed_evidence_count}
            )
            logger.info(
                f"[Event:{self.id}] Stuck event expired: exceeded max OPEN state duration "
                f"(frames_in_state={frames_in_state}, max={self.config.max_open_state_frames})"
            )
            return False, 'expire'
        
        elif self.state == EventState.CLOSING and frames_in_state > self.config.max_closing_state_frames:
            # Stuck in CLOSING - force to CLOSED if sufficient evidence, otherwise expire
            if self.closed_evidence_count >= self.config.min_closed_evidence_count:
                self._transition_to(EventState.CLOSED, current_time_ms,
                                    f"max_closing_state_exceeded (frames={frames_in_state})")
                structured_logger.event_forced_close(
                    event_id=self.id,
                    state="CLOSING",
                    reason="max_closing_state_exceeded_to_closed",
                    frames_in_state=frames_in_state,
                    max_allowed_frames=self.config.max_closing_state_frames,
                    evidence={'open': self.open_evidence_count, 'closed': self.closed_evidence_count}
                )
                logger.info(
                    f"[Event:{self.id}] Stuck event forced to CLOSED: exceeded max CLOSING duration "
                    f"(frames_in_state={frames_in_state}, max={self.config.max_closing_state_frames})"
                )
                # Continue processing in CLOSED state
            else:
                # Not enough evidence - expire
                structured_logger.event_forced_close(
                    event_id=self.id,
                    state="CLOSING",
                    reason="max_closing_state_exceeded_expire",
                    frames_in_state=frames_in_state,
                    max_allowed_frames=self.config.max_closing_state_frames,
                    evidence={'open': self.open_evidence_count, 'closed': self.closed_evidence_count}
                )
                logger.info(
                    f"[Event:{self.id}] Stuck event expired: exceeded max CLOSING duration without sufficient evidence "
                    f"(frames_in_state={frames_in_state}, max={self.config.max_closing_state_frames})"
                )
                return False, 'expire'
        
        elif self.state == EventState.CLOSED and frames_in_state > self.config.max_closed_state_frames:
            # Stuck in CLOSED - force commit if sufficient evidence
            total_evidence = self.open_evidence_count + self.closed_evidence_count
            closed_ratio = self.closed_evidence_count / total_evidence if total_evidence > 0 else 0
            
            if closed_ratio >= self.config.commit_min_closed_ratio:
                self._transition_to(EventState.COMMITTED, current_time_ms,
                                    f"max_closed_state_exceeded (frames={frames_in_state})")
                self.commit_reason = "max_closed_state"
                structured_logger.event_forced_close(
                    event_id=self.id,
                    state="CLOSED",
                    reason="max_closed_state_exceeded_commit",
                    frames_in_state=frames_in_state,
                    max_allowed_frames=self.config.max_closed_state_frames,
                    evidence={'open': self.open_evidence_count, 'closed': self.closed_evidence_count}
                )
                logger.info(
                    f"[Event:{self.id}] Stuck event committed: exceeded max CLOSED duration "
                    f"(frames_in_state={frames_in_state}, max={self.config.max_closed_state_frames})"
                )
                return True, 'commit'
            else:
                # Not enough evidence - expire
                structured_logger.event_forced_close(
                    event_id=self.id,
                    state="CLOSED",
                    reason="max_closed_state_exceeded_expire",
                    frames_in_state=frames_in_state,
                    max_allowed_frames=self.config.max_closed_state_frames,
                    evidence={'open': self.open_evidence_count, 'closed': self.closed_evidence_count}
                )
                logger.info(
                    f"[Event:{self.id}] Stuck event expired: exceeded max CLOSED duration without sufficient evidence "
                    f"(frames_in_state={frames_in_state}, max={self.config.max_closed_state_frames})"
                )
                return False, 'expire'
        
        # FIRST: Check commit eligibility for CLOSED events (independent of ghost timeout)
        if self.state == EventState.CLOSED:
            # Check if enough idle frames have passed
            if self.frames_without_detection >= self.config.commit_idle_frames:
                # Verify we have sufficient closed evidence
                total_evidence = self.open_evidence_count + self.closed_evidence_count
                if total_evidence > 0:
                    closed_ratio = self.closed_evidence_count / total_evidence
                    
                    # Check closed ratio threshold
                    if closed_ratio >= self.config.commit_min_closed_ratio:
                        # Optional: Check time in CLOSED state for extra stability
                        if time_in_state_ms >= self.config.closed_stability_time_ms:
                            self._transition_to(EventState.COMMITTED, current_time_ms,
                                                f"idle_commit (idle={self.frames_without_detection} frames, "
                                                f"closed_ratio={closed_ratio:.2f}, "
                                                f"time_in_closed={time_in_state_ms:.0f}ms)")
                            self.commit_reason = "idle_commit"
                            logger.info(
                                f"[Event:{self.id}] Idle commit: bag counted after idle threshold "
                                f"(idle={self.frames_without_detection} frames, "
                                f"time_since_detection={time_since_detection_ms:.0f}ms, "
                                f"time_in_closed={time_in_state_ms:.0f}ms, "
                                f"centroid={self.last_centroid})"
                            )
                            return True, 'commit'
                        else:
                            # Need more time in CLOSED state
                            logger.debug(
                                f"[Event:{self.id}] CLOSED but waiting for stability "
                                f"(time_in_closed={time_in_state_ms:.0f}ms, "
                                f"required={self.config.closed_stability_time_ms}ms)"
                            )
                            return False, 'waiting'
                    else:
                        # Not enough closed evidence ratio
                        logger.debug(
                            f"[Event:{self.id}] CLOSED but insufficient closed ratio "
                            f"({closed_ratio:.2f} < {self.config.commit_min_closed_ratio})"
                        )
                        # Keep alive - might get more closed detections
                        return False, 'keep_alive'
            else:
                # Not enough idle frames yet
                logger.debug(
                    f"[Event:{self.id}] CLOSED but waiting for idle threshold "
                    f"(idle_frames={self.frames_without_detection}, "
                    f"required={self.config.commit_idle_frames})"
                )
                return False, 'waiting'
        
        # ISSUE #2 FIX: Check if event is out of work zone and expire faster
        if self.config.work_zone_enabled and self.config.enforce_work_zone_associations:
            # Check if centroid is out of zone
            in_zone = (self.config.work_zone_x1 <= self.last_centroid[0] <= self.config.work_zone_x2 and
                      self.config.work_zone_y1 <= self.last_centroid[1] <= self.config.work_zone_y2)
            
            if not in_zone:
                # Track how long event has been out of zone
                if self.out_of_zone_since_ms is None:
                    self.out_of_zone_since_ms = self.last_detection_time_ms
                    self.frames_out_of_zone = 0
                
                # Increment out-of-zone counter (independent of frames_without_detection)
                self.frames_out_of_zone += 1

                if self.frames_out_of_zone >= self.config.out_of_zone_grace_frames:
                    if self.state == EventState.CLOSED and getattr(self.config, 'fast_commit_on_out_of_zone', False):
                        # Commit instead of expire!
                        self._transition_to(EventState.COMMITTED, current_time_ms,
                                            f"out_of_zone_commit (frames={self.frames_out_of_zone})")
                        self.commit_reason = "out_of_zone_commit"
                        logger.info(
                            f"[Event:{self.id}] COMMITTED: out of work zone for {self.frames_out_of_zone} frames "
                            f"(grace={self.config.out_of_zone_grace_frames} frames, "
                            f"centroid={self.last_centroid})"
                        )
                        return True, 'commit'
                    else:
                        # Other states: expire as before
                        logger.info(
                            f"[Event:{self.id}] Expired: out of work zone for {self.frames_out_of_zone} frames "
                            f"(grace={self.config.out_of_zone_grace_frames} frames, "
                            f"centroid={self.last_centroid})"
                        )
                        return False, 'expire'
            else:
                # Back in zone - reset tracking
                self.out_of_zone_since_ms = None
                self.frames_out_of_zone = 0
        
        # SECOND: Check ghost timeout using adaptive frame-based threshold
        # V6: Use adaptive ghost timeout based on velocity
        effective_ghost_timeout = self.get_adaptive_ghost_timeout_frames()
        if self.frames_without_detection >= effective_ghost_timeout:
            # Ghost timeout exceeded - decide whether to commit or expire
            
            # Commit-on-ghost-expire for finalization states (CLOSING/CLOSED)
            # This treats ghost timeout as a throw-finalization window
            if self.state in [EventState.CLOSING, EventState.CLOSED]:
                # Safety gating: only commit if sufficient evidence exists
                total_evidence = self.open_evidence_count + self.closed_evidence_count
                has_sufficient_evidence = (
                    total_evidence >= 3 and  # At least 3 total detections
                    self.closed_evidence_count >= 1 and  # At least 1 closed detection
                    self.open_evidence_count >= self.config.min_open_evidence_count  # Saw it as open
                )
                
                if has_sufficient_evidence:
                    # Commit the event - bag likely thrown/removed
                    self._transition_to(EventState.COMMITTED, current_time_ms,
                                        f"ghost_commit (state={self.state.name}, "
                                        f"idle={self.frames_without_detection} frames, "
                                        f"open_ev={self.open_evidence_count}, closed_ev={self.closed_evidence_count})")
                    self.commit_reason = "ghost_finalization"
                    logger.info(
                        f"[Event:{self.id}] Ghost-finalization commit: bag counted after ghost timeout in {self.state.name} state "
                        f"(idle={self.frames_without_detection} frames, ghost_timeout={effective_ghost_timeout} frames, "
                        f"open_ev={self.open_evidence_count}, closed_ev={self.closed_evidence_count})"
                    )
                    return True, 'commit'
                else:
                    # Insufficient evidence - expire instead of commit
                    logger.debug(
                        f"[Event:{self.id}] Ghost expired in {self.state.name}: insufficient evidence "
                        f"(idle={self.frames_without_detection} frames, "
                        f"open_ev={self.open_evidence_count}, closed_ev={self.closed_evidence_count})"
                    )
                    return False, 'expire'
            else:
                # OPEN state: simply expire after ghost timeout
                logger.debug(
                    f"[Event:{self.id}] Ghost expired in state {self.state.name} "
                    f"after {self.frames_without_detection} frames without detection "
                    f"(ghost_timeout={effective_ghost_timeout} frames)"
                )
                return False, 'expire'
        
        # Event is still alive in ghost state, waiting for detection or state change
        return False, 'keep_alive'
    
    def get_roi_candidates(self) -> List[Dict[str, Any]]:
        """
        V4 Phase 3: Get ROI candidates for classification with lazy cropping support.
        V7.3: Enhanced with None ROI filtering and validation.
        
        For lazy candidates, this triggers on-demand cropping when the event
        is ready for classification (only after passing all quality gates).
        
        Returns candidates formatted for ClassifierService.
        """
        candidates = []
        none_roi_count = 0
        
        for idx, roi_cand in enumerate(self.roi_candidates):
            relative_time = idx / max(1, len(self.roi_candidates) - 1) if len(self.roi_candidates) > 1 else 0.5
            
            # V4 Phase 3: Get ROI (triggers lazy cropping if needed)
            roi = roi_cand.get_roi() if roi_cand.lazy else roi_cand.roi
            
            # V7.3: Filter out None ROIs with structured warning
            if roi is None:
                none_roi_count += 1
                logger.warning(
                    f"[ROI] Dropping None ROI candidate "
                    f"(idx={idx}, lazy={roi_cand.lazy}, "
                    f"frame_ref_present={roi_cand.frame_ref is not None if hasattr(roi_cand, 'frame_ref') else 'N/A'})"
                )
                continue
            
            candidates.append({
                'roi': roi,
                'sharpness': roi_cand.sharpness,
                'frame_index': roi_cand.frame_index,
                'bbox_area': roi_cand.size[0] * roi_cand.size[1],
                'confidence': roi_cand.confidence,
                'relative_time': relative_time,
                'state': 'open' if roi_cand.is_open else 'closed',
                'bbox': roi_cand.bbox,
            })
        
        # V7.3: Log if we dropped any None ROIs
        if none_roi_count > 0:
            logger.warning(
                f"[ROI] Dropped {none_roi_count} None ROI candidates before classification "
                f"(total_candidates={len(self.roi_candidates)}, valid={len(candidates)})"
            )
        
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
            'frames_decimated': 0,  # V6: Track temporal decimation
        }
        
        # Time scaling for testing mode
        self._time_scale_factor = self.config.testing_time_scale_factor
        self._enable_auto_scaling = self.config.enable_auto_time_scaling
        self._frame_times = []  # Track recent frame processing times for auto-scaling
        self._last_timestamp = None
        self._auto_scale_warmup_frames = self.config.auto_scaling_warmup_frames
        self._auto_scale_target_frame_time = self.config.auto_scaling_target_frame_time_ms
        self._auto_scale_threshold = self.config.auto_scaling_activation_threshold
        self._frame_count = 0
        
        # V6: Temporal decimation state
        self._last_processed_detections: Dict[int, Dict[str, Any]] = {}  # event_id -> last detection info
        self._frames_since_update: Dict[int, int] = {}  # event_id -> frames skipped
        self._last_processed_frame_index: int = 0  # For retention safety
        
        # Apply time scaling to create effective thresholds
        self._update_scaled_thresholds()
        
        # Helper to format base value (handle None for deprecated ms parameters)
        def format_base(ms_val, frame_val, fps):
            if ms_val is not None:
                return f"{ms_val}ms"
            return f"{frame_val} frames @ {fps}fps"
        
        logger.info(
            f"[EventCentricTracker] Initialized with: "
            f"D={self.config.association_distance_px}px, "
            f"T={self._scaled_association_time_ms}ms (base={self.config.association_time_ms}ms), "
            f"G={self._scaled_ghost_timeout_ms}ms (base={format_base(self.config.ghost_timeout_ms, self.config.ghost_timeout_frames, self.config.target_fps)}), "
            f"commit_idle_frames={self.config.commit_idle_frames}, "
            f"suppression_distance={self.config.suppression_distance_px}px, "
            f"suppression_duration={self._scaled_suppression_duration_ms}ms (base={format_base(self.config.suppression_duration_ms, self.config.suppression_duration_frames, self.config.target_fps)}), "
            f"time_scale_factor={self._time_scale_factor}, auto_scaling={self._enable_auto_scaling}"
        )
    
    def _update_scaled_thresholds(self):
        """
        Apply time scaling factor to all millisecond-based parameters.
        
        This creates scaled versions of time thresholds that are used throughout
        the tracker. The scaled values ensure that timing logic behaves equivalently
        in testing mode (slower processing) and production mode (real-time).
        
        For deprecated *_ms parameters that may be None, we derive the ms value
        from frame-based parameters using target_fps.
        """
        scale = self._time_scale_factor
        ms_per_frame = 1000.0 / self.config.target_fps
        
        # Helper to get ms value: use provided ms if not None, else derive from frames
        def get_ms_value(ms_value, frame_value):
            if ms_value is not None:
                return ms_value * scale
            return frame_value * ms_per_frame * scale
        
        # Association and ghost timeouts
        self._scaled_association_time_ms = self.config.association_time_ms * scale
        self._scaled_ghost_timeout_ms = get_ms_value(
            self.config.ghost_timeout_ms, self.config.ghost_timeout_frames
        )
        self._scaled_max_event_lifetime_ms = get_ms_value(
            self.config.max_event_lifetime_ms, self.config.max_event_lifetime_frames
        )
        
        # Suppression and cooldown
        self._scaled_suppression_duration_ms = get_ms_value(
            self.config.suppression_duration_ms, self.config.suppression_duration_frames
        )
        self._scaled_min_event_creation_interval_ms = get_ms_value(
            self.config.min_event_creation_interval_ms, self.config.temporal_cooldown_frames
        )
        
        # State transition timing
        self._scaled_open_to_closing_time_ms = get_ms_value(
            self.config.open_to_closing_time_ms, self.config.open_to_closing_frames
        )
        self._scaled_closing_stability_time_ms = get_ms_value(
            self.config.closing_stability_time_ms, self.config.closing_stability_frames
        )
        self._scaled_closed_stability_time_ms = get_ms_value(
            self.config.closed_stability_time_ms, self.config.closed_stability_frames
        )
        
        # Velocity and prediction (max_prediction_time_ms is always set, not deprecated)
        self._scaled_max_prediction_time_ms = self.config.max_prediction_time_ms * scale
        
        # Logging thresholds (min_gap_duration_for_logging_ms is always set, not deprecated)
        self._scaled_min_gap_duration_for_logging_ms = self.config.min_gap_duration_for_logging_ms * scale
        
        logger.debug(
            f"[EventCentricTracker] Applied time scaling factor {scale:.2f}: "
            f"association_time={self._scaled_association_time_ms:.1f}ms, "
            f"ghost_timeout={self._scaled_ghost_timeout_ms:.1f}ms"
        )
    
    def _get_scaled_config(self) -> EventConfig:
        """
        Create a config copy with scaled time parameters for BreadBagEvent instances.
        
        This allows events to use the scaled thresholds without modifying the
        original config or requiring events to know about scaling.
        
        Returns:
            EventConfig with scaled time parameters
        """
        return replace(
            self.config,
            association_time_ms=self._scaled_association_time_ms,
            ghost_timeout_ms=self._scaled_ghost_timeout_ms,
            max_event_lifetime_ms=self._scaled_max_event_lifetime_ms,
            suppression_duration_ms=self._scaled_suppression_duration_ms,
            min_event_creation_interval_ms=self._scaled_min_event_creation_interval_ms,
            open_to_closing_time_ms=self._scaled_open_to_closing_time_ms,
            closing_stability_time_ms=self._scaled_closing_stability_time_ms,
            closed_stability_time_ms=self._scaled_closed_stability_time_ms,
            max_prediction_time_ms=self._scaled_max_prediction_time_ms,
            min_gap_duration_for_logging_ms=self._scaled_min_gap_duration_for_logging_ms,
        )
    
    def _update_auto_time_scaling(self, current_timestamp_ms: float):
        """
        Auto-calculate time scaling factor based on measured processing speed.
        
        After a warmup period, computes the ratio of actual frame time to target
        frame time and updates the scaling factor accordingly.
        
        Args:
            current_timestamp_ms: Current timestamp in milliseconds
        """
        if not self._enable_auto_scaling:
            return
        
        self._frame_count += 1
        
        # Track frame intervals
        if self._last_timestamp is not None:
            frame_interval = current_timestamp_ms - self._last_timestamp
            self._frame_times.append(frame_interval)
            
            # Keep only recent frames (up to warmup count)
            if len(self._frame_times) > self._auto_scale_warmup_frames:
                self._frame_times.pop(0)
        
        self._last_timestamp = current_timestamp_ms
        
        # After warmup, calculate and apply auto-scaling
        if self._frame_count == self._auto_scale_warmup_frames and len(self._frame_times) > 50:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            
            # Calculate scale factor using configured target frame time
            new_scale_factor = avg_frame_time / self._auto_scale_target_frame_time
            
            # Only apply if significantly different from 1.0 (above activation threshold)
            if new_scale_factor > self._auto_scale_threshold:
                old_factor = self._time_scale_factor
                self._time_scale_factor = new_scale_factor
                self._update_scaled_thresholds()
                
                logger.info(
                    f"[EventCentricTracker] Auto-scaling enabled: "
                    f"avg_frame_time={avg_frame_time:.1f}ms, target={self._auto_scale_target_frame_time:.1f}ms, "
                    f"scale_factor={new_scale_factor:.2f} (was {old_factor:.2f})"
                )
            else:
                logger.info(
                    f"[EventCentricTracker] Auto-scaling measured {new_scale_factor:.2f}x but keeping 1.0x "
                    f"(processing close to real-time, threshold={self._auto_scale_threshold})"
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
        
        # Update auto time scaling if enabled
        self._update_auto_time_scaling(timestamp_ms)
        
        # Clean up old recently_committed entries based on suppression duration (use scaled)
        self.recently_committed = [
            rc for rc in self.recently_committed
            if timestamp_ms - rc['timestamp_ms'] < self._scaled_suppression_duration_ms
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
        
        # ISSUE #1 FIX: Build association candidates for one-to-one matching (greedy assignment)
        # Instead of greedily associating detections as we iterate, we now:
        # 1. Collect all possible (detection, event, score) tuples
        # 2. Sort by score (best first)
        # 3. Assign greedily ensuring one-to-one mapping
        
        # List of (det_idx, event_id, score, distance, iou_value, evidence) tuples
        association_candidates = []
        
        # 1. Build all possible associations
        for det_idx, evidence in enumerate(detection_evidences):
            # ISSUE #2 FIX: Check work zone for associations
            if self.config.work_zone_enabled and self.config.enforce_work_zone_associations:
                if not self._is_in_work_zone(evidence.centroid_x, evidence.centroid_y):
                    # Skip detections outside work zone during association
                    continue
            
            # Track all candidates for this detection
            for event in self.active_events.values():
                if event.state == EventState.COMMITTED:
                    continue
                
                can_assoc, distance, reason, iou_value = event.can_associate(evidence)
                if not can_assoc:
                    continue
                
                # Hybrid scoring: Prioritize IoU over distance when IoU is significant
                # IoU > 0.5: Very likely the same object (even if centroid moved)
                # IoU 0.3-0.5: Possibly the same object, consider distance
                # IoU < 0.3: Rely primarily on distance
                #
                # Score calculation:
                # - IoU weight: High when IoU is significant (0.5+)
                # - Distance weight: High when IoU is low (<0.3)
                # - Normalize distance to 0-1 range (inverse, so closer is better)
                
                # Normalize distance to 0-1 range (where 1 is best/closest)
                # Formula: normalized = max(0, 1 - distance/max_distance)
                # The max(0, ...) clamps negative values when distance > max_distance
                # to ensure normalized_distance stays in valid [0, 1] range
                if self.config.max_association_distance_px > 0:
                    normalized_distance = max(0, 1.0 - (distance / self.config.max_association_distance_px))
                else:
                    # Fallback: If max_distance is 0, use binary close/far logic
                    normalized_distance = 1.0 if distance == 0 else 0.0
                
                # Compute hybrid score with adaptive weighting
                if iou_value >= 0.5:
                    # High IoU: Trust it heavily (80% IoU, 20% distance)
                    score = 0.8 * iou_value + 0.2 * normalized_distance
                elif iou_value >= 0.3:
                    # Moderate IoU: Balance both (60% IoU, 40% distance)
                    score = 0.6 * iou_value + 0.4 * normalized_distance
                else:
                    # Low IoU: Trust distance more (30% IoU, 70% distance)
                    score = 0.3 * iou_value + 0.7 * normalized_distance
                
                # Store candidate for greedy assignment
                association_candidates.append((
                    det_idx, event.id, score, distance, iou_value, evidence
                ))
        
        # 2. Greedy one-to-one assignment: Sort by score and assign best matches first
        # This ensures that the best detection-event pair is always chosen, and each
        # detection/event is matched at most once per frame
        association_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by score descending
        
        assigned_detections = set()
        assigned_events = set()
        
        for det_idx, event_id, score, distance, iou_value, evidence in association_candidates:
            # Skip if already assigned
            if det_idx in assigned_detections or event_id in assigned_events:
                continue
            
            # Assign this detection to this event
            event = self.active_events[event_id]
            event.add_detection(evidence, frame_img)
            assigned_detections.add(det_idx)
            assigned_events.add(event_id)
            associated_detection_indices.add(det_idx)
            
            # Log if noteworthy (low score or multiple candidates competed)
            if score < self.config.low_score_threshold:
                logger.debug(
                    f"[ASSOCIATION_SELECTED] det={det_idx} -> event={event_id}, "
                    f"score={score:.3f}, iou={iou_value:.2f}, dist={distance:.1f}px (low_confidence)"
                )
        
        # 2. Create new events for unassociated open detections
        # First, collect unassociated open detections
        unassociated_open_detections = []
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
            
            unassociated_open_detections.append(evidence)
        
        # Cluster nearby detections to prevent duplicate event creation
        clustered_detections = self._cluster_detections(unassociated_open_detections)
        
        # Now create events from clustered detections
        for evidence in clustered_detections:
            # Check max active events
            if len(self.active_events) >= self.config.max_active_events:
                logger.warning(
                    f"[EventCentricTracker] Max active events reached ({self.config.max_active_events})"
                )
                break
            
            # Check if detection is already covered by an active event
            if self._is_covered_by_active_event(evidence):
                logger.debug(
                    f"[EventCentricTracker] Skipping event creation: detection covered by active event"
                )
                self.stats['events_suppressed'] += 1
                continue
            
            # Check suppression against recently committed events (includes temporal cooldown)
            if self._should_suppress(evidence, timestamp_ms):
                self.stats['events_suppressed'] += 1
                continue
            
            # Create new event with scaled config
            new_event = BreadBagEvent(
                initial_detection=evidence,
                config=self._get_scaled_config(),
                open_class_id=self.open_class_id,
                closed_class_id=self.closed_class_id
            )
            self.active_events[new_event.id] = new_event
            self.stats['events_created'] += 1
        
        # 3. Update ghost state for events without detections
        # CRITICAL: Commit eligibility is evaluated BEFORE expiration
        # - CLOSED events are committed based on idle_frames, not ghost_timeout
        # - Only non-CLOSED events are expired by ghost_timeout
        events_to_remove = []
        
        for event_id, event in self.active_events.items():
            # Check if THIS specific event received a detection this frame
            if event.last_detection_time_ms != timestamp_ms:
                # No detection for this event - update ghost state
                should_commit, status = event.update_ghost_state(timestamp_ms, frame_size, frame_index)
                
                if status == 'commit':
                    # Event is ready for classification
                    ready_events.append(self._prepare_event_output(event))
                    events_to_remove.append(event_id)
                    self.stats['events_committed'] += 1
                    
                    # Add to recently committed (Issue #3: include box for conditional suppression)
                    self.recently_committed.append({
                        'centroid': event.last_centroid,
                        'box': event.last_box,
                        'timestamp_ms': timestamp_ms,
                        'event_id': event_id
                    })
                
                elif status == 'expire':
                    # Event expired (non-CLOSED state exceeded ghost_timeout)
                    events_to_remove.append(event_id)
                    self.stats['events_expired'] += 1
                    
                    # Log expiration with detailed info for debugging
                    time_since = timestamp_ms - event.last_detection_time_ms
                    total_evidence = event.open_evidence_count + event.closed_evidence_count
                    closed_ratio = event.closed_evidence_count / total_evidence if total_evidence > 0 else 0
                    time_in_state = timestamp_ms - event.state_enter_time_ms
                    
                    structured_logger.event_expired(
                        event_id=event_id,
                        state=event.state.name,
                        frames_tracked=event.total_frames_observed,
                        open_hits=event.open_evidence_count,
                        closed_hits=event.closed_evidence_count,
                        frames_since_update=event.frames_without_detection,
                        avg_motion=event.get_centroid_stability()
                    )
                    
                    # Additional detailed logging for audit trail
                    logger.info(
                        f"[Event:{event_id}] EXPIRED: state={event.state.name}, "
                        f"time_since_detection={time_since:.0f}ms, "
                        f"frames_without_detection={event.frames_without_detection}, "
                        f"closed_ratio={closed_ratio:.2f}, "
                        f"time_in_state={time_in_state:.0f}ms, "
                        f"expiration_reason=ghost_timeout_exceeded"
                    )
                
                # status == 'keep_alive' or 'waiting': event stays active
        
        # Remove committed/expired events
        for event_id in events_to_remove:
            del self.active_events[event_id]
        
        # V6: Clean up temporal decimation tracking for removed events
        self._cleanup_decimation_tracking(events_to_remove)
        
        # V6: Update last processed frame index (for retention safety)
        self._last_processed_frame_index = frame_index
        
        return ready_events
    
    def _is_in_work_zone(self, x: float, y: float) -> bool:
        """Check if position is within configured work zone."""
        return (self.config.work_zone_x1 <= x <= self.config.work_zone_x2 and
                self.config.work_zone_y1 <= y <= self.config.work_zone_y2)
    
    def _should_suppress(self, evidence: DetectionEvidence, timestamp_ms: float) -> bool:
        """
        Check if new event should be suppressed.
        
        Anti-Double-Counting:
        Prevents new events from being created for a bag that was temporarily
        lost then re-detected after commitment. This ensures each physical bag
        is counted exactly once.
        
        ISSUE #3 FIX: Conditional suppression using box overlap
        - If suppression_require_box_overlap is True, suppression requires both:
          1. Centroid proximity (within suppression_distance_px)
          2. Box overlap with last committed box (IoU >= suppression_iou_threshold)
        - This allows new bags to start immediately at the same location if there's
          no box overlap (worker starts new bag after removing the last one)
        
        TEMPORAL COOLDOWN: Hard temporal cooldown zone
        - No new events allowed within temporal_cooldown_distance_px for
          min_event_creation_interval_ms after commit
        - This catches detection flickering and rapid re-detections
        
        Args:
            evidence: Detection evidence for potential new event
            timestamp_ms: Current timestamp in milliseconds
            
        Returns:
            True if event should be suppressed, False otherwise
        """
        for rc in self.recently_committed:
            dx = evidence.centroid_x - rc['centroid'][0]
            dy = evidence.centroid_y - rc['centroid'][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate time since this event was committed
            time_since_commit = timestamp_ms - rc['timestamp_ms']
            
            # TEMPORAL COOLDOWN CHECK: Hard temporal cooldown - no new events in same area for X ms
            # This is the most aggressive check and happens first
            if time_since_commit < self._scaled_min_event_creation_interval_ms:
                if distance < self.config.temporal_cooldown_distance_px:
                    # Within cooldown period and distance - suppress
                    logger.debug(
                        f"[EventCentricTracker] Suppressing new event: "
                        f"within temporal cooldown zone of recently committed {rc['event_id']} "
                        f"(time_since_commit={time_since_commit:.0f}ms < {self._scaled_min_event_creation_interval_ms}ms, "
                        f"distance={distance:.1f}px < {self.config.temporal_cooldown_distance_px}px)"
                    )
                    return True
            
            # Check centroid distance for standard suppression
            if distance >= self.config.suppression_distance_px:
                # Too far away, no suppression
                continue
            
            # ISSUE #3 FIX: If box overlap is required, check IoU with last committed box
            if self.config.suppression_require_box_overlap:
                if 'box' in rc:
                    # Compute IoU between new detection and last committed box
                    iou = self._compute_iou_static(rc['box'], evidence.box)
                    
                    if iou < self.config.suppression_iou_threshold:
                        # No significant box overlap - allow new event
                        # This handles the case where worker starts a new bag at the same location
                        logger.debug(
                            f"[EventCentricTracker] Allowing new event despite proximity: "
                            f"no box overlap with recently committed {rc['event_id']} "
                            f"(distance={distance:.1f}px, iou={iou:.3f} < threshold={self.config.suppression_iou_threshold})"
                        )
                        continue
            
            # Suppress: close proximity and (if required) box overlap detected
            logger.debug(
                f"[EventCentricTracker] Suppressing new event: "
                f"too close to recently committed {rc['event_id']} "
                f"(distance={distance:.1f}px < suppression_threshold={self.config.suppression_distance_px}px)"
            )
            return True
        
        return False
    
    def _compute_iou_static(self, box1: Tuple[float, float, float, float], 
                           box2: Tuple[float, float, float, float]) -> float:
        """
        Static method to compute IoU between two boxes.
        Used for suppression logic.
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
    
    def _is_covered_by_active_event(self, evidence: DetectionEvidence) -> bool:
        """
        Check if a detection is already covered by an active event.
        
        This prevents creating new events when another active event already
        covers the detection area. Helps prevent duplicates from:
        - Detection flickering/splitting
        - Temporarily lost detections that immediately return
        - Multiple detections of same bag in one frame
        
        Performance: O(n) where n is number of active events (typically 1-5, max 15).
        Linear search is efficient for this scale and avoids complexity of spatial indexing.
        
        Args:
            evidence: Detection evidence to check
            
        Returns:
            True if detection is covered by an active event, False otherwise
        """
        for event in self.active_events.values():
            # Skip committed events (they're handled by suppression)
            if event.state == EventState.COMMITTED:
                continue
            
            # Check centroid proximity
            dx = evidence.centroid_x - event.last_centroid[0]
            dy = evidence.centroid_y - event.last_centroid[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.config.active_event_exclusion_distance_px:
                logger.debug(
                    f"[EventCentricTracker] Detection covered by active event {event.id}: "
                    f"centroid distance={distance:.1f}px < threshold={self.config.active_event_exclusion_distance_px}px"
                )
                return True
            
            # Also check IoU with active event's box
            if event.last_box is not None:
                iou = self._compute_iou_static(event.last_box, evidence.box)
                if iou > self.config.active_event_exclusion_iou:
                    logger.debug(
                        f"[EventCentricTracker] Detection covered by active event {event.id}: "
                        f"IoU={iou:.3f} > threshold={self.config.active_event_exclusion_iou}"
                    )
                    return True
        
        return False
    
    def _cluster_detections(self, evidences: List[DetectionEvidence]) -> List[DetectionEvidence]:
        """
        Cluster nearby detections and return representative detection per cluster.
        
        Before creating new events, nearby unassociated detections are clustered
        together. Only the highest confidence detection from each cluster is used
        to create an event. This prevents duplicate events from:
        - Detection flickering (same bag detected multiple times)
        - Detection splitting (one bag split into multiple boxes)
        - Noisy detections around the same physical bag
        
        Performance: O(n²) complexity, but acceptable for this use case since:
        - Typically 0-5 unassociated detections per frame
        - Most detections associate with existing events
        - Only runs on the small subset of unassociated detections
        
        Args:
            evidences: List of detection evidences to cluster
            
        Returns:
            List of representative detections (one per cluster)
        """
        if len(evidences) <= 1:
            return evidences
        
        clustered = []
        used = set()
        
        for i, ev1 in enumerate(evidences):
            if i in used:
                continue
            
            # Start a new cluster with this detection
            cluster = [ev1]
            used.add(i)
            
            # Find all nearby detections to add to this cluster
            for j, ev2 in enumerate(evidences):
                if j in used:
                    continue
                
                # Calculate distance between detections
                dx = ev1.centroid_x - ev2.centroid_x
                dy = ev1.centroid_y - ev2.centroid_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < self.config.detection_cluster_distance_px:
                    cluster.append(ev2)
                    used.add(j)
            
            # Pick highest confidence detection as representative of this cluster
            representative = max(cluster, key=lambda e: e.confidence)
            clustered.append(representative)
            
            # Log if we clustered multiple detections
            if len(cluster) > 1:
                logger.debug(
                    f"[EventCentricTracker] Clustered {len(cluster)} nearby detections "
                    f"at ({representative.centroid_x:.0f}, {representative.centroid_y:.0f}), "
                    f"using highest confidence detection (conf={representative.confidence:.2f})"
                )
        
        return clustered
    
    def _prepare_event_output(self, event: BreadBagEvent) -> Dict[str, Any]:
        """
        Prepare committed event data for classification.
        
        Returns data in format compatible with ClassifierService.
        """
        candidates = event.get_roi_candidates()
        debug_info = event.get_debug_info()

        event_stats = {
            'total': len(candidates),
            'open_count': event.open_roi_count,
            'closed_count': event.closed_roi_count,
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
            'last_processed_frame_index': self._last_processed_frame_index,  # V6: For retention safety
        }
    
    def _should_skip_temporal_decimation(
        self, 
        event_id: int, 
        evidence: DetectionEvidence
    ) -> bool:
        """
        V6: Check if this detection update can be skipped (temporal decimation).
        
        Skip monitor update when:
        - Bounding box area change < epsilon
        - Centroid shift < delta
        - Confidence unchanged
        
        This reduces CPU cost significantly while preserving correctness.
        Detection still runs every frame; only redundant state updates are skipped.
        
        Args:
            event_id: ID of the event being updated
            evidence: New detection evidence
            
        Returns:
            True if update can be skipped, False if update is required
        """
        if not self.config.temporal_decimation_enabled:
            return False
        
        # Always process if this is a new event
        if event_id not in self._last_processed_detections:
            self._last_processed_detections[event_id] = {
                'centroid': (evidence.centroid_x, evidence.centroid_y),
                'area': (evidence.box[2] - evidence.box[0]) * (evidence.box[3] - evidence.box[1]),
                'confidence': evidence.confidence,
                'frame_index': evidence.frame_index,
            }
            self._frames_since_update[event_id] = 0
            return False
        
        last = self._last_processed_detections[event_id]
        frames_skipped = self._frames_since_update.get(event_id, 0)
        
        # Force update if max skip frames exceeded
        if frames_skipped >= self.config.temporal_decimation_max_skip_frames:
            # Update tracking and process
            self._last_processed_detections[event_id] = {
                'centroid': (evidence.centroid_x, evidence.centroid_y),
                'area': (evidence.box[2] - evidence.box[0]) * (evidence.box[3] - evidence.box[1]),
                'confidence': evidence.confidence,
                'frame_index': evidence.frame_index,
            }
            self._frames_since_update[event_id] = 0
            return False
        
        # Check centroid shift
        dx = evidence.centroid_x - last['centroid'][0]
        dy = evidence.centroid_y - last['centroid'][1]
        centroid_shift = math.sqrt(dx*dx + dy*dy)
        
        if centroid_shift >= self.config.temporal_decimation_centroid_delta_px:
            # Significant movement - process
            self._last_processed_detections[event_id] = {
                'centroid': (evidence.centroid_x, evidence.centroid_y),
                'area': (evidence.box[2] - evidence.box[0]) * (evidence.box[3] - evidence.box[1]),
                'confidence': evidence.confidence,
                'frame_index': evidence.frame_index,
            }
            self._frames_since_update[event_id] = 0
            return False
        
        # Check area change
        new_area = (evidence.box[2] - evidence.box[0]) * (evidence.box[3] - evidence.box[1])
        if last['area'] > 0:
            area_change = abs(new_area - last['area']) / last['area']
            if area_change >= self.config.temporal_decimation_area_epsilon:
                # Significant area change - process
                self._last_processed_detections[event_id] = {
                    'centroid': (evidence.centroid_x, evidence.centroid_y),
                    'area': new_area,
                    'confidence': evidence.confidence,
                    'frame_index': evidence.frame_index,
                }
                self._frames_since_update[event_id] = 0
                return False
        
        # Check confidence change
        conf_change = abs(evidence.confidence - last['confidence'])
        if conf_change >= self.config.temporal_decimation_confidence_epsilon:
            # Significant confidence change - process
            self._last_processed_detections[event_id] = {
                'centroid': (evidence.centroid_x, evidence.centroid_y),
                'area': new_area,
                'confidence': evidence.confidence,
                'frame_index': evidence.frame_index,
            }
            self._frames_since_update[event_id] = 0
            return False
        
        # All checks passed - skip this update
        self._frames_since_update[event_id] = frames_skipped + 1
        self.stats['frames_decimated'] += 1
        return True
    
    def _cleanup_decimation_tracking(self, removed_event_ids: List[int]):
        """Clean up temporal decimation tracking for removed events."""
        for event_id in removed_event_ids:
            self._last_processed_detections.pop(event_id, None)
            self._frames_since_update.pop(event_id, None)
    
    def get_last_processed_frame_index(self) -> int:
        """
        V6: Get the last processed frame index for retention safety.
        
        Returns:
            Frame index of the last fully processed frame
        """
        return self._last_processed_frame_index
