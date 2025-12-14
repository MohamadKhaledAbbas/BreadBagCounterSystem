"""
Centralized configuration for detection and tracking parameters.

This module contains all tunable parameters for the bag detection and tracking system.
Adjust these values to tune the system's sensitivity and behavior.

V3 Performance Optimization Notes:
- min_roi_size reduced to 100 (from 300) to avoid blocking the pipeline
- min_roi_sharpness reduced to 300 (from 400) to accept more samples
- Parameters tuned for 25fps throughput at 720p resolution

V5 Event-Centric Tracking Notes:
- Added event-centric tracking parameters (D, T, G, exit_timeout)
- Centroid-based association replaces IoU for rotation tolerance
- Millisecond-based timing replaces frame counts
- Exit-boundary-based counting rule
"""

from dataclasses import dataclass
import os


@dataclass
class TrackingConfig:
    """
    Configuration for bag detection and tracking system.
    
    These parameters control how the system detects, tracks, and classifies bread bags.
    Adjust these values to tune sensitivity and accuracy.
    
    V3 Optimizations:
    - Reduced expiry timeouts for faster event cleanup
    - Relaxed ROI quality gates to avoid pipeline blocking
    - Adjusted sample counts for memory efficiency
    """
    
    # ============================================================================
    # IoU Matching Parameters
    # ============================================================================
    
    iou_threshold: float = 0.45
    """
    IoU (Intersection over Union) threshold for matching detections to existing events.
    
    Range: 0.0 - 1.0
    - Lower values (e.g., 0.3): More lenient matching, good for fast-moving bags
    - Higher values (e.g., 0.5): Stricter matching, reduces false associations
    
    Default: 0.45
    """

    # ============================================================================
    # Number of frames to suppress new events
    # ============================================================================
    lockout_window: int = 50  # V3: Reduced from 25 for faster event recovery


    # ============================================================================
    # State Transition Thresholds
    # ============================================================================
    
    min_open_frames: int = 5  # V3: Reduced from 5 for faster state transitions
    """
    Minimum consecutive frames a bag must be detected as "open" before allowing 
    transition to "closed" state.
    
    Range: 1 - 20
    - Lower values: Faster state transitions, more responsive
    - Higher values: More stable, reduces noise-induced transitions
    
    Default: 4 (V3: reduced from 5)
    """
    
    min_closed_frames: int = 3  # V3: Reduced from 3 for faster counting
    """
    Minimum consecutive frames a bag must be detected as "closed" to trigger 
    classification and counting.
    
    Range: 1 - 10
    - Lower values: Faster counting, may catch partial closures
    - Higher values: More reliable closed detection, may miss quick closures
    
    Default: 2 (V3: reduced from 3)
    """
    
    # ============================================================================
    # Detection Confidence Thresholds
    # ============================================================================
    
    min_conf_threshold: float = 0.4
    """
    Minimum confidence score for creating new tracking events from detections.
    
    Range: 0.0 - 1.0
    - Lower values (e.g., 0.1): Catch more potential bags, more false positives
    - Higher values (e.g., 0.4): Only high-confidence detections, may miss some bags
    
    Default: 0.4
    """
    
    # ============================================================================
    # Event Management
    # ============================================================================
    
    max_active_events: int = 15  # V3: Increased from 10 for better tracking
    """
    Maximum number of concurrent tracking events to prevent memory issues.
    
    Range: 10 - 100
    If this limit is reached, new detections will be ignored until events expire.
    
    Default: 15 (V3: increased from 10)
    """
    
    # ============================================================================
    # State-Aware Expiry Timeouts
    # ============================================================================
    
    expiry_detecting_open: int = 20   # V3: Reduced from 10 for faster cleanup
    """
    Frames without update before expiring an event in 'detecting_open' state.
    
    Range: 5 - 30
    - Lower values: Faster cleanup of lost tracks
    - Higher values: More persistent tracking through occlusions
    
    Default: 8 (V3: reduced from 10)
    """
    
    expiry_detecting_closed: int = 20  # V3: Reduced from 10 for faster cleanup
    """
    Frames without update before expiring an event in 'detecting_closed' state.
    
    Range: 10 - 30
    - Lower values: Faster cleanup
    - Higher values: Give more time for closed detection to stabilize
    
    Default: 8 (V3: reduced from 10)
    """
    
    expiry_counted: int = 3  # V3: Reduced from 5 for faster cleanup
    """
    Frames without update before expiring an event in 'counted' state.
    
    Range: 3 - 15
    Events in counted state are ready for cleanup and should expire quickly.
    
    Default: 3 (V3: reduced from 5)
    """
    
    # ============================================================================
    # ROI Collection Parameters (BagEvent)
    # ============================================================================
    
    max_open_samples: int = 5  # V3: Reduced from 6 for memory efficiency
    """
    Maximum number of ROI samples to collect during the 'open' phase.
    
    Range: 4 - 15
    More samples provide better classification but use more memory.
    
    Default: 5 (V3: reduced from 6)
    """
    
    max_closed_samples: int = 3  # V3: Reduced from 4 for memory efficiency
    """
    Maximum number of ROI samples to collect during the 'closed' phase.
    
    Range: 2 - 10
    More samples provide better classification but use more memory.
    
    Default: 3 (V3: reduced from 4)
    """
    
    # ============================================================================
    # ROI Quality Validation
    # ============================================================================
    
    min_roi_size: int = 100  # V3: CRITICAL FIX - was 300 which blocked the pipeline
    """
    Minimum width/height (in pixels) for a valid ROI.
    
    Range: 50 - 200
    ROIs smaller than this are rejected as too small for reliable classification.
    
    Default: 100 (V3: reduced from 300 - this was blocking the entire pipeline!)
    
    IMPORTANT: The log showed ROIs of ~160x175 pixels were being rejected due to
    min_size=300. This was preventing classification from ever running.
    """
    
    min_roi_sharpness: float = 500  # V3: Reduced from 400 for more accepted samples
    """
    Minimum sharpness score (Laplacian variance) for a valid ROI.
    
    Range: 10 - 500
    - Lower values: Accept more blurry images, more samples
    - Higher values: Only sharp images, fewer samples but better quality
    
    Default: 300 (V3: reduced from 400)
    """

    min_mean_brightness: int = 80  # V3: Reduced from 100 for darker environments
    """
    Minimum mean brightness for a valid ROI.
    
    Default: 80 (V3: reduced from 100)
    """

    max_mean_brightness: int = 220  # V3: Increased from 200 for brighter environments
    """
    Maximum mean brightness for a valid ROI.
    
    Default: 220 (V3: increased from 200)
    """


    # ============================================================================
    # Classification Parameters (V4: Evidence-Based Classification)
    # ============================================================================
    
    top_k_candidates: int = 5
    """
    Number of top ROI candidates to select for classification at track end.
    
    Range: 3 - 7
    These are selected by sharpness (primary) and frame recency (secondary).
    
    Default: 5
    """
    
    min_total_evidence_score: float = 0.3
    """
    Minimum total evidence score required to accept a classification.
    
    Range: 0.1 - 1.0
    Below this threshold, the track is classified as "Unknown".
    
    Default: 0.3
    """
    
    evidence_ratio_threshold: float = 1.5
    """
    Minimum ratio of winner score to runner-up score for acceptance.
    
    Range: 1.1 - 3.0
    Higher values require more confident differentiation between top classes.
    
    Default: 1.5
    """
    
    min_candidates_for_classification: int = 2
    """
    Minimum number of valid ROI candidates required for classification.
    
    Range: 1 - 5
    Tracks with fewer candidates are classified as "Unknown" (insufficient data).
    
    Default: 2
    """
    
    min_track_frames: int = 3
    """
    Minimum number of frames a track must exist before classification.
    
    Range: 2 - 10
    Very short tracks are considered unreliable and classified as "Unknown".
    
    Default: 3
    """
    
    sharpness_weight_scale: float = 100.0
    """
    Scaling factor for sharpness-based weighting.
    
    Range: 50.0 - 500.0
    Higher values give more weight to sharper frames.
    
    Default: 100.0
    """
    
    temporal_weight_scale: float = 0.5
    """
    Weight given to later frames in the track (temporal recency).
    
    Range: 0.0 - 1.0
    Higher values favor frames captured later in the track lifecycle.
    
    Default: 0.5
    """
    
    max_single_roi_weight: float = 0.6
    """
    Maximum weight any single ROI can contribute to the final evidence.
    
    Range: 0.3 - 1.0
    Prevents one very confident ROI from overwhelming all other evidence.
    
    Default: 0.6
    """

    # ============================================================================
    # V5: Event-Centric Tracking Parameters
    # ============================================================================
    # These parameters control the new event-centric tracking system that replaces
    # IoU-based tracking for better handling of rotation, occlusion, and deformation.
    
    use_event_centric_tracking: bool = True
    """
    Enable event-centric tracking instead of IoU-based tracking.
    
    When True: Uses centroid distance + time for association
    When False: Uses legacy IoU-based tracking (deprecated)
    
    Default: True (V5)
    """
    
    # --------------------------------------------------------------------------
    # Association Parameters (D, T from requirements)
    # --------------------------------------------------------------------------
    
    association_distance_px: float = 100.0
    """
    D: Maximum centroid distance (pixels) to associate a detection with an event.
    
    Range: 50 - 200
    - Lower values: Stricter association, may lose track during fast movement
    - Higher values: More lenient, may incorrectly merge nearby bags
    
    Tuning: Based on expected bag movement per frame at your FPS.
    At 25fps with typical human manipulation speed, 100px works well.
    
    Default: 100.0
    """
    
    association_time_ms: float = 400.0
    """
    T: Maximum time gap (milliseconds) to associate detection with an event.
    
    Range: 200 - 800
    - Lower values: Faster rejection of lost tracks
    - Higher values: More tolerant of brief detection gaps
    
    Tuning: Should be longer than typical detection flicker but shorter than
    the time between handling two different bags.
    
    Default: 400.0
    """
    
    # --------------------------------------------------------------------------
    # Ghost Event Parameters (G from requirements)
    # --------------------------------------------------------------------------
    
    ghost_timeout_ms: float = 1000.0
    """
    G: Time (milliseconds) to keep event alive without detections.
    
    Range: 500 - 2000
    - Lower values: Faster cleanup of lost events
    - Higher values: Survives longer occlusions (hand over bag)
    
    Tuning: Should cover typical hand occlusion duration during tying.
    1000ms (1 second) handles most manipulation scenarios.
    
    Default: 1000.0
    """
    
    # --------------------------------------------------------------------------
    # Exit and Counting Parameters (CRITICAL for counting rule)
    # --------------------------------------------------------------------------
    
    exit_timeout_ms: float = 800.0
    """
    Time (milliseconds) after CLOSED state before COMMIT (counting).
    
    Range: 500 - 1500
    - Lower values: Faster counting, risk of premature commit
    - Higher values: More certain the bag has left, but slower
    
    Tuning: Should ensure bag has actually left the scene.
    
    Default: 800.0
    """
    
    exit_boundary_margin_px: int = 50
    """
    Margin from frame edge to consider "near exit boundary".
    
    Range: 30 - 100
    Centroids within this distance from frame edge trigger exit detection.
    
    Default: 50
    """
    
    # --------------------------------------------------------------------------
    # State Transition Temporal Stability
    # --------------------------------------------------------------------------
    
    open_to_closing_time_ms: float = 100.0
    """
    Minimum time (milliseconds) in OPEN state before transitioning to CLOSING.
    
    Range: 50 - 300
    Prevents noise from immediately triggering state changes.
    
    Default: 100.0
    """
    
    closing_stability_time_ms: float = 150.0
    """
    Time (milliseconds) closed detections must persist for CLOSED state.
    
    Range: 100 - 400
    Ensures bag is actually closing, not just a detection artifact.
    
    Default: 150.0
    """
    
    closed_stability_time_ms: float = 200.0
    """
    Minimum time (milliseconds) in CLOSED state before COMMIT is eligible.
    
    Range: 100 - 500
    Gives time to collect ROIs and stabilize classification.
    
    Default: 200.0
    """
    
    centroid_stability_px: float = 30.0
    """
    Maximum centroid movement (pixels) to consider position "stable".
    
    Range: 10 - 50
    Used to validate geometric stability during state transitions.
    
    Default: 30.0
    """
    
    # --------------------------------------------------------------------------
    # Evidence Thresholds (decoupled from YOLO output)
    # --------------------------------------------------------------------------
    
    min_open_evidence_count: int = 3
    """
    Minimum open detections before state can transition to CLOSING.
    
    Range: 2 - 10
    Ensures bag was actually seen as open before counting.
    
    Default: 3
    """
    
    min_closed_evidence_count: int = 2
    """
    Minimum closed detections required for CLOSED state.
    
    Range: 1 - 5
    Prevents single noisy detection from triggering state change.
    
    Default: 2
    """
    
    # --------------------------------------------------------------------------
    # Work Zone Configuration
    # --------------------------------------------------------------------------
    
    work_zone_enabled: bool = False
    """
    Enable work zone filtering for event creation.
    
    When True: Only create events for detections inside work zone
    When False: Entire frame is valid for event creation
    
    Default: False (use entire frame)
    """
    
    work_zone_x1: int = 0
    work_zone_y1: int = 0
    work_zone_x2: int = 1280
    work_zone_y2: int = 720
    """
    Work zone boundaries (pixels): [x1, y1] to [x2, y2]
    
    Only used when work_zone_enabled is True.
    """
    
    # --------------------------------------------------------------------------
    # Event-Centric ROI Collection
    # --------------------------------------------------------------------------
    
    event_max_roi_samples: int = 8
    """
    Maximum ROIs to collect during CLOSED state for classification.
    
    Range: 5 - 15
    More samples improve classification but use more memory.
    
    Default: 8
    """
    
    # --------------------------------------------------------------------------
    # Classification Voting (V5 Temporal Voting)
    # --------------------------------------------------------------------------
    
    voting_agreement_threshold_pct: float = 60.0
    """
    Minimum percentage of votes that must agree for accepted classification.
    
    Range: 50 - 80
    - Lower: More permissive, may accept ambiguous results
    - Higher: Stricter, more UNKNOWN results
    
    Default: 60.0
    """
    
    confidence_margin_threshold: float = 0.15
    """
    Minimum confidence margin between top two classes.
    
    Range: 0.1 - 0.3
    If margin is smaller, classification is considered ambiguous -> UNKNOWN.
    
    Default: 0.15
    """


# Global configuration instance
tracking_config = TrackingConfig()


def get_event_config():
    """
    Create EventConfig from TrackingConfig for EventCentricTracker.
    
    This bridges the existing configuration system with the new event-centric tracker.
    """
    from src.tracking.EventCentricTracker import EventConfig
    
    return EventConfig(
        # Work zone
        work_zone_enabled=tracking_config.work_zone_enabled,
        work_zone_x1=tracking_config.work_zone_x1,
        work_zone_y1=tracking_config.work_zone_y1,
        work_zone_x2=tracking_config.work_zone_x2,
        work_zone_y2=tracking_config.work_zone_y2,
        
        # Association (D, T)
        association_distance_px=tracking_config.association_distance_px,
        association_time_ms=tracking_config.association_time_ms,
        
        # Ghost (G)
        ghost_timeout_ms=tracking_config.ghost_timeout_ms,
        
        # Exit and counting
        exit_timeout_ms=tracking_config.exit_timeout_ms,
        exit_boundary_margin_px=tracking_config.exit_boundary_margin_px,
        
        # State transition timing
        open_to_closing_time_ms=tracking_config.open_to_closing_time_ms,
        closing_stability_time_ms=tracking_config.closing_stability_time_ms,
        closed_stability_time_ms=tracking_config.closed_stability_time_ms,
        centroid_stability_px=tracking_config.centroid_stability_px,
        
        # Evidence thresholds
        min_open_evidence_count=tracking_config.min_open_evidence_count,
        min_closed_evidence_count=tracking_config.min_closed_evidence_count,
        min_detection_confidence=tracking_config.min_conf_threshold,
        
        # ROI collection
        max_roi_samples=tracking_config.event_max_roi_samples,
        min_roi_size=tracking_config.min_roi_size,
        min_roi_sharpness=tracking_config.min_roi_sharpness,
        min_brightness=tracking_config.min_mean_brightness,
        max_brightness=tracking_config.max_mean_brightness,
        
        # Classification voting
        min_voting_agreement_pct=tracking_config.voting_agreement_threshold_pct,
        confidence_margin_threshold=tracking_config.confidence_margin_threshold,
        
        # Resource limits
        max_active_events=tracking_config.max_active_events,
    )
