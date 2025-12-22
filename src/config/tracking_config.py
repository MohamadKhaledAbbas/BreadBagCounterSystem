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
from typing import Optional
import os

from src.utils.platform import IS_WINDOWS


def _parse_bool_env(env_var: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    value = os.getenv(env_var)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def _parse_float_env(env_var: str, default: float) -> float:
    """Parse float from environment variable."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_int_env(env_var: str, default: int) -> int:
    """Parse int from environment variable."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


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
    
    max_open_samples: int = 10  # V3: Reduced from 6 for memory efficiency
    """
    Maximum number of ROI samples to collect during the 'open' phase.
    
    Range: 4 - 15
    More samples provide better classification but use more memory.
    
    Default: 5 (V3: reduced from 6)
    """
    
    max_closed_samples: int = 10  # V3: Reduced from 4 for memory efficiency
    """
    Maximum number of ROI samples to collect during the 'closed' phase.
    
    Range: 2 - 10
    More samples provide better classification but use more memory.
    
    Default: 3 (V3: reduced from 4)
    """
    
    # ============================================================================
    # ROI Quality Validation
    # ============================================================================
    
    min_roi_size: int = 70  # V3: CRITICAL FIX - was 300 which blocked the pipeline
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

    min_mean_brightness: int = 60  # V3: Reduced from 100 for darker environments
    """
    Minimum mean brightness for a valid ROI.
    
    Default: 80 (V3: reduced from 100)
    """

    max_mean_brightness: int = 240  # V3: Increased from 200 for brighter environments
    """
    Maximum mean brightness for a valid ROI.
    
    Default: 220 (V3: increased from 200)
    """

    save_all_rois: bool = True

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
    
    high_confidence_threshold: float = 0.5
    """
    Confidence threshold for "high" vs "low" confidence tier.
    
    Range: 0.3 - 0.7
    Confidence >= this value is considered "high confidence"
    Confidence < this value is considered "low confidence"
    
    This enables analytics to show separate counts for high vs low confidence
    detections, providing better visibility into classification quality.
    
    Default: 0.5
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
    # Classification Stability Heuristics (Production-Grade)
    # ============================================================================
    
    enable_label_reuse: bool = _parse_bool_env("ENABLE_LABEL_REUSE", True)
    """
    Enable previous-label reuse when confidence is low but evidence is strong.
    
    When True: Allows reusing previous track classification if current confidence
               is below LOW_CONF_THRESHOLD but there's strong historical evidence
               (streak length >= STREAK_MIN, burst dominance, etc.)
    When False: Always use current classification (default safe behavior)
    
    Feature-flagged for safe rollout. Can also be controlled via environment:
        ENABLE_LABEL_REUSE=true
    
    Default: False (disabled for safety, opt-in only)
    """
    
    low_conf_threshold: float = _parse_float_env("LOW_CONF_THRESHOLD", 0.7)
    """
    Confidence threshold below which previous-label reuse is considered.
    
    Range: 0.5 - 0.8
    
    Rationale: Classifications with confidence < 0.65 are considered "uncertain"
    and may benefit from historical context. This threshold is set below the
    high_confidence_threshold (0.7) but high enough to avoid reusing labels for
    very low-confidence predictions that should be marked as Unknown.
    
    Tuning: 
    - Lower values (0.5-0.6): More aggressive reuse, may mask genuine label changes
    - Higher values (0.7-0.8): Conservative reuse, only for borderline cases
    
    Environment: LOW_CONF_THRESHOLD=0.65
    
    Default: 0.65 (between low and high confidence tiers)
    """
    
    streak_min_length: int = _parse_int_env("STREAK_MIN_LENGTH", 3)
    """
    Minimum streak length required to allow previous-label reuse.
    
    Range: 2 - 10
    
    Rationale: A streak indicates consistent classification over multiple bags,
    suggesting the previous label is reliable. Minimum of 3 ensures we're not
    reusing labels from single-bag noise or isolated classifications.
    
    Tuning:
    - Lower values (2-3): More responsive to short-term patterns
    - Higher values (5-10): Only trust very stable long-term patterns
    
    Default: 3 (requires at least 3 consecutive bags of same type)
    """
    
    burst_dominance_min_ratio: float = _parse_float_env("BURST_DOMINANCE_MIN_RATIO", 0.75)
    """
    Minimum ratio of dominant label in recent burst for reuse validation.
    
    Range: 0.6 - 0.9
    
    Rationale: If analyzing a recent time window (e.g., last minute), the dominant
    label must represent at least 75% of classifications to be considered a valid
    "burst pattern". This guards against reusing labels in mixed-variety scenarios.
    
    Tuning:
    - Lower values (0.6-0.7): Allow reuse in more diverse scenarios
    - Higher values (0.8-0.9): Only allow reuse in very homogeneous bursts
    
    This is checked along with streak length to ensure both sequential and
    temporal consistency before reusing a previous label.
    
    Environment: BURST_DOMINANCE_MIN_RATIO=0.75
    
    Default: 0.75 (75% majority required)
    """
    
    burst_window_size: int = _parse_int_env("BURST_WINDOW_SIZE", 10)
    """
    Number of recent classifications to analyze for burst dominance.
    
    Range: 5 - 20
    
    Defines the sliding window for burst dominance calculation. A larger window
    provides more stable burst detection but is less responsive to variety changes.
    
    Default: 10 (last 10 classifications)
    """
    
    track_volatility_threshold: float = _parse_float_env("TRACK_VOLATILITY_THRESHOLD", 0.3)
    """
    Label change rate above which a track is flagged as high-volatility.
    
    Range: 0.2 - 0.5
    
    Volatility = (number of label changes) / (track lifespan in bags)
    
    Rationale: A volatility score > 0.3 means the label changed more than once
    every 3 bags, indicating classification instability. Such tracks should be
    flagged for review as they may indicate:
    - Poor model quality
    - Ambiguous bag types
    - Incorrect label reuse
    
    Tuning:
    - Lower values (0.2): Flag more tracks, stricter stability requirement
    - Higher values (0.4-0.5): Only flag very unstable tracks
    
    Environment: TRACK_VOLATILITY_THRESHOLD=0.3
    
    Default: 0.3 (one change per 3 bags)
    """
    
    enable_volatility_logging: bool = _parse_bool_env("ENABLE_VOLATILITY_LOGGING", True)
    """
    Enable structured logging for high-volatility tracks.
    
    When True: Emits structured logs for tracks exceeding volatility threshold
    When False: Only calculates volatility, no logging
    
    Default: True (enable for production monitoring)
    """

    # ============================================================================
    # Degraded Mode Parameters (Overload Handling)
    # ============================================================================
    
    degraded_mode_enabled: bool = True
    """
    Enable degraded mode to handle overload gracefully.
    
    When True: Automatically reduce non-critical work under overload
    When False: Continue normal operation regardless of load
    
    Default: True
    """
    
    degraded_mode_queue_threshold: float = 0.7
    """
    Queue utilization threshold to trigger degraded mode.
    
    Range: 0.5 - 0.9
    When input queue utilization exceeds this value, degraded mode activates.
    
    Default: 0.7 (70% full)
    """
    
    degraded_mode_delay_threshold_ms: float = 100.0
    """
    Average queue delay (milliseconds) to trigger degraded mode.
    
    Range: 50 - 300
    When average time frames spend in queue exceeds this, degraded mode activates.
    
    Default: 100.0 ms
    """
    
    degraded_mode_disable_roi_saving: bool = True
    """
    Disable ROI image saving in degraded mode.
    
    Default: True (saves disk I/O)
    """
    
    degraded_mode_disable_visualization: bool = False
    """
    Disable visualization work in degraded mode.
    
    Note: Set to False by default to maintain UI visibility.
    Set to True if visualization is not critical and you need maximum throughput.
    
    Default: False (keep visualization enabled)
    """
    
    degraded_mode_skip_low_detection_frames: bool = True
    """
    Skip processing frames with no detections in degraded mode.
    
    This is safe because frames without detections don't contribute to tracking.
    
    Default: True
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
    
    association_distance_px: float = 40.0
    """
    D: Maximum centroid distance (pixels) to associate a detection with an event.
    
    Range: 50 - 200
    - Lower values: Stricter association, may lose track during fast movement
    - Higher values: More lenient, may incorrectly merge nearby bags
    
    Tuning: Based on expected bag movement per frame at your FPS.
    At 25fps with typical human manipulation speed, 80px reduces teleportation
    when multiple bags are on the table.
    
    Default: 80.0 (reduced from 100.0 to prevent teleportation in crowded scenes)
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
    # IoU-Based Association (parallel hybrid with centroid distance)
    # --------------------------------------------------------------------------
    # PARALLEL HYBRID ASSOCIATION: Both centroid distance AND IoU are ALWAYS
    # computed for every association attempt. A detection associates if EITHER
    # criterion is met. This provides robustness during:
    # - Bag flips/spins: centroid may jump but IoU remains high
    # - Fast slides: IoU may drop but centroid distance stays close
    # - Partial occlusions: one metric may fail while the other succeeds
    
    iou_association_enabled: bool = True
    """
    Enable IoU as a parallel association criterion.
    
    When True: Detection can associate if IoU is high enough, even if centroid
               distance exceeds threshold. Both metrics are ALWAYS computed and
               logged regardless of this setting (for debugging).
    When False: Only centroid distance is used for the match decision, but IoU
                is still computed and logged for debugging purposes.
    
    Typical use cases when IoU rescues association:
    - Bag flip/spin: centroid jumps but boxes still overlap
    - Partial occlusion: centroid shifts but most of box still visible
    
    Default: True
    """
    
    iou_association_threshold: float = 0.45
    """
    Minimum IoU value to associate a detection with an event.
    
    Range: 0.2 - 0.5
    - Lower values (0.2): More lenient IoU matching, catches more flip scenarios
    - Higher values (0.5): Stricter IoU matching, reduces false associations
    
    This threshold is checked in parallel with centroid distance. Association
    succeeds if EITHER (centroid_distance <= threshold) OR (IoU >= this threshold).
    
    Tuning guidelines:
    - For flip-heavy scenarios: Use 0.25-0.3
    - For more stable tracking: Use 0.35-0.45
    - Values below 0.2 may cause false associations
    - Values above 0.5 may miss legitimate flip associations
    
    Default: 0.4 (increased from 0.3 to prevent teleportation to nearby bags)
    """
    
    iou_box_margin_enabled: bool = True
    """
    Enable box margin expansion for IoU computation during flip/spin scenarios.
    
    When True: Computes IoU with both original box AND expanded box, using the
               higher value. This helps maintain tracking during rotation/flip
               where the bounding box may shift significantly but still be nearby.
    When False: Only computes IoU with the original box.
    
    This is especially useful for:
    - Bag flip/spin: where the box shape may change dramatically
    - Rotation: where the centroid shifts but the bag is still the same
    - Fast movements: where the box may trail behind the actual object position
    
    Default: True
    """
    
    iou_box_margin_ratio: float = 0.15
    """
    Ratio to expand the bounding box for margin-based IoU computation.
    
    Range: 0.1 - 0.5
    - 0.1: 10% expansion on each side (20% total increase in width/height)
    - 0.25: 25% expansion on each side (50% total increase in width/height)
    - 0.5: 50% expansion on each side (100% total increase in width/height)
    
    The expansion is applied uniformly to all sides of the event's bounding box,
    creating a larger search area for association during flip/spin scenarios.
    
    Tuning guidelines:
    - For tight tracking: Use 0.1-0.15
    - For flip-heavy scenarios: Use 0.2-0.3
    - Values above 0.4 may cause false associations with nearby objects
    
    Default: 0.25
    """
    
    iou_expanded_threshold: float = 0.35
    """
    Minimum IoU value with expanded box to associate a detection with an event.
    
    Range: 0.1 - 0.3
    This threshold is lower than iou_association_threshold because the expanded
    box naturally has higher potential for overlap. The expanded box IoU is only
    used as a fallback when normal IoU fails.
    
    Tuning guidelines:
    - Keep this value lower than iou_association_threshold
    - For flip-heavy scenarios: Use 0.1-0.15
    - Values below 0.1 may cause false associations
    
    Default: 0.15
    """
    
    # --------------------------------------------------------------------------
    # Velocity-Based Association (robust tracking during fast movements)
    # --------------------------------------------------------------------------
    
    velocity_scaling_enabled: bool = True
    """
    Enable velocity-based association distance scaling.
    
    When True: Association distance scales up for fast-moving bags
    When False: Fixed association distance only
    
    This helps maintain tracking during bag flipping/throwing.
    
    Default: True
    """
    
    velocity_scale_factor: float = 2.5
    """
    Maximum multiplier for association distance based on velocity.
    
    Range: 1.5 - 4.0
    Higher values allow tracking faster movements but may cause false associations.
    
    Default: 2.5
    """
    
    max_association_distance_px: float = 180.0
    """
    Absolute maximum association distance regardless of velocity.
    
    Range: 150 - 400
    Prevents association distance from growing too large.
    
    Default: 180.0 (reduced from 250.0 to prevent teleportation to distant bags)
    """
    
    # --------------------------------------------------------------------------
    # Hard Constraints for Preventing Teleportation (ISSUE #1 FIX)
    # --------------------------------------------------------------------------
    
    max_jump_distance_px: float = 160.0
    """
    ISSUE #1 FIX: Hard cap on centroid jump distance per association.
    
    Even if IoU or expanded IoU passes, associations are rejected if the
    centroid moves more than this distance. This prevents events from
    teleporting to distant detections during crowded scenes with multiple bags.
    
    Range: 150 - 300 pixels
    - Lower values: Stricter, may lose track during very fast movements
    - Higher values: More lenient, may allow teleportation
    
    Tuning: Should be slightly larger than max_association_distance_px to
    allow velocity-scaled associations, but still provide a hard upper limit.
    
    Default: 200.0
    """
    
    require_centroid_proximity_for_expanded_iou: bool = True
    """
    ISSUE #1 FIX: Require reasonable centroid proximity for expanded IoU associations.
    
    When True: Expanded IoU associations still require the centroid to be
    within max_centroid_distance_for_expanded_iou. This prevents expanded
    IoU from matching detections that are too far away.
    
    When False: Expanded IoU alone can trigger association regardless of
    centroid distance (not recommended in crowded scenes).
    
    Default: True
    """
    
    max_centroid_distance_for_expanded_iou: float = 250.0
    """
    ISSUE #1 FIX: Maximum centroid distance for expanded IoU associations.
    
    When require_centroid_proximity_for_expanded_iou is True, expanded IoU
    associations are only allowed if the centroid distance is within this limit.
    
    Range: 200 - 400 pixels
    - Lower values: Stricter expanded IoU matching
    - Higher values: More lenient, but may cause teleportation
    
    Should be larger than max_association_distance_px but not too large to
    allow unreasonable jumps.
    
    Default: 250.0
    """
    
    min_velocity_threshold: float = 0.01
    """
    Minimum velocity (pixels per millisecond) to trigger velocity scaling.
    
    Range: 0.005 - 0.05
    Below this velocity, standard association distance is used.
    0.01 px/ms = 10 px/s, a reasonable minimum for intentional movement.
    
    Default: 0.01
    """
    
    max_prediction_time_ms: float = 500.0
    """
    Maximum time ahead (milliseconds) to predict centroid position.
    
    Range: 200 - 1000
    Limits how far ahead velocity-based prediction can extrapolate.
    
    Default: 500.0
    """
    
    # --------------------------------------------------------------------------
    # Ghost Event Parameters (G from requirements)
    # --------------------------------------------------------------------------
    
    ghost_timeout_ms: Optional[float] = None
    """
    G: Time (milliseconds) to keep event alive without detections (DEPRECATED - use ghost_timeout_frames).
    
    For migration compatibility only. If provided, will be converted to frames using target_fps.
    
    Default: None (use frame-based threshold instead)
    """
    
    ghost_timeout_frames: int = 40
    """
    G: Frames to keep event alive without detections (frame-based threshold).
    
    Range: 12 - 50 frames
    - Lower values: Faster cleanup of lost events
    - Higher values: Survives longer occlusions (hand over bag)
    
    Tuning: Should cover typical hand occlusion duration during tying.
    25 frames @ 25fps = 1000ms (1 second) handles most manipulation scenarios.
    
    Default: 25 frames
    """
    
    # --------------------------------------------------------------------------
    # Timeout-Based Commitment Parameters (Exclusive Method)
    # --------------------------------------------------------------------------
    # NOTE: Commitment is based exclusively on timeout (idle time without detection).
    # Exit boundary logic has been removed for simplicity and robustness.
    
    commit_idle_frames: int = 18
    """
    Number of frames without detection before committing (counting) a bag.
    
    Range: 15 - 50
    - Lower values: Faster counting, may count prematurely
    - Higher values: More certain the bag is done, but slower
    
    At 25fps, default of 25 frames = 1 second of no detection.
    
    Default: 25
    """
    
    commit_min_closed_ratio: float = 0.3
    """
    Minimum ratio of closed evidence to total evidence for commitment.
    
    Range: 0.2 - 0.6
    Ensures the bag actually showed closed state before counting.
    
    Default: 0.3
    """
    
    # --------------------------------------------------------------------------
    # Anti-Double-Counting Suppression Parameters
    # --------------------------------------------------------------------------
    # These parameters prevent new events from being created for a bag that was
    # temporarily lost then re-detected after commitment.
    
    suppression_distance_px: float = 100.0
    """
    Distance (pixels) within which new events are suppressed near recent commits.
    
    Range: 100 - 250
    - Lower values: Allow events closer to recent commits
    - Higher values: More aggressive suppression, prevents double-counting
    
    Should be larger than association_distance_px to ensure bags don't get
    re-counted after brief re-detection.
    
    Default: 120.0 (reduced from 150.0 for tighter suppression zone)
    """
    
    suppression_duration_ms: Optional[float] = None
    """
    Duration (milliseconds) to suppress new events after a commit (DEPRECATED - use suppression_duration_frames).
    
    For migration compatibility only. If provided, will be converted to frames using target_fps.
    
    Default: None (use frame-based threshold instead)
    """
    
    suppression_duration_frames: int = 10
    """
    Frames to suppress new events after a commit (frame-based threshold).
    
    Range: 12 - 50 frames
    - Lower values: Allow new events sooner after commit
    - Higher values: Longer suppression window, prevents double-counting
    
    Should be long enough that a temporarily lost bag won't be re-detected
    as a new event.
    
    Default: 38 frames @ 25fps = 1520ms (increased from 25 for longer suppression window)
    """
    
    # --------------------------------------------------------------------------
    # Conditional Suppression (ISSUE #3 FIX)
    # --------------------------------------------------------------------------
    
    suppression_require_box_overlap: bool = True
    """
    ISSUE #3 FIX: Require box overlap with last committed box for suppression.
    
    When True: Suppression requires BOTH:
      1. Centroid proximity (within suppression_distance_px)
      2. Box overlap with last committed box (IoU >= suppression_iou_threshold)
    
    When False: Only centroid proximity is required (legacy behavior)
    
    This allows workers to start a new bag immediately at the same location
    after removing the previous bag, as there will be no box overlap between
    the new detection and the removed bag.
    
    Benefits:
    - Reduces false suppression when worker starts new bag quickly
    - Still prevents double-counting of the same physical bag
    - More tolerant of fast workflows
    
    Default: True
    """
    
    suppression_iou_threshold: float = 0.10
    """
    ISSUE #3 FIX: Minimum IoU with last committed box to trigger suppression.
    
    Only used when suppression_require_box_overlap is True.
    
    If a new detection has:
    - Centroid within suppression_distance_px of last commit, AND
    - IoU >= this threshold with last committed box
    Then suppression is triggered (likely the same bag re-detected).
    
    If IoU < this threshold despite proximity, the new detection is allowed
    (likely a new bag at the same location).
    
    Range: 0.1 - 0.3
    - Lower values (0.1): More aggressive suppression, catches slight movements
    - Higher values (0.3): More lenient, allows more variation
    
    Tuning: Lower than iou_association_threshold since we're looking for
    overlap with a bag that may have moved slightly before commitment.
    
    Default: 0.10 (reduced from 0.15 for more aggressive suppression)
    """
    
    # --------------------------------------------------------------------------
    # Temporal Cooldown for New Event Creation
    # --------------------------------------------------------------------------
    
    min_event_creation_interval_ms: Optional[float] = None
    """
    Minimum time (milliseconds) before allowing new event creation at same location (DEPRECATED - use temporal_cooldown_frames).
    
    For migration compatibility only. If provided, will be converted to frames using target_fps.
    
    Default: None (use frame-based threshold instead)
    """
    
    temporal_cooldown_frames: int = 12
    """
    Minimum frames before allowing new event creation at same location (frame-based threshold).
    
    After an event is committed, this cooldown prevents rapid creation of new events
    at the same spatial location. This catches detection flickering and momentary
    re-detections of the same bag.
    
    Range: 5 - 20 frames
    - Lower values: Allow new events sooner (more responsive)
    - Higher values: More aggressive duplicate prevention
    
    Works in conjunction with temporal_cooldown_distance_px to define a
    space-time exclusion zone around recently committed events.
    
    Default: 10 frames @ 25fps = 400ms
    """
    
    temporal_cooldown_distance_px: float = 120.0
    """
    Spatial distance (pixels) within which temporal cooldown applies.
    
    Defines the radius around a recently committed event's location where
    the min_event_creation_interval_ms cooldown is enforced.
    
    Range: 80 - 200
    - Lower values: Tighter cooldown zone (more localized)
    - Higher values: Wider cooldown zone (more aggressive)
    
    Should be similar to suppression_distance_px for consistency.
    
    Default: 120.0
    """
    
    # --------------------------------------------------------------------------
    # Active Event Spatial Exclusion
    # --------------------------------------------------------------------------
    
    active_event_exclusion_distance_px: float = 60.0
    """
    Distance (pixels) within which new events are blocked if an active event exists.
    
    Before creating a new event, checks if any active (non-COMMITTED) event already
    covers this spatial area. This prevents duplicate events when:
    - Detection temporarily lost and immediately re-detected
    - Multiple detections of the same bag in one frame
    - Detection flickering/splitting
    
    Range: 40 - 100
    - Lower values: Allow events closer together (may miss duplicates)
    - Higher values: More aggressive duplicate prevention (may block legitimate new bags)
    
    Should be smaller than association_distance_px since we only want to block
    very close duplicates, not all nearby bags.
    
    Default: 60.0
    """
    
    active_event_exclusion_iou: float = 0.25
    """
    IoU threshold for active event spatial exclusion.
    
    If a new detection overlaps with an existing active event's bounding box
    by this IoU amount or more, don't create a new event (likely duplicate).
    
    Range: 0.15 - 0.40
    - Lower values: More aggressive duplicate prevention
    - Higher values: Only block very similar detections
    
    Higher than suppression_iou_threshold since we're comparing with currently
    active events (should be very similar if they're duplicates).
    
    Default: 0.25
    """
    
    # --------------------------------------------------------------------------
    # Detection Clustering Parameters
    # --------------------------------------------------------------------------
    
    detection_cluster_distance_px: float = 80.0
    """
    Distance threshold for clustering nearby unassociated detections.
    
    Before creating events, nearby detections are clustered together and only
    the highest confidence detection from each cluster creates an event. This
    prevents multiple events from detection splits or flickering.
    
    Range: 50 - 120
    - Lower values: Less aggressive clustering (more events may be created)
    - Higher values: More aggressive clustering (may merge distinct bags)
    
    Should be close to association_distance_px for consistency.
    
    Default: 80.0
    """
    
    # --------------------------------------------------------------------------
    # State Transition Temporal Stability
    # --------------------------------------------------------------------------
    
    open_to_closing_time_ms: Optional[float] = None
    """
    Minimum time (milliseconds) in OPEN state before transitioning to CLOSING (DEPRECATED - use open_to_closing_frames).
    
    For migration compatibility only. If provided, will be converted to frames using target_fps.
    
    Default: None (use frame-based threshold instead)
    """
    
    open_to_closing_frames: int = 3
    """
    Minimum frames in OPEN state before transitioning to CLOSING (frame-based threshold).
    
    Range: 2 - 8 frames
    Prevents noise from immediately triggering state changes.
    
    Default: 3 frames @ 25fps = 120ms
    """
    
    closing_stability_time_ms: Optional[float] = None
    """
    Time (milliseconds) closed detections must persist for CLOSED state (DEPRECATED - use closing_stability_frames).
    
    For migration compatibility only. If provided, will be converted to frames using target_fps.
    
    Default: None (use frame-based threshold instead)
    """
    
    closing_stability_frames: int = 4
    """
    Frames closed detections must persist for CLOSED state (frame-based threshold).
    
    Range: 3 - 10 frames
    Ensures bag is actually closing, not just a detection artifact.
    
    Default: 4 frames @ 25fps = 160ms
    """
    
    closed_stability_time_ms: Optional[float] = None
    """
    Minimum time (milliseconds) in CLOSED state before COMMIT is eligible (DEPRECATED - use closed_stability_frames).
    
    For migration compatibility only. If provided, will be converted to frames using target_fps.
    
    Default: None (use frame-based threshold instead)
    """
    
    closed_stability_frames: int = 5
    """
    Minimum frames in CLOSED state before COMMIT is eligible (frame-based threshold).
    
    Range: 3 - 12 frames
    Gives time to collect ROIs and stabilize classification.
    
    Default: 5 frames @ 25fps = 200ms
    """
    
    centroid_stability_px: float = 30.0
    """
    Maximum centroid movement (pixels) to consider position "stable".
    
    Range: 10 - 50
    Used to validate geometric stability during state transitions.
    
    Default: 30.0
    """
    
    # --------------------------------------------------------------------------
    # State Reversion Parameters (prevents OPEN<->CLOSING oscillation)
    # --------------------------------------------------------------------------
    
    closing_revert_open_count: int = 3
    """
    Number of open detections in recent window to revert CLOSING -> OPEN.
    
    Range: 2 - 5
    Higher values prevent oscillation during noisy detection phases.
    
    Default: 3 (was 2, increased to reduce oscillation)
    """
    
    closing_revert_window_size: int = 5
    """
    Window size (frames) to check for revert condition.
    
    Range: 3 - 8
    Larger windows provide more stability against noise.
    
    Default: 5 (was 3, increased to reduce oscillation)
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
    # Max Event Lifetime (Force Expiration)
    # --------------------------------------------------------------------------
    
    max_event_lifetime_ms: Optional[float] = None
    """
    Maximum lifetime for an event in milliseconds (DEPRECATED - use max_event_lifetime_frames).
    
    For migration compatibility only. If provided, will be converted to frames using target_fps.
    
    Default: None (use frame-based threshold instead)
    """
    
    max_event_lifetime_frames: int = 250
    """
    Maximum lifetime for an event in frames (frame-based threshold).
    
    After this duration, the event will be expired and counted regardless of
    whether it's still on screen. This prevents events from staying active
    indefinitely when workers don't remove bags fast enough.
    
    Range: 125 - 750 frames (5-30 seconds @ 25fps)
    - Lower values: More aggressive cleanup, may count prematurely
    - Higher values: More patient, but events may accumulate
    
    Default: 250 frames @ 25fps = 10 seconds
    """
    
    # State-specific maximum lifetimes (stuck event fail-safes)
    max_open_state_frames: int = 150
    """
    Maximum frames an event can stay in OPEN state before forced transition.
    
    Range: 75 - 300 frames (3-12 seconds @ 25fps)
    Prevents events from getting stuck in OPEN state indefinitely.
    
    Default: 150 frames @ 25fps = 6 seconds
    """
    
    max_closing_state_frames: int = 75
    """
    Maximum frames an event can stay in CLOSING state before forced transition.
    
    Range: 50 - 150 frames (2-6 seconds @ 25fps)
    Prevents events from getting stuck in CLOSING state indefinitely.
    
    Default: 75 frames @ 25fps = 3 seconds
    """
    
    max_closed_state_frames: int = 100
    """
    Maximum frames an event can stay in CLOSED state before forced commit.
    
    Range: 50 - 200 frames (2-8 seconds @ 25fps)
    Prevents events from getting stuck in CLOSED state indefinitely.
    
    Default: 100 frames @ 25fps = 4 seconds
    """
    
    # --------------------------------------------------------------------------
    # Logging Control Parameters
    # --------------------------------------------------------------------------
    
    min_gap_duration_for_logging_ms: float = 500.0
    """
    Minimum detection gap duration to log.
    
    Only gaps longer than this will be logged to reduce log flooding.
    
    Range: 100 - 1000 (milliseconds)
    Default: 500.0 (0.5 seconds)
    """
    
    min_candidates_for_logging: int = 3
    """
    Minimum candidate count to log association candidates.
    
    Only log when there are this many or more competing candidates (ambiguous cases).
    
    Range: 2 - 5
    Default: 3 (truly ambiguous cases)
    """
    
    low_score_threshold: float = 0.7
    """
    Association score threshold below which to log.
    
    Associations with scores below this are considered low-confidence and logged.
    
    Range: 0.5 - 0.9
    Default: 0.7
    """
    
    noteworthy_match_types: tuple = (
        'ghost_iou_match', 'ghost_centroid_match', 'ghost_both_match',
        'expanded_iou_match', 'ghost_expanded_iou_match'
    )
    """
    Match types that are always logged.
    
    These represent special recovery cases (ghost reattachment, expanded IoU)
    that are important for debugging tracking robustness.
    """
    
    # --------------------------------------------------------------------------
    # Work Zone Configuration
    # --------------------------------------------------------------------------
    
    work_zone_enabled: bool = True
    """
    Enable work zone filtering for event creation.
    
    When True: Only create events for detections inside work zone
    When False: Entire frame is valid for event creation
    
    Default: True (enabled to suppress events outside main work area)
    """
    
    work_zone_x1: int = 0
    work_zone_y1: int = 0
    work_zone_x2: int = 1280
    work_zone_y2: int = 620
    """
    Work zone boundaries (pixels): [x1, y1] to [x2, y2]
    
    Only used when work_zone_enabled is True.
    Bottom boundary (y2) set to 620 (was 720) to move the work zone up and
    suppress events near the bottom of the frame where bags may pile up.
    """
    
    enforce_work_zone_associations: bool = True
    """
    ISSUE #2 FIX: Prevent associations for detections outside work zone.
    
    When True: Detections outside work zone won't associate with active events
    When False: Only event creation is filtered by work zone
    
    This prevents events from "following" bags that drift outside the work area.
    
    Range: True/False
    Default: True
    """
    
    out_of_zone_grace_frames: int = 5
    """
    ISSUE #2 FIX: Number of frames an event can remain outside work zone.
    
    After an event's last detection was outside the work zone for this many
    frames without new detections, it will be expired (faster than ghost_timeout).
    
    This ensures events don't stay alive indefinitely when bags drift outside
    the designated work area.
    
    Range: 3 - 20 frames
    Default: 5 (at 25fps = 200ms grace period)
    """
    
    exit_boundary_margin_px: int = 100
    """
    Exit boundary margin in pixels for visualization.
    
    Range: 20 - 100
    This is used by the Visualizer to draw the exit boundary zone.
    Note: Exit boundary logic for commitment has been removed; this is
    purely for visualization purposes.
    
    Default: 100 (increased from 50 to show a larger exit zone aligned with work area)
    """
    
    # --------------------------------------------------------------------------
    # Event-Centric ROI Collection
    # --------------------------------------------------------------------------
    
    event_max_roi_samples: int = 20
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

    # ============================================================================
    # Part 1: Size-Based Disambiguation (Brown_Orange_Overlay vs Brown_Orange_Small)
    # ============================================================================
    # These parameters control the post-detection disambiguation between visually
    # similar classes using RAW bounding box area on CLOSED state ROIs only.
    
    disambiguation_enabled: bool = _parse_bool_env("DISAMBIGUATION_ENABLED", True)
    """
    Enable/disable size-based disambiguation between Brown_Orange_Overlay and Brown_Orange_Small.
    
    When True: Uses raw bbox area on CLOSED ROIs to disambiguate similar classes
    When False: Relies solely on YOLO classifier predictions
    
    Default: True
    """
    
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    """
    Class pair (family members) to disambiguate using size-based logic.
    First element = "regular/larger" class, Second = "small" class.
    
    When the classifier returns ANY of these classes, we treat them as a "family"
    and decide the specific class purely based on size measurement on CLOSED ROIs.
    """
    
    disambiguation_family_name: str = 'Brown_Orange_Family'
    """
    Name of the class family for logging and future classifier training.
    
    This is used for:
    1. Debug logging to show which family a detection belongs to
    2. Future-proofing: if the classifier is retrained to return this family name
       directly (e.g., "Brown_Orange_Family"), the disambiguation logic will 
       automatically recognize it and apply size-based decision.
    
    Default: 'Brown_Orange_Family'
    """
    
    disambiguation_small_threshold: float = _parse_float_env("DISAMBIGUATION_SMALL_THRESHOLD", 7000.0)
    """
    Raw area threshold (pixels²) below which a detection is classified as "small".
    
    If raw_area < small_threshold => force class to Brown_Orange_Small
    
    Range: 8000 - 15000 (depends on camera setup and bag sizes)
    Tuning: Plot raw_area vs true label to find optimal threshold.
    
    Note: This is RAW AREA in pixels², not adjusted for perspective.
    Only applies to CLOSED state ROIs.
    
    Default: 10000.0 (pixels²)
    """
    
    disambiguation_regular_threshold: float = _parse_float_env("DISAMBIGUATION_REGULAR_THRESHOLD", 8500.0)
    """
    Raw area threshold (pixels²) above which a detection is classified as "regular/overlay".
    
    If raw_area > regular_threshold => force class to Brown_Orange_Overlay
    
    Range: 15000 - 30000 (depends on camera setup and bag sizes)
    Tuning: Plot raw_area vs true label to find optimal threshold.
    
    Note: This is RAW AREA in pixels², not adjusted for perspective.
    Only applies to CLOSED state ROIs.
    
    Default: 20000.0 (pixels²)
    """
    
    disambiguation_gray_zone_behavior: str = 'keep_original'
    """
    Behavior when raw_area falls in the gray zone (between small and regular thresholds).
    Options:
    - 'keep_original': Keep YOLO's original prediction
    - 'uncertain': Return "Uncertain" classification
    - 'prefer_small': Prefer small classification in gray zone
    - 'prefer_regular': Prefer regular/overlay classification in gray zone
    
    Default: 'keep_original'
    """
    
    disambiguation_debug_logging: bool = _parse_bool_env("DISAMBIGUATION_DEBUG", True)
    """
    Enable detailed debug logging for disambiguation tuning.
    
    When True: Logs raw area, decision, and reasoning for each disambiguation
    
    Default: False
    """
    
    disambiguation_confidence_penalty: float = _parse_float_env("DISAMBIGUATION_CONFIDENCE_PENALTY", 0.9)
    """
    Confidence multiplier applied when disambiguation overrides the classifier.
    
    When disambiguation changes the label, the confidence is multiplied by this factor.
    Range: 0.8 - 1.0 (1.0 = no penalty)
    
    Default: 0.9 (10% reduction)
    """
    
    disambiguation_penalty_on_change_only: bool = _parse_bool_env("DISAMBIGUATION_PENALTY_ON_CHANGE_ONLY", False)
    """
    Only apply confidence penalty when size-based decision differs from classifier.
    
    When False (default): Always apply penalty for family members (conservative)
    When True: Only apply penalty when classifier's prediction differs from size decision
    
    Rationale:
    - False = conservative: All family detections are treated as "size-decided", 
      so confidence reflects this override regardless of classifier accuracy.
    - True = optimistic: Trust classifier's confidence when it agrees with size.
    
    Default: False (conservative approach)
    """
    
    # ============================================================================
    # Part 1.5: Probability Mass Transfer (Variant B)
    # ============================================================================
    # These parameters control how probability vectors are adjusted after
    # disambiguation to ensure the evidence accumulator reflects the
    # disambiguated label decision.
    
    prob_adjustment_strategy: str = 'proportional_transfer'
    """
    Strategy for transferring probability mass after disambiguation.
    
    Options:
    - 'full_transfer': Transfer ALL family mass to target class (default)
    - 'proportional_transfer': Transfer only from source, proportionally
    - 'swap': Swap probabilities between source and target
    
    Default: 'full_transfer' (most conservative, ensures accumulator respects disambiguation)
    """
    
    prob_adjustment_transfer_ratio: float = _parse_float_env("PROB_ADJUSTMENT_TRANSFER_RATIO", 0.5)
    """
    Transfer ratio for 'proportional_transfer' strategy.
    
    Amount to transfer = from_label_prob * transfer_ratio
    
    Range: 0.0 - 1.0
    Default: 1.0 (full transfer)
    """
    
    prob_adjustment_epsilon: float = 1e-9
    """
    Epsilon value for numerical stability in probability adjustments.
    
    Used to avoid exact zeros which can cause issues with log-evidence.
    
    Default: 1e-9
    """
    
    prob_adjustment_debug_logging: bool = _parse_bool_env("PROB_ADJUSTMENT_DEBUG", True)
    """
    Enable detailed debug logging for probability adjustments.
    
    When True: Logs each adjustment with before/after values
    When False: Silent (only summary in metadata)
    
    Default: False (production)
    """
    
    # ============================================================================
    # Part 2: Trust-Weighted Temporal Evidence Accumulation
    # ============================================================================
    # These parameters control the noise-resistant track-level classification
    # using trust-weighted log evidence aggregation.
    
    evidence_accumulation_enabled: bool = _parse_bool_env("EVIDENCE_ACCUMULATION_ENABLED", True)
    """
    Enable trust-weighted log-evidence accumulation for classification.
    
    When True: Uses weighted log-evidence across ROIs
    When False: Uses legacy evidence-based classification
    
    Default: True
    """
    
    # --------------------------------------------------------------------------
    # Trust Scoring Parameters
    # --------------------------------------------------------------------------
    
    trust_open_max: float = _parse_float_env("TRUST_OPEN_MAX", 1.0)
    """
    Maximum trust score for Open ROIs.
    
    Open ROIs typically have better view of bag details.
    Range: 0.8 - 1.0
    
    Default: 1.0
    """
    
    trust_closed_max: float = _parse_float_env("TRUST_CLOSED_MAX", 0.7)
    """
    Maximum trust score for Closed ROIs (capped).
    
    Closed ROIs may have caps/deformation but still provide regularization.
    Range: 0.5 - 0.8
    
    Default: 0.7
    """
    
    trust_sharpness_min: float = _parse_float_env("TRUST_SHARPNESS_MIN", 100.0)
    """
    Minimum sharpness value for trust normalization.
    
    Sharpness below this value gets lowest trust component.
    Range: 50 - 200
    
    Default: 100.0
    """
    
    trust_sharpness_max: float = _parse_float_env("TRUST_SHARPNESS_MAX", 800.0)
    """
    Maximum sharpness value for trust normalization.
    
    Sharpness above this value gets highest trust component.
    Range: 500 - 1500
    
    Default: 800.0
    """
    
    trust_blur_penalty: float = _parse_float_env("TRUST_BLUR_PENALTY", 0.3)
    """
    Penalty factor for blurry ROIs (low sharpness).
    
    Applied as: trust = trust * (1 - blur_penalty) when sharpness is low.
    Range: 0.1 - 0.5
    
    Default: 0.3
    """
    
    trust_size_stability_tolerance: float = _parse_float_env("TRUST_SIZE_TOLERANCE", 0.3)
    """
    Tolerance for ROI size variation from median (as fraction).
    
    ROIs with size deviation > tolerance are penalized.
    Range: 0.2 - 0.5
    
    Default: 0.3
    """
    
    trust_min_for_support: float = _parse_float_env("TRUST_MIN_FOR_SUPPORT", 0.4)
    """
    Minimum trust score for an ROI to count as "supporting" evidence.
    
    Used in stability gate to count trusted ROIs.
    Range: 0.3 - 0.6
    
    Default: 0.4
    """
    
    # --------------------------------------------------------------------------
    # Evidence Accumulation Parameters
    # --------------------------------------------------------------------------
    
    evidence_epsilon: float = _parse_float_env("EVIDENCE_EPSILON", 1e-6)
    """
    Small constant to avoid log(0) in evidence calculation.
    
    Range: 1e-9 to 1e-4
    
    Default: 1e-6
    """
    
    evidence_top_k_rois: int = _parse_int_env("EVIDENCE_TOP_K_ROIS", 7)
    """
    Number of top ROIs (by trust) to use for evidence accumulation.
    
    Quality-first selection: pick best K ROIs regardless of total count.
    Range: 3 - 10
    
    Default: 7
    """
    
    # --------------------------------------------------------------------------
    # Temporal Consistency Parameters (Class Switch Penalty)
    # --------------------------------------------------------------------------
    
    temporal_inertia_enabled: bool = _parse_bool_env("TEMPORAL_INERTIA_ENABLED", True)
    """
    Enable class-switch penalty to reduce flip-flopping within tracks.
    
    When True: Penalizes late class switches unless evidence is overwhelming
    When False: Each classification decision is independent
    
    Default: True
    """
    
    temporal_inertia_strength: float = _parse_float_env("TEMPORAL_INERTIA_STRENGTH", 0.15)
    """
    Strength of the inertia/penalty for class switching.
    
    Applied as a log-evidence bonus to the previously leading class.
    Range: 0.1 - 0.3
    
    Default: 0.15
    """
    
    temporal_inertia_decay: float = _parse_float_env("TEMPORAL_INERTIA_DECAY", 0.8)
    """
    Decay factor for inertia over subsequent ROIs.
    
    Each new ROI reduces inertia by this factor, allowing legitimate switches.
    Range: 0.6 - 0.95
    
    Default: 0.8
    """
    
    # --------------------------------------------------------------------------
    # Stability Gate Parameters
    # --------------------------------------------------------------------------
    
    stability_gate_enabled: bool = _parse_bool_env("STABILITY_GATE_ENABLED", True)
    """
    Enable stability gate to prevent forced decisions under ambiguity.
    
    When True: Returns "Uncertain" when evidence is insufficient
    When False: Always returns best available prediction
    
    Default: True
    """
    
    stability_margin_threshold: float = _parse_float_env("STABILITY_MARGIN_THRESHOLD", 0.5)
    """
    Minimum log-evidence margin between winner and runner-up.
    
    If margin < threshold, result is marked as "Uncertain".
    Range: 0.2 - 1.0
    
    Default: 0.5
    """
    
    stability_min_trusted_rois: int = _parse_int_env("STABILITY_MIN_TRUSTED_ROIS", 2)
    """
    Minimum number of trusted ROIs (trust >= trust_min_for_support) required.
    
    If fewer trusted ROIs, result is marked as "Uncertain".
    Range: 1 - 5
    
    Default: 2
    """

    use_frame_timestamps: bool = IS_WINDOWS

    # ==========================================================================
    # Testing Mode Time Scaling (for Windows/Development environments)
    # ==========================================================================
    
    testing_time_scale_factor: float = 1.0
    """
    Time scaling multiplier for testing/development mode.
    
    When running on slower hardware (e.g., Windows PCs without BPU acceleration),
    the system processes all frames but at a slower effective speed than production.
    This parameter scales all time-based thresholds to compensate for the slower
    processing speed, ensuring event lifecycles behave as they would in production.
    
    Calculation:
        scale_factor = (actual_processing_time_per_frame / target_frame_time)
        
        For example, if production runs at 25fps (40ms per frame) but testing
        takes 200ms per frame (5fps effective), the scale factor would be:
        scale_factor = 200ms / 40ms = 5.0
    
    Usage:
        - 1.0 (default): No scaling, production behavior
        - 5.0: Testing runs 5x slower, multiply all timeouts by 5
        - Auto: Set to 0.0 to auto-calculate based on measured FPS
    
    Affects ALL time-based parameters:
        - association_time_ms
        - ghost_timeout_ms
        - max_event_lifetime_ms
        - suppression_duration_ms
        - min_event_creation_interval_ms
        - open_to_closing_time_ms
        - closing_stability_time_ms
        - closed_stability_time_ms
        - max_prediction_time_ms
        - degraded_mode_delay_threshold_ms
        - min_gap_duration_for_logging_ms
    
    Note: This only affects time-based parameters when use_frame_timestamps=True
          (typically Windows/testing mode). Frame-based parameters (commit_idle_frames,
          etc.) are not affected as they already scale naturally with frame rate.
    
    Range: 1.0 - 20.0 (or 0.0 for auto)
    Default: 1.0 (no scaling)
    """
    
    enable_auto_time_scaling: bool = IS_WINDOWS
    """
    Automatically calculate time scaling factor based on actual processing speed.
    
    When enabled, the system measures the actual time taken per frame and
    calculates the appropriate time scaling factor dynamically. This is useful
    when the processing speed varies or is unknown.
    
    The auto-calculated factor is applied after a warm-up period (first 100 frames)
    to ensure stable measurements.
    
    Precedence: If both `enable_auto_time_scaling=True` and a manual 
    `testing_time_scale_factor` are set, the manual factor is used initially,
    then replaced by the auto-calculated factor after warmup (if significantly
    different). To maintain a fixed manual factor, set `enable_auto_time_scaling=False`.
    
    Default: True on Windows, False on RDK (production)
    """
    
    auto_scaling_target_frame_time_ms: float = 40.0
    """
    Target frame time in milliseconds for auto-scaling calculation.
    Default: 40ms (25fps). The auto-scaling factor is calculated as:
    measured_frame_time / target_frame_time.
    """
    
    auto_scaling_warmup_frames: int = 100
    """
    Number of frames to process before calculating auto-scaling factor.
    Ensures stable measurements by allowing system to reach steady state.
    Default: 100 frames
    """
    
    auto_scaling_activation_threshold: float = 1.2
    """
    Minimum scale factor to activate auto-scaling.
    If calculated factor is below this threshold (processing close to real-time),
    no scaling is applied.
    Range: 1.1 - 2.0
    Default: 1.2 (20% slower than target)
    """
    
    # Target FPS for ms-to-frames conversion
    target_fps: float = 25.0
    """
    Target FPS for converting millisecond thresholds to frame-based thresholds.
    
    This is a configuration constant (not measured FPS) used to ensure consistent
    behavior across different processing speeds. When time-based (ms) parameters
    are provided, they are converted to frames using this target FPS.
    
    Default: 25.0 fps (matches production frame rate)
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
        enforce_work_zone_associations=tracking_config.enforce_work_zone_associations,
        out_of_zone_grace_frames=tracking_config.out_of_zone_grace_frames,
        
        # Association (D, T)
        association_distance_px=tracking_config.association_distance_px,
        association_time_ms=tracking_config.association_time_ms,
        
        # IoU-based association
        iou_association_enabled=tracking_config.iou_association_enabled,
        iou_association_threshold=tracking_config.iou_association_threshold,
        
        # IoU box margin expansion (for flip/spin scenarios)
        iou_box_margin_enabled=tracking_config.iou_box_margin_enabled,
        iou_box_margin_ratio=tracking_config.iou_box_margin_ratio,
        iou_expanded_threshold=tracking_config.iou_expanded_threshold,
        
        # Velocity-based association
        velocity_scaling_enabled=tracking_config.velocity_scaling_enabled,
        velocity_scale_factor=tracking_config.velocity_scale_factor,
        max_association_distance_px=tracking_config.max_association_distance_px,
        min_velocity_threshold=tracking_config.min_velocity_threshold,
        max_prediction_time_ms=tracking_config.max_prediction_time_ms,
        
        # Hard constraints for preventing teleportation (Issue #1)
        max_jump_distance_px=tracking_config.max_jump_distance_px,
        require_centroid_proximity_for_expanded_iou=tracking_config.require_centroid_proximity_for_expanded_iou,
        max_centroid_distance_for_expanded_iou=tracking_config.max_centroid_distance_for_expanded_iou,
        
        # Ghost (G) - frame-based with ms migration
        ghost_timeout_ms=tracking_config.ghost_timeout_ms,
        ghost_timeout_frames=tracking_config.ghost_timeout_frames,
        
        # Max event lifetime - frame-based with ms migration
        max_event_lifetime_ms=tracking_config.max_event_lifetime_ms,
        max_event_lifetime_frames=tracking_config.max_event_lifetime_frames,
        max_open_state_frames=tracking_config.max_open_state_frames,
        max_closing_state_frames=tracking_config.max_closing_state_frames,
        max_closed_state_frames=tracking_config.max_closed_state_frames,
        
        # Timeout-based commitment (exclusive method)
        commit_idle_frames=tracking_config.commit_idle_frames,
        commit_min_closed_ratio=tracking_config.commit_min_closed_ratio,
        
        # Anti-double-counting suppression - frame-based with ms migration
        suppression_distance_px=tracking_config.suppression_distance_px,
        suppression_duration_ms=tracking_config.suppression_duration_ms,
        suppression_duration_frames=tracking_config.suppression_duration_frames,
        suppression_require_box_overlap=tracking_config.suppression_require_box_overlap,
        suppression_iou_threshold=tracking_config.suppression_iou_threshold,
        
        # Temporal cooldown for new event creation - frame-based with ms migration
        min_event_creation_interval_ms=tracking_config.min_event_creation_interval_ms,
        temporal_cooldown_frames=tracking_config.temporal_cooldown_frames,
        temporal_cooldown_distance_px=tracking_config.temporal_cooldown_distance_px,
        
        # Active event spatial exclusion
        active_event_exclusion_distance_px=tracking_config.active_event_exclusion_distance_px,
        active_event_exclusion_iou=tracking_config.active_event_exclusion_iou,
        
        # Detection clustering
        detection_cluster_distance_px=tracking_config.detection_cluster_distance_px,
        
        # State transition timing - frame-based with ms migration
        open_to_closing_time_ms=tracking_config.open_to_closing_time_ms,
        open_to_closing_frames=tracking_config.open_to_closing_frames,
        closing_stability_time_ms=tracking_config.closing_stability_time_ms,
        closing_stability_frames=tracking_config.closing_stability_frames,
        closed_stability_time_ms=tracking_config.closed_stability_time_ms,
        closed_stability_frames=tracking_config.closed_stability_frames,
        centroid_stability_px=tracking_config.centroid_stability_px,
        
        # State reversion (anti-oscillation)
        closing_revert_open_count=tracking_config.closing_revert_open_count,
        closing_revert_window_size=tracking_config.closing_revert_window_size,
        
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
        max_open_roi_samples=tracking_config.max_open_samples,
        max_closed_roi_samples=tracking_config.max_closed_samples,
        
        # Classification voting
        min_voting_agreement_pct=tracking_config.voting_agreement_threshold_pct,
        confidence_margin_threshold=tracking_config.confidence_margin_threshold,
        
        # Resource limits
        max_active_events=tracking_config.max_active_events,
        
        # Logging control
        min_gap_duration_for_logging_ms=tracking_config.min_gap_duration_for_logging_ms,
        min_candidates_for_logging=tracking_config.min_candidates_for_logging,
        low_score_threshold=tracking_config.low_score_threshold,
        noteworthy_match_types=tracking_config.noteworthy_match_types,

        use_frame_timestamps=tracking_config.use_frame_timestamps,
        
        # Testing mode time scaling
        testing_time_scale_factor=tracking_config.testing_time_scale_factor,
        enable_auto_time_scaling=tracking_config.enable_auto_time_scaling,
        auto_scaling_target_frame_time_ms=tracking_config.auto_scaling_target_frame_time_ms,
        auto_scaling_warmup_frames=tracking_config.auto_scaling_warmup_frames,
        auto_scaling_activation_threshold=tracking_config.auto_scaling_activation_threshold,
        
        # Target FPS for ms-to-frames conversion
        target_fps=tracking_config.target_fps,
    )
