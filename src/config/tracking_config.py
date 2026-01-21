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


def _parse_str_env(env_var: str, default: str) -> str:
    """Parse string from environment variable."""
    return os.getenv(env_var, default)


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
    
    min_open_frames: int = 3  # V3: Reduced from 5 for faster state transitions
    """
    Minimum consecutive frames a bag must be detected as "open" before allowing 
    transition to "closed" state.
    
    Range: 1 - 20
    - Lower values: Faster state transitions, more responsive
    - Higher values: More stable, reduces noise-induced transitions
    
    Default: 4 (V3: reduced from 5)
    """
    
    min_closed_frames: int = 5  # V3: Reduced from 3 for faster counting
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
    
    max_open_samples: int = 10
    """
    Maximum number of ROI samples to collect during the 'open' phase.
    
    Range: 4 - 15
    More samples provide better classification but use more memory.
    
    BALANCED COLLECTION: Set to 10 to match max_closed_samples for balanced
    evidence accumulation.Previously was 15, creating imbalance.
    
    Default: 10 (balanced with closed samples)
    """
    
    max_closed_samples: int = 10
    """
    Maximum number of ROI samples to collect during the 'closed' phase.
    
    Range: 2 - 10
    More samples provide better classification but use more memory.
    
    BALANCED COLLECTION: Set to 10 to match max_open_samples.Closed ROIs are
    essential for size-based disambiguation, so equal collection is critical.
    Previously was 5, which was insufficient.
    
    Default: 10 (balanced with open samples)
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
    min_size=300.This was preventing classification from ever running.
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
    
    top_k_candidates: int = 10
    """
    Number of top ROI candidates to select for classification at track end.
    
    Range: 5 - 10
    These are selected using stratified sampling to ensure minimum closed ROI
    representation (see min_closed_rois_in_top_k).
    
    Increased from 7 to 10 to ensure sufficient evidence with stratified selection.
    
    Default: 10
    """
    
    min_closed_rois_in_top_k: int = 3
    """
    Minimum number of closed ROIs to guarantee in top-K selection.
    
    Range: 2 - 5
    Ensures that size-based disambiguation has sufficient closed ROIs available.
    Top-K selection uses stratified sampling: guarantees min_closed closed ROIs
    if available, then fills remaining slots with best ROIs by trust.
    
    This prevents scenarios where all top-K ROIs are open (due to higher trust/
    sharpness) leaving zero closed ROIs for disambiguation.
    
    Default: 3
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
    DEPRECATED: This parameter is unused in the current implementation.
    
    Evidence accumulation uses margin-based decision (winner vs runner-up margin)
    rather than absolute evidence score threshold.Log-evidence scores are negative,
    making this positive threshold meaningless.
    
    See stability_margin_threshold for the actual decision threshold used.
    
    Kept for backward compatibility. Will be removed in future version.
    
    Default: 0.3 (unused)
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
    
    classifier_reject_labels: tuple = ('Rejected',)
    """
    List of classifier labels to reject/skip during voting and aggregation.
    
    Predictions with these labels are excluded from evidence accumulation and
    do not contribute to the final classification decision. This is useful for
    handling low-quality frames or ambiguous predictions that the classifier
    explicitly marks as unreliable.
    
    Behavior:
    - ROIs with reject labels are skipped during evidence accumulation
    - They do not count toward the minimum candidates threshold
    - They do not contribute to winner/runner-up scoring
    - If ALL predictions are rejected, track is classified as "Unknown"
    
    Default: ('Rejected',)
    
    Example:
        classifier_reject_labels = ('Rejected', 'Uncertain', 'LowQuality')
    """
    
    # ============================================================================
    # Classification Stability Heuristics (Production-Grade)
    # ============================================================================
    
    enable_label_reuse: bool = _parse_bool_env("ENABLE_LABEL_REUSE", False)
    """
    Enable previous-label reuse when confidence is low but evidence is strong.
    
    DEPRECATED: This feature adds complexity and may not be needed with improved
    evidence accumulation. Disabled by default.
    
    When True: Allows reusing previous track classification if current confidence
               is below LOW_CONF_THRESHOLD but there's strong historical evidence
               (streak length >= STREAK_MIN, burst dominance, etc.)
    When False: Always use current classification (recommended behavior)
    
    Feature-flagged for safe rollout. Can also be controlled via environment:
        ENABLE_LABEL_REUSE=true
    
    Consider removing this feature in a future release after validating that
    improved evidence accumulation provides sufficient reliability.
    
    Default: False (disabled, consider removing feature)
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
    # Smart Frame Skipping in Degraded Mode
    # ============================================================================
    
    degraded_mode_smart_skip_enabled: bool = True
    """
    Enable smart pattern-based frame skipping in degraded mode.
    
    When True: Uses intelligent skip patterns (every 2nd/3rd frame) to reduce load
              while ensuring events receive sufficient frames for tracking
    When False: Uses legacy binary skip logic (all or nothing)
    
    Smart skipping ensures:
    - Events get minimum required frames for detection/tracking
    - Critical states (CLOSING, early OPEN) are never skipped
    - Skip pattern adapts to queue pressure and active event count
    
    Default: True
    """
    
    degraded_mode_skip_pattern: str = 'adaptive'
    """
    Frame skip pattern to use in degraded mode.
    
    Options:
    - 'every_2nd': Skip every 2nd frame (50% reduction, processes 50% of frames)
    - 'every_3rd': Skip every 3rd frame (33% reduction, processes 67% of frames)
    - 'adaptive': Dynamically adjust based on queue pressure:
        * 50-70% queue: skip every 3rd frame (mild load)
        * 70-85% queue: skip every 2nd frame (moderate load)
        * 85-95% queue: skip 2 out of 3 frames (heavy load)
        * 95%+ queue: skip 3 out of 4 frames (critical load)
    
    Recommendation: Use 'adaptive' for production (best balance)
    
    Default: 'adaptive'
    """
    
    degraded_mode_min_frames_per_event: int = 15
    """
    Minimum frames an event must receive for reliable tracking.
    
    Smart skipping ensures each event gets at least this many frames before
    it can be committed or expired. This prevents skipping from breaking tracking.
    
    Range: 10 - 25 frames
    - Lower values (10-12): More aggressive skipping, may affect tracking quality
    - Higher values (20-25): Conservative, better tracking but less skip benefit
    
    Calculation basis:
    - ghost_timeout_frames = 40 frames
    - commit_idle_frames = 18 frames
    - With 50% skip rate: 40 frames → 20 processed
    - Minimum 15 frames ensures adequate state transitions
    
    Default: 15 (ensures reliable tracking with 50% skip rate)
    """
    
    degraded_mode_skip_with_active_events_only: bool = False
    """
    Only apply smart skipping when active events exist.
    
    When True: No skipping when no events are active (preserve responsiveness)
    When False: Always apply skip pattern in degraded mode (maximize throughput)
    
    Recommendation: False for production (consistent throughput)
    
    Default: False
    """
    
    degraded_mode_preserve_critical_states: bool = True
    """
    Never skip frames when events are in critical states.
    
    When True: Processes all frames when any event is in CLOSING or early OPEN state
    When False: Applies skip pattern regardless of event states
    
    Critical states:
    - CLOSING: Bag is being tied, need continuous frames for state transition
    - Early OPEN: First few frames of a new event, critical for association
    
    Default: True (ensures reliable state transitions)
    """
    
    degraded_mode_critical_state_frame_threshold: int = 5
    """
    Number of frames to consider an OPEN event as "early" (critical).
    
    Events in OPEN state for fewer than this many frames are considered critical
    and will prevent frame skipping.
    
    Range: 3 - 10 frames
    
    Default: 5 (ensures new events get good initial tracking)
    """
    
    degraded_mode_max_skip_rate_with_events: float = 0.5
    """
    Maximum skip rate when active events exist.
    
    Limits how aggressively we skip when tracking is active.
    Range: 0.3 - 0.6 (30% - 60%)
    
    Examples:
    - 0.5: Skip at most 50% of frames (every 2nd frame)
    - 0.33: Skip at most 33% of frames (every 3rd frame)
    
    Default: 0.5 (balanced throughput and tracking quality)
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
    
    association_distance_px: float = 60.0
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
    
    max_association_distance_px: float = 160.0
    """
    Absolute maximum association distance regardless of velocity.
    
    Range: 150 - 400
    Prevents association distance from growing too large.
    
    Default: 180.0 (reduced from 250.0 to prevent teleportation to distant bags)
    """
    
    # --------------------------------------------------------------------------
    # Hard Constraints for Preventing Teleportation (ISSUE #1 FIX)
    # --------------------------------------------------------------------------
    
    max_jump_distance_px: float = 140.0
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
    # Adaptive Ghost Timeout (V6 Performance Optimization)
    # --------------------------------------------------------------------------
    # Ghost timeout scales with object velocity to handle spinning/thrown bags
    
    adaptive_ghost_timeout_enabled: bool = True
    """
    Enable velocity-based ghost timeout scaling.
    
    When True: ghost_timeout = base_timeout + k * recent_velocity
    When False: Use fixed ghost_timeout_frames
    
    Benefits:
    - Spinning objects survive short occlusions (higher velocity = longer timeout)
    - Thrown/fast objects terminate quickly (prevent stale events)
    - More responsive to object motion dynamics
    
    Default: True
    """
    
    adaptive_ghost_velocity_factor: float = 2.0
    """
    Velocity scaling factor (k) for adaptive ghost timeout.
    
    Formula: ghost_timeout = base_timeout + k * velocity_magnitude * time_scale
    
    Range: 1.0 - 5.0
    - Higher values: More tolerance for fast-moving objects
    - Lower values: More conservative timeout scaling
    
    Default: 2.0
    """
    
    adaptive_ghost_min_timeout_frames: int = 15
    """
    Minimum ghost timeout frames (floor for adaptive scaling).
    
    Range: 10 - 30 frames
    Prevents timeout from being too short for slow-moving objects.
    
    Default: 15 frames
    """
    
    adaptive_ghost_max_timeout_frames: int = 75
    """
    Maximum ghost timeout frames (ceiling for adaptive scaling).
    
    Range: 50 - 150 frames
    Prevents timeout from being too long for very fast objects.
    
    Default: 75 frames
    """
    
    # --------------------------------------------------------------------------
    # Timeout-Based Commitment Parameters (Exclusive Method)
    # --------------------------------------------------------------------------
    # NOTE: Commitment is based exclusively on timeout (idle time without detection).
    # Exit boundary logic has been removed for simplicity and robustness.
    
    commit_idle_frames: int = 6
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
    
    suppression_distance_px: float = 120.0
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
      1. Centroid proximity (within suppression_distance_px or adaptive distance)
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
    
    suppression_iou_threshold: float = 0.25
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
    # Size-Adaptive Suppression (ISSUE #4 FIX)
    # --------------------------------------------------------------------------
    # Problem: Large bags overcount (+8-13%), small bags undercount (-12%)
    # Root cause: Fixed suppression distance doesn't account for bag size variance
    # Solution: Make suppression distance proportional to bag diagonal
    
    suppression_use_adaptive_distance: bool = _parse_bool_env("SUPPRESSION_USE_ADAPTIVE_DISTANCE", True)
    """
    ISSUE #4 FIX: Enable size-adaptive suppression distance.
    
    When True: Suppression distance = bag_diagonal * suppression_diagonal_multiplier
               This ensures larger bags have larger suppression zones and smaller
               bags have smaller suppression zones.
    When False: Use fixed suppression_distance_px for all bags (legacy behavior)
    
    Benefits:
    - Large bags: Larger suppression zone prevents overcounting
    - Small bags: Smaller suppression zone prevents undercounting
    - More accurate counting across all bag sizes
    
    Environment: SUPPRESSION_USE_ADAPTIVE_DISTANCE=true
    Default: True
    """
    
    suppression_diagonal_multiplier: float = _parse_float_env("SUPPRESSION_DIAGONAL_MULTIPLIER", 1.2)
    """
    Multiplier for bag diagonal when calculating adaptive suppression distance.
    
    Formula: effective_suppression_distance = bag_diagonal * multiplier
    
    Range: 1.0 - 2.5
    - 1.0: Suppression zone equals bag diagonal (tight)
    - 1.5: Suppression zone 50% larger than diagonal (recommended)
    - 2.0: Suppression zone twice the diagonal (aggressive)
    
    Tuning guidelines:
    - If large bags still overcount: Increase multiplier (1.75, 2.0)
    - If small bags still undercount: Decrease multiplier (1.25, 1.0)
    - Start with 1.5 and adjust based on production results
    
    Environment: SUPPRESSION_DIAGONAL_MULTIPLIER=1.5
    Default: 1.5
    """
    
    suppression_min_distance_px: float = _parse_float_env("SUPPRESSION_MIN_DISTANCE_PX", 60.0)
    """
    Minimum suppression distance (floor for adaptive calculation).
    
    Ensures very small bags still have a reasonable suppression zone.
    The adaptive distance will never go below this value.
    
    Range: 40 - 100 pixels
    Default: 60.0 pixels
    """
    
    suppression_max_distance_px: float = _parse_float_env("SUPPRESSION_MAX_DISTANCE_PX", 250.0)
    """
    Maximum suppression distance (ceiling for adaptive calculation).
    
    Prevents very large bags from having unreasonably large suppression zones
    that might block legitimate new bag detections.
    
    Range: 150 - 400 pixels
    Default: 250.0 pixels
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

    temporal_cooldown_frames: int = 10
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
    
    active_event_exclusion_distance_px: float = 70.0
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
    
    active_event_exclusion_iou: float = 0.20
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

    min_open_duration_ms: float = 500.0
    """
    Minimum duration (milliseconds) an event must remain in OPEN state before transitioning to CLOSING.
    
    This parameter prevents "ghost" events caused by detection flicker where a closed bag
    might briefly be detected as open, creating a new event that quickly transitions and counts.
    By enforcing a minimum duration in OPEN state, we ensure that only events representing
    real "human-scale" bag manipulation are allowed to proceed to CLOSING/CLOSED states.
    
    Range: 200 - 1000 ms
    - Lower values (200-300ms): More responsive, may count flickers
    - Higher values (500-1000ms): More conservative, filters out quick flickers
    
    Typical human manipulation takes at least 500ms, so this default ensures that
    only genuine bag opening events are counted.
    
    Default: 500.0 ms
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
    
    min_closed_evidence_count: int = 3
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
    max_open_state_frames: int = 100
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
    
    disambiguation_small_threshold: float = _parse_float_env("DISAMBIGUATION_SMALL_THRESHOLD", 15200.0)
    """
    Raw area threshold (pixels²) below which a detection is classified as "small".
    
    If raw_area < small_threshold => force class to Brown_Orange_Small
    
    Range: 8000 - 15000 (depends on camera setup and bag sizes)
    Tuning: Plot raw_area vs true label to find optimal threshold.
    
    Production Value: 9000.0 pixels²
    Rationale (based on log data analysis):
      - Case 2 logs show all true Small events have area < 10,000
      - Setting to 9,000 provides 1,000 px² safety margin
      - Catches 90%+ of true Small bags with high confidence
      - See docs/ROI_FILTERING_AND_THRESHOLDS.md for details
    
    Note: This is RAW AREA in pixels², not adjusted for perspective.
    Only applies to CLOSED state ROIs.
    
    Default: 9000.0 (pixels²) - UPDATED from 7000.0 based on production logs
    """
    
    disambiguation_regular_threshold: float = _parse_float_env("DISAMBIGUATION_REGULAR_THRESHOLD", 18600.0)
    """
    Raw area threshold (pixels²) above which a detection is classified as "regular/overlay".
    
    If raw_area > regular_threshold => force class to Brown_Orange_Overlay
    
    Range: 15000 - 30000 (depends on camera setup and bag sizes)
    Tuning: Plot raw_area vs true label to find optimal threshold.
    
    Production Value: 11000.0 pixels²
    Rationale (based on log data analysis):
      - Case 1 logs show most true Overlay events have area > 10,000
      - Setting to 11,000 provides 1,000 px² safety margin above boundary
      - Catches 85%+ of true Overlay bags with high confidence
      - Gray zone [9000, 11000] covers observed ambiguous range (8200-9900)
      - See docs/ROI_FILTERING_AND_THRESHOLDS.md for details
    
    Note: This is RAW AREA in pixels², not adjusted for perspective.
    Only applies to CLOSED state ROIs.
    
    Default: 11000.0 (pixels²) - UPDATED from 8500.0 based on production logs
    """

    # === Gray Zone Confidence Penalties ===

    disambiguation_gray_zone_penalty_homography: float = 0.75
    """
    Confidence penalty for gray zone classifications with homography. 

    Gray zone indicates ambiguous size.  Penalty ensures confidence is low enough
    to trigger bidirectional smoother's batch-context resolution. 

    Range: 0.6-0.85
    - Lower: More aggressive, ensures context override
    - Higher: More conservative, trusts size measurement more

    Default: 0.75 (25% penalty)
    """

    disambiguation_gray_zone_penalty_pixel: float = 0.65
    """
    Confidence penalty for gray zone classifications with pixel fallback. 

    Pixel measurements are less reliable than homography, so penalty is more aggressive.

    Range: 0.5-0.75
    - Lower: More aggressive batch-context influence
    - Higher: More trust in pixel-based size

    Default: 0.65 (35% penalty)
    """

    penalty_for_roi_in_gray_zone: float = 0.3
    """
    Penalty for ROI quality when it is in the gray zone of raw area  
    """

    disambiguation_gray_zone_behavior: str = 'keep_original'
    """
    Behavior when raw_area falls in the gray zone (between small and regular thresholds).
    
    Gray Zone Range: [9000, 11000] pixels² (2000 px² wide)
    Frequency: ~15-20% of Brown_Orange_Family detections fall in this range
    
    Options:
    - 'keep_original': Keep YOLO's original prediction (RECOMMENDED for production)
        → Trusts classifier within ambiguous zone where it has seen features
        → Best when classifier has good accuracy on family classes
        
    - 'uncertain': Return "Uncertain" classification (CONSERVATIVE)
        → Admits ambiguity rather than risk misclassification
        → Use when cost of misclassification is high
        
    - 'prefer_small': Prefer small classification in gray zone (BIASED)
        → Defaults to Brown_Orange_Small when uncertain
        → Use when Small bags are more common in production
        
    - 'prefer_regular': Prefer regular/overlay classification in gray zone (BIASED)
        → Defaults to Brown_Orange_Overlay when uncertain
        → Use when Overlay bags are more common in production
    
    Rationale for 'keep_original':
      - Within gray zone, classifier has seen bag features (color, texture, logos)
      - Size alone is ambiguous, but visual features may still discriminate
      - Most gray zone cases (80%+) are correctly resolved by classifier
      - See docs/ROI_FILTERING_AND_THRESHOLDS.md for empirical analysis
    
    Default: 'keep_original' (production-recommended)
    """
    
    disambiguation_debug_logging: bool = _parse_bool_env("DISAMBIGUATION_DEBUG", True)
    """
    Enable detailed debug logging for disambiguation tuning.
    
    When True: Logs raw area, decision, and reasoning for each disambiguation
    
    Default: False
    """
    
    disambiguation_confidence_penalty: float = _parse_float_env("DISAMBIGUATION_CONFIDENCE_PENALTY", 0.8)
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
    # Part 1.1: Enhanced Disambiguation V2 (Production-Grade)
    # ============================================================================
    # These parameters control the enhanced V2 disambiguation module with
    # multi-threshold logic, validation, and detailed diagnostics.
    
    disambiguation_v2_enabled: bool = _parse_bool_env("DISAMBIGUATION_V2_ENABLED", True)
    """
    Enable enhanced V2 disambiguation module with production-grade features.
    
    When True: Uses V2 module with validation, multi-thresholds, and detailed logging
    When False: Falls back to V1 module (legacy behavior)
    
    V2 Features:
    - Multi-threshold size bins (very_small, small, gray_zone, regular, large)
    - Aspect ratio and area validation
    - Enhanced gray zone strategies (including 'use_confidence')
    - Detailed diagnostic metadata
    - Context-aware logging (track_id, frame_index)
    
    Default: True (V2 enabled)
    """
    
    # Validation Parameters
    disambiguation_v2_min_aspect_ratio: float = _parse_float_env("DISAMBIGUATION_V2_MIN_ASPECT_RATIO", 0.3)
    """
    Minimum acceptable aspect ratio (width/height) for bounding box validation.
    
    Detects unrealistically elongated or squished bboxes that may indicate
    detection artifacts or occlusion issues.
    
    Range: 0.2 - 0.5
    Default: 0.3 (width >= 30% of height)
    """
    
    disambiguation_v2_max_aspect_ratio: float = _parse_float_env("DISAMBIGUATION_V2_MAX_ASPECT_RATIO", 3.0)
    """
    Maximum acceptable aspect ratio (width/height) for bounding box validation.
    
    Range: 2.5 - 4.0
    Default: 3.0 (width <= 3x height)
    """
    
    disambiguation_v2_aspect_ratio_penalty: float = _parse_float_env("DISAMBIGUATION_V2_ASPECT_RATIO_PENALTY", 0.3)
    """
    Confidence penalty for bboxes with suspicious aspect ratios.
    
    Applied when aspect ratio is outside acceptable range but not degenerate.
    Range: 0.2 - 0.5
    Default: 0.3 (30% confidence reduction)
    """
    
    disambiguation_v2_min_realistic_area: float = _parse_float_env("DISAMBIGUATION_V2_MIN_REALISTIC_AREA", 1000.0)
    """
    Minimum realistic area (pixels²) for a bread bag bbox.
    
    Areas below this are penalized as unrealistically small (possible artifacts).
    Range: 500 - 2000
    Default: 1000.0 pixels²
    """
    
    disambiguation_v2_max_realistic_area: float = _parse_float_env("DISAMBIGUATION_V2_MAX_REALISTIC_AREA", 100000.0)
    """
    Maximum realistic area (pixels²) for a bread bag bbox.
    
    Areas above this are penalized as unrealistically large (possible merge artifacts).
    Range: 50000 - 200000
    Default: 100000.0 pixels²
    """
    
    disambiguation_v2_unrealistic_area_penalty: float = _parse_float_env("DISAMBIGUATION_V2_UNREALISTIC_AREA_PENALTY", 0.5)
    """
    Confidence penalty for unrealistic area sizes.
    
    Range: 0.3 - 0.7
    Default: 0.5 (50% confidence reduction)
    """
    
    # Multi-Threshold Size Bins
    disambiguation_v2_very_small_threshold: float = _parse_float_env("DISAMBIGUATION_V2_VERY_SMALL_THRESHOLD", 5000.0)
    """
    Threshold for 'very_small' bin (well below normal small bag sizes).
    
    Areas below this are confidently classified as Brown_Orange_Small.
    Range: 3000 - 7000
    Default: 5000.0 pixels²
    """
    
    disambiguation_v2_large_threshold: float = _parse_float_env("DISAMBIGUATION_V2_LARGE_THRESHOLD", 25000.0)
    """
    Threshold for 'large' bin (well above normal regular bag sizes).
    
    Areas above this are confidently classified as Brown_Orange_Overlay.
    Range: 20000 - 50000
    Default: 25000.0 pixels²
    """
    
    # Gray Zone Confidence Strategy
    disambiguation_v2_gray_zone_confidence_threshold: float = _parse_float_env("DISAMBIGUATION_V2_GRAY_ZONE_CONF_THRESHOLD", 0.6)
    """
    Confidence threshold for 'use_confidence' gray zone strategy.
    
    When gray_zone_behavior='use_confidence':
    - confidence >= threshold: Keep classifier prediction
    - confidence < threshold: Return "Uncertain"
    
    Range: 0.5 - 0.8
    Default: 0.6
    """
    
    disambiguation_v2_debug_logging: bool = _parse_bool_env("DISAMBIGUATION_V2_DEBUG", True)
    """
    Enable detailed debug logging for V2 disambiguation.
    
    Logs include:
    - Before/after labels and confidence
    - Area, aspect ratio, size bin
    - Validation results and penalties
    - Resolution reason with full context
    - Track and frame context
    
    Default: True (enable for initial deployment, disable after validation)
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
    
    trust_closed_max: float = _parse_float_env("TRUST_CLOSED_MAX", 1.0)
    """
    Maximum trust score for Closed ROIs.
    
    EQUAL TRUST DESIGN: Set to 1.0 (same as open) to let quality metrics determine
    trust, not the state. Previous value of 0.7 artificially biased evidence toward
    open ROIs. Closed ROIs are essential for disambiguation and should be weighted
    equally based on sharpness, brightness, and other quality factors.
    
    If closed ROIs have lower quality, they will naturally get lower trust through
    sharpness penalties, blur penalties, etc. The state itself should not impose
    a blanket cap.
    
    Range: 0.8 - 1.0
    
    Default: 1.0 (equal to open, changed from 0.7)
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
    
    trust_min_for_support: float = _parse_float_env("TRUST_MIN_FOR_SUPPORT", 0.35)
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
    
    evidence_top_k_rois: int = _parse_int_env("EVIDENCE_TOP_K_ROIS", 10)
    """
    Number of top ROIs (by trust) to use for evidence accumulation.
    
    Quality-first selection: pick best K ROIs regardless of total count.
    Uses stratified sampling to ensure minimum closed ROI representation.
    
    Increased from 7 to 10 to provide more evidence while maintaining quality
    through trust-based weighting.
    
    Range: 5 - 10
    
    Default: 10 (increased from 7)
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
    
    stability_margin_threshold: float = _parse_float_env("STABILITY_MARGIN_THRESHOLD", 0.25)
    """
    Minimum log-evidence margin between winner and runner-up.
    
    If margin < threshold, result is marked as "Uncertain".
    Range: 0.2 - 1.0
    
    TUNED FOR LOG-EVIDENCE: Log-evidence scores are negative (e.g., -3.5 vs -4.2),
    so margins are relatively small. A threshold of 0.3 provides good discrimination
    while avoiding excessive "Uncertain" classifications. Previous value of 0.5 was
    too strict, causing too many uncertain results.
    
    Typical margins:
    - Clear winner: margin > 0.5 (e.g., -2.0 vs -2.8)
    - Moderate confidence: margin 0.3-0.5 (e.g., -3.5 vs -4.0)
    - Ambiguous: margin < 0.3 (e.g., -4.0 vs -4.2) → Uncertain
    
    Default: 0.3 (reduced from 0.5)
    """
    
    stability_min_trusted_rois: int = _parse_int_env("STABILITY_MIN_TRUSTED_ROIS", 2)
    """
    Minimum number of trusted ROIs (trust >= trust_min_for_support) required.
    
    If fewer trusted ROIs, result is marked as "Uncertain".
    Range: 1 - 5
    
    Increased from 2 to 3 to ensure sufficient high-quality evidence before
    accepting a classification. With balanced ROI collection (10+10) and top-K
    of 10, requiring 3 trusted ROIs is achievable and provides better reliability.
    
    Default: 3 (increased from 2)
    """
    
    # ============================================================================
    # Velocity Stability Gate for ROI Collection
    # ============================================================================
    # These parameters implement a "Time-To-Live" (TTL) gate that ensures bags
    # have truly settled before collecting ROIs. This prevents blurry ROIs from
    # bags that are still vibrating or moving.
    
    velocity_stability_gate_enabled: bool = _parse_bool_env("VELOCITY_STABILITY_GATE_ENABLED", False)
    """
    Enable velocity stability gating for ROI collection.
    
    When True: ROIs are only collected after the bag has been stable
               (velocity below threshold) for a minimum duration.
    When False: ROIs are collected immediately regardless of velocity.
               Relies on classifier's "Rejected" class to filter unusable ROIs.
    
    V8 CHANGE: Default changed from True to False.
    
    Rationale: Strict velocity gates caused too few ROIs to be collected (2-3 per track
    instead of 8-15), leading to "Uncertain" classifications. By disabling velocity
    gates, we rely on:
    1. Essential quality checks (min size, aspect ratio, brightness, sharpness)
    2. Classifier's "Rejected" class to identify unusable ROIs
    3. Quality-based selection (keep best N ROIs by composite quality score)
    
    This approach collects more ROI candidates, giving the classifier sufficient
    evidence for confident decisions while letting it handle marginal cases.
    
    Set VELOCITY_STABILITY_GATE_ENABLED=true to restore strict behavior.
    
    Default: False (V8: relaxed from True for better ROI collection)
    """
    
    velocity_stability_threshold: float = _parse_float_env("VELOCITY_STABILITY_THRESHOLD", 0.25)
    """
    Maximum velocity (pixels per millisecond) to consider position "stable".
    
    If velocity > threshold, stability timer resets.
    If velocity < threshold, stability timer accumulates.
    
    Range: 0.05 - 0.5 px/ms
    - 0.05 px/ms = 50 px/s (very strict, bag nearly stationary)
    - 0.25 px/ms = 250 px/s (moderate, catches settling and slow rotation)
    - 0.5 px/ms = 500 px/s (lenient, bag moving slowly)
    
    Default: 0.25 (250 pixels per second) - increased from 0.15 to better catch spinning bags
    """
    
    velocity_stability_min_duration_ms: float = _parse_float_env("VELOCITY_STABILITY_MIN_DURATION_MS", 150.0)
    """
    Minimum time (milliseconds) the bag must remain stable before collecting ROIs.
    
    This ensures the bag has truly settled, not just paused for one frame.
    
    Range: 50 - 300 ms
    - 50ms: Less strict, catches brief pauses
    - 150ms: Moderate, ensures genuine stability
    - 300ms: Very strict, may miss some ROIs
    
    Default: 150.0 (150 milliseconds)
    """
    
    spin_detection_min_boxes: int = _parse_int_env("SPIN_DETECTION_MIN_BOXES", 5)
    """
    Minimum number of bounding boxes needed to detect spinning.
    
    Spinning is detected by analyzing the variance in bounding box aspect ratios
    over recent history. When a bag spins, its aspect ratio changes significantly
    as it rotates (e.g., narrow when viewed from side, wider from front).
    
    Range: 3 - 10
    - 3: Very quick detection, but may have false positives
    - 5: Balanced detection
    - 10: More confident detection, but delayed
    
    Default: 5
    """
    
    spin_detection_ar_variance_threshold: float = _parse_float_env("SPIN_DETECTION_AR_VARIANCE_THRESHOLD", 0.02)
    """
    Aspect ratio variance threshold to detect spinning.
    
    When the variance of aspect ratios over recent bounding boxes exceeds
    this threshold, the bag is considered to be spinning. Lower values are
    more sensitive to rotation.
    
    Range: 0.01 - 0.1
    - 0.01: Very sensitive, detects slight rotations
    - 0.02: Moderate sensitivity (default)
    - 0.1: Only detects significant spinning
    
    Default: 0.02
    """
    
    spin_detection_box_history_size: int = _parse_int_env("SPIN_DETECTION_BOX_HISTORY_SIZE", 15)
    """
    Maximum number of bounding boxes to keep in history for spin detection.
    
    Larger values allow detection of slower spinning but use more memory.
    Smaller values detect only rapid spinning.
    
    Range: 10 - 30
    - 10: Only recent history, detects rapid spinning
    - 15: Balanced (default)
    - 30: Longer history, detects slower spinning
    
    Default: 15
    """
    
    # ============================================================================
    # Bidirectional Context-Aware Classification Smoothing
    # ============================================================================
    # These parameters implement a buffered validation queue that uses both
    # previous and future context to correct low-confidence classifications.
    # This exploits the batch nature of production lines where bags of the
    # same type are processed sequentially.
    
    bidirectional_smoothing_enabled: bool = _parse_bool_env("BIDIRECTIONAL_SMOOTHING_ENABLED", True)
    """
    Enable bidirectional context-aware classification smoothing.
    
    When True: Classifications are buffered and validated using both
               previous and next items before final commit.
    When False: Classifications are committed immediately (legacy behavior).
    
    Default: True
    """
    
    bidirectional_buffer_size: int = _parse_int_env("BIDIRECTIONAL_BUFFER_SIZE", 7)
    """
    Size of the validation buffer for bidirectional smoothing.
    
    The center item (buffer_size // 2) is validated using context from
    both sides. Buffer must be odd for symmetric context.
    
    Range: 5 - 11 (odd numbers recommended)
    - 5: Context of 2 before + 2 after (faster commit, less context)
    - 7: Context of 3 before + 3 after (balanced)
    - 11: Context of 5 before + 5 after (more context, slower commit)
    
    Default: 7 (3 items before + center + 3 items after)
    """
    
    bidirectional_confidence_threshold: float = _parse_float_env("BIDIRECTIONAL_CONFIDENCE_THRESHOLD", 0.90)
    """
    Confidence threshold above which classifications bypass context checking.
    
    High-confidence classifications (>= threshold) are trusted and not
    overridden by context, even if context disagrees.
    
    Range: 0.80 - 0.95
    - 0.80: More items bypass context check (trust classifier more)
    - 0.90: Balanced (default)
    - 0.95: Most items use context check (trust context more)
    
    Default: 0.90
    """
    
    bidirectional_context_agreement_ratio: float = _parse_float_env("BIDIRECTIONAL_CONTEXT_AGREEMENT_RATIO", 0.8)
    """
    Minimum ratio of context items that must agree to override center item.
    
    For a buffer of 7 (3+1+3), with this set to 0.8:
    - Check prev_3 and next_3 (6 context items)
    - At least 5 of 6 (80%) must agree to override center
    
    Range: 0.6 - 1.0
    - 0.6: 60% agreement needed (more aggressive smoothing)
    - 0.8: 80% agreement needed (balanced)
    - 1.0: 100% agreement needed (only unanimous context overrides)
    
    Default: 0.8
    """
    
    bidirectional_batch_transition_protection: bool = _parse_bool_env("BIDIRECTIONAL_BATCH_TRANSITION_PROTECTION", True)
    """
    Protect batch transitions from being incorrectly smoothed.
    
    When True: If prev_context disagrees with next_context, this indicates
               a batch transition (e.g., Brown -> White). In this case,
               do NOT override the center item; trust the classifier.
    When False: Always apply smoothing if context ratio is met.
    
    Default: True
    """
    
    bidirectional_inactivity_timeout_ms: float = _parse_float_env("BIDIRECTIONAL_INACTIVITY_TIMEOUT_MS", 300_000.0)
    """
    Time in milliseconds after which buffered events are committed if no new events arrive.
    
    This ensures that events in the bidirectional buffer are committed even during
    periods of inactivity (no new bread bags detected), rather than waiting indefinitely
    for more events to fill the buffer.
    
    Range: 1000 - 30000 ms
    - 1000ms: Quick flush, less context available
    - 5000ms: Balanced, waits 5 seconds for more bags
    - 30000ms: Very patient, waits up to 30 seconds
    
    Default: 5000.0 (5 seconds)
    """
    
    bidirectional_uncertain_override_ratio: float = _parse_float_env("BIDIRECTIONAL_UNCERTAIN_OVERRIDE_RATIO", 0.5)
    """
    Context agreement ratio for overriding Uncertain/Unknown labels (relaxed threshold).
    
    This parameter applies specifically to "Uncertain" and "Unknown" classifications,
    which are treated more aggressively than regular labels. Unlike regular labels that
    require 80% agreement (bidirectional_context_agreement_ratio), Uncertain/Unknown
    labels only require majority vote (50% by default) to be overridden by context.
    
    Rationale:
    - Uncertain/Unknown labels indicate classifier uncertainty, making them good
      candidates for context-based inference
    - Lower threshold allows override even with partial agreement
    - 50% (majority vote) means if more context items agree on a label than disagree,
      that label wins
    
    Special Handling for Uncertain/Unknown:
    - Always check context (skip high-confidence bypass)
    - Filter Uncertain/Unknown from context when computing agreement
    - Skip batch transition protection (allow override at transitions)
    - Mark overridden labels with confidence_tier='low' and uncertain_override=True
    
    Range: 0.4 - 0.8
    - 0.4: Very aggressive override (40% agreement needed)
    - 0.5: Majority vote (balanced, default)
    - 0.6: 60% agreement needed
    - 0.8: Same as regular threshold (conservative)
    
    Environment: BIDIRECTIONAL_UNCERTAIN_OVERRIDE_RATIO=0.5
    
    Default: 0.5 (majority vote)
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
    
    # ==========================================================================
    # V6 Performance & Reliability Optimization Parameters
    # ==========================================================================
    # These parameters implement the Event-Centric Bread Counting Pipeline
    # optimizations for production-grade reliability.
    
    # --------------------------------------------------------------------------
    # Temporal Decimation (Skip Redundant Monitor Updates)
    # --------------------------------------------------------------------------
    # Key insight: Detection must run every frame. Matching does not.
    
    temporal_decimation_enabled: bool = _parse_bool_env("TEMPORAL_DECIMATION_ENABLED", True)
    """
    Enable temporal decimation to skip redundant monitor updates.
    
    When True: Skip monitor update when:
    - Bounding box area change < epsilon
    - Centroid shift < delta
    - Confidence unchanged
    
    Benefits:
    - Significant CPU cost reduction (30-50%)
    - Preserves correctness (detection still runs every frame)
    - Only skips redundant state updates
    
    Default: True
    """
    
    temporal_decimation_area_epsilon: float = _parse_float_env("TEMPORAL_DECIMATION_AREA_EPSILON", 0.05)
    """
    Area change threshold for temporal decimation.
    
    Skip monitor update if |new_area - last_area| / last_area < epsilon
    
    Range: 0.02 - 0.10
    - Lower values: More sensitive, fewer skips
    - Higher values: Less sensitive, more skips
    
    Default: 0.05 (5% area change threshold)
    """
    
    temporal_decimation_centroid_delta_px: float = _parse_float_env("TEMPORAL_DECIMATION_CENTROID_DELTA", 5.0)
    """
    Centroid shift threshold (pixels) for temporal decimation.
    
    Skip monitor update if centroid_distance < delta
    
    Range: 2.0 - 15.0 pixels
    - Lower values: More sensitive to movement
    - Higher values: More tolerant of small movements
    
    Default: 5.0 pixels
    """
    
    temporal_decimation_confidence_epsilon: float = _parse_float_env("TEMPORAL_DECIMATION_CONF_EPSILON", 0.05)
    """
    Confidence change threshold for temporal decimation.
    
    Skip monitor update if |new_conf - last_conf| < epsilon
    
    Range: 0.02 - 0.10
    - Lower values: More sensitive to confidence changes
    - Higher values: More tolerant of confidence fluctuations
    
    Default: 0.05 (5% confidence change threshold)
    """
    
    temporal_decimation_max_skip_frames: int = _parse_int_env("TEMPORAL_DECIMATION_MAX_SKIP", 3)
    """
    Maximum consecutive frames to skip before forcing an update.
    
    Range: 1 - 5 frames
    Ensures events are updated periodically even if changes are minimal.
    
    Default: 3 frames
    """

    # --------------------------------------------------------------------------
    # Multi-Stage Matching Early Rejection
    # --------------------------------------------------------------------------
    # Order matching gates for cheap rejection before expensive IOU computation
    # Matching Pipeline: Ghost timeout → Centroid → Area ratio → IOU
    
    early_rejection_enabled: bool = _parse_bool_env("EARLY_REJECTION_ENABLED", True)
    """
    Enable early rejection gates before IOU computation.
    
    Matching pipeline order:
    1. Ghost timeout check (instant rejection)
    2. Centroid distance gate (cheap)
    3. Area ratio gate (cheap)
    4. IOU computation (expensive - only if above pass)
    
    Benefits:
    - Most candidates rejected cheaply
    - IOU only runs on tiny subset
    - Significant CPU reduction
    
    Default: True
    """
    
    early_rejection_area_ratio_min: float = _parse_float_env("EARLY_REJECTION_AREA_RATIO_MIN", 0.4)
    """
    Minimum area ratio for early rejection.
    
    Reject if min(area1, area2) / max(area1, area2) < threshold
    
    Range: 0.3 - 0.7
    - Lower values: More lenient, allows more size variation
    - Higher values: Stricter, requires similar sizes
    
    Default: 0.4 (allow up to 2.5x size difference)
    """
    
    early_rejection_area_ratio_max: float = _parse_float_env("EARLY_REJECTION_AREA_RATIO_MAX", 2.5)
    """
    Maximum area ratio for early rejection.
    
    Reject if max(area1, area2) / min(area1, area2) > threshold
    
    Range: 1.5 - 3.0
    - Lower values: Stricter size matching
    - Higher values: More lenient
    
    Default: 2.5 (allow up to 2.5x size difference)
    """

    # --------------------------------------------------------------------------
    # Spatial Zones (Explicit Zone Definitions)
    # --------------------------------------------------------------------------
    # Predictable regions for event lifecycle management
    
    spatial_zones_enabled: bool = _parse_bool_env("SPATIAL_ZONES_ENABLED", True)
    """
    Enable explicit spatial zone definitions.
    
    Zones:
    - ENTRY_ZONE: Where new events can be created
    - ACTIVE_ZONE: Where events participate in matching
    - EXIT_ZONE: Where events are finalized
    
    Benefits:
    - Reduces IOU comparisons drastically
    - Prevents tracking irrelevant history
    - Predictable event lifecycle
    
    Default: True
    """
    
    entry_zone_margin_px: int = _parse_int_env("ENTRY_ZONE_MARGIN_PX", 50)
    """
    Margin from frame edges for entry zone constraint.
    
    Events can only be created if centroid is at least this far from edges.
    Set to 0 to allow events anywhere.
    
    Range: 0 - 100 pixels
    Default: 50 pixels
    """
    
    exit_zone_margin_px: int = _parse_int_env("EXIT_ZONE_MARGIN_PX", 80)
    """
    Margin from edges defining the exit zone.
    
    Events with centroid within this margin from edges are candidates
    for faster finalization.
    
    Range: 50 - 150 pixels  
    Default: 80 pixels
    """

    # --------------------------------------------------------------------------
    # Retention Safety
    # --------------------------------------------------------------------------
    # Ensure retention never deletes unprocessed data
    
    retention_safety_enabled: bool = _parse_bool_env("RETENTION_SAFETY_ENABLED", True)
    """
    Enable retention safety rule.
    
    When True: Retention must respect processor progress
    Rule: segment.frame_index >= last_committed_index
    
    Benefits:
    - Never deletes unprocessed data
    - Prevents data loss under load
    - Production-safe operation
    
    Default: True
    """
    
    # ==========================================================================
    # V4 Performance Optimization Parameters
    # ==========================================================================
    
    # === Phase 1: Detection Queue (Decouple Detection from Monitor) ===
    detection_queue_enabled: bool = _parse_bool_env("DETECTION_QUEUE_ENABLED", True)
    """
    Enable detection results queue to decouple detection from monitor processing.
    
    When True: Detection runs at full BPU speed without blocking on monitor
    When False: Detection and monitor run sequentially (legacy behavior)
    
    Benefits:
    - Detection runs at full BPU speed without blocking
    - Natural backpressure handling via queue
    - Better CPU/BPU utilization through parallel execution
    - Immediate ~30% performance gain
    
    Default: True (V4 optimization)
    """
    
    detection_queue_size: int = _parse_int_env("DETECTION_QUEUE_SIZE", 10)
    """
    Maximum size of detection results queue.
    
    Range: 5 - 30
    - Lower values: Less memory, faster backpressure response
    - Higher values: More buffering, handles burst loads better
    
    Default: 10 (balance between memory and throughput)
    """
    
    detection_queue_warning_threshold: float = _parse_float_env("DETECTION_QUEUE_WARNING_THRESHOLD", 0.7)
    """
    Queue utilization threshold (0.0-1.0) to trigger warnings.
    
    When queue exceeds this threshold, warnings are logged to indicate
    monitor processing is falling behind detection.
    
    Range: 0.5 - 0.9
    Default: 0.7 (70% full)
    """
    
    # === Phase 2: Batch Inference (CRITICAL for 2.5x speedup) ===
    detection_batch_enabled: bool = _parse_bool_env("DETECTION_BATCH_ENABLED", False)
    """
    Enable BPU batch inference for detection.
    
    When True: Accumulate frames and process in batches (2-4 frames)
    When False: Single-frame processing (legacy behavior)
    
    Benefits:
    - 40-60% speedup over single-frame processing
    - YOLOv8n achieves 220 FPS with batching vs 140 FPS single-frame
    - BPU can process 2-4 frames simultaneously with minimal overhead
    
    Default: True (V4 critical optimization)
    """
    
    detection_batch_size: int = _parse_int_env("DETECTION_BATCH_SIZE", 2)
    """
    Number of frames to process in each batch.
    
    Range: 2 - 4
    - 2: Conservative, ~40% speedup, minimal latency impact
    - 3: Balanced, ~50% speedup
    - 4: Aggressive, ~60% speedup, may increase latency
    
    Tuning: Start with 2, increase to 4 if latency is acceptable.
    
    Default: 2 (conservative start, tune to 4 for production)
    """
    
    detection_batch_timeout_ms: float = _parse_float_env("DETECTION_BATCH_TIMEOUT_MS", 5.0)
    """
    Maximum time (milliseconds) to wait for batch to fill.
    
    Prevents latency spikes from waiting for full batch when frame rate drops.
    If batch doesn't fill within timeout, process partial batch.
    
    Range: 2.0 - 10.0 ms
    - Lower values: More responsive, may reduce batch efficiency
    - Higher values: More efficient batching, may increase latency
    
    Default: 5.0 ms (balance between latency and throughput)
    """
    
    # === Phase 3: Monitor Processing Optimization ===
    lazy_roi_cropping_enabled: bool = _parse_bool_env("LAZY_ROI_CROPPING_ENABLED", True)
    """
    Enable lazy ROI cropping for memory and CPU efficiency.
    
    When True: Store only metadata during tracking, crop on-demand for classification
    When False: Crop and store ROIs immediately (legacy behavior)
    
    Benefits:
    - Reduces memory bandwidth (no immediate cropping)
    - Reduces CPU overhead (only crop what's needed)
    - Events that expire never trigger cropping
    - 30-50% reduction in monitor processing time
    
    Default: True (V4 optimization)
    """
    
    vectorized_iou_enabled: bool = _parse_bool_env("VECTORIZED_IOU_ENABLED", True)
    """
    Enable vectorized IoU calculations for event association.
    
    When True: Use numpy vectorized operations for IoU computation
    When False: Use loop-based IoU calculation (legacy behavior)
    
    Benefits:
    - Replaces O(n*m) loops with O(1) vectorized ops
    - 2-3x faster association for multiple events
    - Better performance with many active events
    
    Default: True (V4 optimization)
    """
    
    # === Phase 4: Classification Batch Processing (LOW PRIORITY) ===
    classification_batch_enabled: bool = _parse_bool_env("CLASSIFICATION_BATCH_ENABLED", False)
    """
    Enable batch processing for classification (FUTURE OPTIMIZATION).
    
    When True: Group ROIs from multiple events for batch classification
    When False: Classify each event individually (current behavior)
    
    Note: Lower priority as classification already runs async.
    Only enable if classification becomes bottleneck after Phases 1-3.
    
    Default: False (not implemented yet)
    """
    
    classification_batch_size: int = _parse_int_env("CLASSIFICATION_BATCH_SIZE", 4)
    """
    Number of events to batch for classification.
    
    Range: 2 - 8
    Only used when classification_batch_enabled=True.
    
    Default: 4
    """
    
    # ============================================================================
    # Homography-Based Size Classification
    # ============================================================================
    # These parameters enable accurate size-based classification using the work
    # table as a reference plane for perspective transformation.
    
    homography_enabled: bool = _parse_bool_env("HOMOGRAPHY_ENABLED", True)
    """
    Enable homography-based size estimation for physically accurate measurements.
    
    When True: Use homography transformation to convert pixel measurements to
               real-world centimeter measurements using the table as reference.
    When False: Use raw pixel area for size estimation (current behavior).
    
    Benefits:
    - Perspective-invariant: Works at any camera distance/angle
    - Physically accurate: Measures actual bread size in cm², not pixels
    - Debuggable: Real measurements vs arbitrary pixels
    
    Requires: homography_table_corners and homography_table_size_cm to be set.
    
    Default: False (requires calibration)
    """
    
    homography_table_corners: str = _parse_str_env(
        "HOMOGRAPHY_TABLE_CORNERS", 
        "[[401.0, 292.0], [913.0, 297.0], [1062.0, 564.0], [354.0, 578.0]]"  # Empty means not calibrated
    )
    """
    Table corner positions in pixel coordinates for homography calibration.
    
    Format: JSON array of 4 corners in clockwise order starting from top-left.
    Example: "[[150,100],[950,120],[980,650],[120,680]]"
    
    The corners define the work table boundaries in the camera view.
    Used to compute the perspective transformation matrix.
    
    Default: "" (not calibrated)
    """
    
    homography_table_width_cm: float = _parse_float_env("HOMOGRAPHY_TABLE_WIDTH_CM", 200.0)
    """
    Physical width of the work table in centimeters.
    
    Range: 40.0 - 200.0 (typical work tables)
    
    Default: 80.0 cm
    """
    
    homography_table_height_cm: float = _parse_float_env("HOMOGRAPHY_TABLE_HEIGHT_CM", 100.0)
    """
    Physical height (depth) of the work table in centimeters.
    
    Range: 30.0 - 150.0 (typical work tables)
    
    Default: 60.0 cm
    """
    
    homography_small_threshold_cm2: float = _parse_float_env("HOMOGRAPHY_SMALL_THRESHOLD_CM2", 5500.0)
    """
    Area threshold (cm²) below which a bag is classified as "Small".
    
    When homography is enabled, this threshold is used instead of pixel-based
    disambiguation_small_threshold.
    
    Range: 50.0 - 200.0 cm²
    Tuning: Measure actual Small bag dimensions and compute area.
    
    Default: 100.0 cm² (approximately 10cm x 10cm)
    """
    
    homography_large_threshold_cm2: float = _parse_float_env("HOMOGRAPHY_LARGE_THRESHOLD_CM2", 7500.0)
    """
    Area threshold (cm²) above which a bag is classified as "Large/Regular".
    
    When homography is enabled, this threshold is used instead of pixel-based
    disambiguation_regular_threshold.
    
    Range: 100.0 - 300.0 cm²
    Tuning: Measure actual Large bag dimensions and compute area.
    
    Default: 150.0 cm² (approximately 12cm x 12cm)
    """
    
    # ============================================================================
    # ROI Candidate Saving (Debug/Analysis)
    # ============================================================================
    # These parameters control saving all ROI candidates with metadata for
    # post-analysis, model improvement, and debugging.
    
    save_roi_candidates: bool = _parse_bool_env("SAVE_ROI_CANDIDATES", False)
    """
    Enable/disable saving all ROI candidates with metadata.
    
    When True: Save all ROI candidates per track with quality metrics and metadata
    When False: Do not save ROI candidates (production behavior)
    
    Useful for:
    - Debugging "Uncertain" or "Rejected" classifications
    - Analyzing ROI quality metrics
    - Collecting real production data for model retraining
    - Quality verification and monitoring
    
    Default: False (enable for debug/analysis)
    """
    
    roi_candidates_dir: str = _parse_str_env("ROI_CANDIDATES_DIR", "data/roi_candidates")
    """
    Directory for saved ROI candidates.
    
    ROIs are organized by classification:
        {roi_candidates_dir}/
        ├── Brown_Orange_Small/
        │   ├── track_12345_roi_0_quality_0.85.jpg
        │   └── track_12345_metadata.json
        ├── Rejected/
        └── Uncertain/
    
    Default: "data/roi_candidates"
    """
    
    save_rejected_tracks: bool = _parse_bool_env("SAVE_REJECTED_TRACKS", True)
    """
    Save ROI candidates for tracks classified as "Rejected".
    
    Useful for analyzing why the classifier rejects certain inputs.
    
    Default: True (save for analysis)
    """
    
    save_uncertain_tracks: bool = _parse_bool_env("SAVE_UNCERTAIN_TRACKS", True)
    """
    Save ROI candidates for tracks classified as "Uncertain" or "Unknown".
    
    Useful for understanding classification failures and edge cases.
    
    Default: True (save for analysis)
    """
    
    max_rois_per_track_save: int = _parse_int_env("MAX_ROIS_PER_TRACK_SAVE", 20)
    """
    Maximum number of ROI candidates to save per track.
    
    Limits disk usage while still capturing sufficient data for analysis.
    
    Range: 5 - 50
    Default: 20
    """
    
    # ==========================================================================
    # V10/V11 Unified Spool Configuration
    # ==========================================================================
    # These parameters control both spool_processor_node and spool_recorder_node.
    # All spool-related configuration is centralized here for consistency.
    
    # --------------------------------------------------------------------------
    # Common Spool Settings (used by both processor and recorder)
    # --------------------------------------------------------------------------
    
    spool_dir: str = _parse_str_env(
        "SPOOL_DIR",
        "/home/sunrise/BreadCounting/data/spool" if IS_WINDOWS is False else "data/spool"
    )
    """
    Directory for spool segment files.
    
    Both spool_processor and spool_recorder use this directory for:
    - Storing recorded H.264 segment files
    - Reading segments for playback
    - Processor state file persistence
    
    Environment: SPOOL_DIR=/path/to/spool
    Default: /home/sunrise/BreadCounting/data/spool (RDK) or data/spool (Windows)
    """
    
    # --------------------------------------------------------------------------
    # Spool Recorder Configuration
    # --------------------------------------------------------------------------
    
    spool_segment_duration: float = _parse_float_env("SPOOL_SEGMENT_DURATION", 5.0)
    """
    Target segment duration in seconds for spool recorder.
    
    The recorder rotates to a new segment file after approximately this duration.
    Actual duration may vary slightly due to IDR frame alignment.
    
    Range: 1.0 - 30.0 seconds
    Default: 5.0 seconds
    """
    
    spool_max_segment_duration: float = _parse_float_env("SPOOL_MAX_SEGMENT_DURATION", 10.0)
    """
    Maximum segment duration in seconds before forced rotation.
    
    If no IDR frame arrives within this time, the recorder forces segment rotation.
    Should be larger than spool_segment_duration.
    
    Range: 5.0 - 60.0 seconds
    Default: 10.0 seconds
    """
    
    spool_retention_seconds: float = _parse_float_env("SPOOL_RETENTION_SECONDS", 180.0)
    """
    Maximum age of spool segments before automatic deletion.
    
    The retention policy deletes segments older than this threshold.
    Should be long enough to handle processing delays but not waste disk space.
    
    Range: 60 - 3600 seconds (1 minute to 1 hour)
    Default: 180.0 seconds (3 minutes)
    """
    
    spool_recorder_queue_size: int = _parse_int_env("SPOOL_RECORDER_QUEUE_SIZE", 100)
    """
    Maximum size of the recorder's internal frame queue.
    
    Frames are buffered here before being written to disk.
    Larger values handle burst loads better but use more memory.
    
    Range: 50 - 500
    Default: 100
    """
    
    spool_recorder_stats_interval: float = _parse_float_env("SPOOL_RECORDER_STATS_INTERVAL", 10.0)
    """
    Interval in seconds between recorder statistics log messages.
    
    Range: 5.0 - 60.0 seconds
    Default: 10.0 seconds
    """
    
    # --------------------------------------------------------------------------
    # Spool Processor Configuration
    # --------------------------------------------------------------------------
    
    spool_processor_target_fps: float = _parse_float_env("SPOOL_PROCESSOR_TARGET_FPS", 20.0)
    """
    Target frames per second for spool processor publishing.
    
    In ACK-free mode, the processor publishes at this rate.
    Adaptive pacing may adjust this based on spool lag.
    
    Range: 10.0 - 60.0 FPS
    Default: 20.0 FPS
    """
    
    spool_processor_poll_interval: float = _parse_float_env("SPOOL_PROCESSOR_POLL_INTERVAL", 1.0)
    """
    Interval in seconds to poll for new segments when spool is empty.
    
    Range: 0.1 - 5.0 seconds
    Default: 1.0 seconds
    """
    
    spool_processor_stats_interval: float = _parse_float_env("SPOOL_PROCESSOR_STATS_INTERVAL", 10.0)
    """
    Interval in seconds between processor statistics log messages.
    
    Range: 5.0 - 60.0 seconds
    Default: 10.0 seconds
    """
    
    spool_processor_prepend_sps_pps: bool = _parse_bool_env("SPOOL_PROCESSOR_PREPEND_SPS_PPS", True)
    """
    Prepend cached SPS/PPS to first frame of each segment.
    
    This ensures decoder can initialize properly at segment boundaries.
    
    Default: True
    """
    
    spool_processor_enable_adaptive_pacing: bool = _parse_bool_env("SPOOL_PROCESSOR_ENABLE_ADAPTIVE_PACING", True)
    """
    Enable adaptive FPS adjustment based on spool lag.
    
    When True: FPS adjusts automatically (relaxed when healthy, faster when behind)
    When False: Fixed FPS mode
    
    Default: True
    """
    
    spool_processor_adaptive_fps_min: float = _parse_float_env("SPOOL_PROCESSOR_ADAPTIVE_FPS_MIN", 20.0)
    """
    Minimum FPS for adaptive pacing (floor value).
    
    Range: 10.0 - 30.0 FPS
    Default: 20.0 FPS
    """
    
    spool_processor_adaptive_fps_relaxed: float = _parse_float_env("SPOOL_PROCESSOR_ADAPTIVE_FPS_RELAXED", 15.0)
    """
    Relaxed FPS when spool lag is healthy (conserve resources).
    
    Range: 10.0 - 25.0 FPS
    Default: 15.0 FPS
    """
    
    spool_processor_adaptive_fps_max: float = _parse_float_env("SPOOL_PROCESSOR_ADAPTIVE_FPS_MAX", 25.0)
    """
    Maximum FPS for adaptive pacing (catching up).
    
    Range: 20.0 - 60.0 FPS
    Default: 25.0 FPS
    """
    
    spool_processor_min_frame_interval_ms: float = _parse_float_env("SPOOL_PROCESSOR_MIN_FRAME_INTERVAL_MS", 5.0)
    """
    Minimum interval between frames in milliseconds.
    
    Prevents the processor from publishing too fast and overwhelming the consumer.
    
    Range: 10.0 - 100.0 ms
    Default: 25.0 ms (40 FPS max)
    """
    
    spool_processor_delete_processed_segments: bool = _parse_bool_env("SPOOL_PROCESSOR_DELETE_PROCESSED_SEGMENTS", True)
    """
    Delete segments immediately after processing to save disk space.
    
    When True: Segments are deleted right after all frames are published
    When False: Rely on retention policy for cleanup
    
    Default: True
    """
    
    spool_processor_watchdog_timeout: float = _parse_float_env("SPOOL_PROCESSOR_WATCHDOG_TIMEOUT", 30.0)
    """
    Timeout in seconds without publishing before alerting (watchdog).
    
    Range: 10.0 - 120.0 seconds
    Default: 30.0 seconds
    """
    
    spool_lag_warn_threshold: int = _parse_int_env("SPOOL_LAG_WARN_THRESHOLD", 5)
    """
    Spool lag threshold (in segments) to trigger warning.
    
    Range: 3 - 20 segments
    Default: 5 segments
    """
    
    spool_lag_error_threshold: int = _parse_int_env("SPOOL_LAG_ERROR_THRESHOLD", 10)
    """
    Spool lag threshold (in segments) to trigger error.
    
    Range: 5 - 50 segments
    Default: 10 segments
    """
    
    spool_lag_healthy_threshold: int = _parse_int_env("SPOOL_LAG_HEALTHY_THRESHOLD", 5)
    """
    Spool lag below which system is considered healthy (relax FPS).
    
    Range: 1 - 10 segments
    Default: 5 segments
    """
    
    spool_lag_normal_threshold: int = _parse_int_env("SPOOL_LAG_NORMAL_THRESHOLD", 10)
    """
    Spool lag threshold between healthy and high lag states.
    
    Range: 5 - 30 segments
    Default: 10 segments
    """
    
    spool_processor_enable_perf_logging: bool = _parse_bool_env("SPOOL_PROCESSOR_ENABLE_PERF_LOGGING", False)
    """
    Enable performance profiling logs for spool processor.
    
    Logs timing metrics for list_segments, get_next_frame, publish_frame.
    
    Default: False
    """
    
    spool_processor_perf_log_interval_sec: float = _parse_float_env("SPOOL_PROCESSOR_PERF_LOG_INTERVAL_SEC", 2.0)
    """
    Interval in seconds between performance metric logs.
    
    Only used when spool_processor_enable_perf_logging is True.
    
    Range: 1.0 - 30.0 seconds
    Default: 2.0 seconds
    """
    
    spool_processor_segment_list_cache_interval: float = _parse_float_env("SPOOL_PROCESSOR_SEGMENT_LIST_CACHE_INTERVAL", 1.0)
    """
    Interval in seconds to cache segment list for performance.
    
    Reduces disk I/O by caching the segment list.
    
    Range: 0.5 - 5.0 seconds
    Default: 1.0 seconds
    """
    
    spool_processor_enable_crc32_logging: bool = _parse_bool_env("SPOOL_PROCESSOR_ENABLE_CRC32_LOGGING", False)
    """
    Include CRC32 checksums in frame publish logs for debugging.
    
    Useful for data integrity verification but adds CPU overhead.
    
    Default: False
    """
    
    # ==========================================================================
    # V11 Spool-Aware Degraded Mode Configuration
    # ==========================================================================
    # These parameters control the spool-aware degraded mode that leverages
    # disk-spooled segments instead of aggressively skipping frames during
    # temporary overload.
    
    spool_aware_degraded_mode_enabled: bool = _parse_bool_env("SPOOL_AWARE_DEGRADED_MODE_ENABLED", True)
    """
    Enable spool-aware degraded mode that benefits from disk-spooled segments.
    
    When True: System leverages spooled segments on disk during overload,
               only skipping frames if we are far behind (> spool_lag_threshold_seconds)
    When False: Use legacy smart skipping that skips frames during queue pressure
    
    Benefits:
    - No frame loss during temporary overload (frames are on disk)
    - Processing catches up naturally when load decreases
    - Only skip if truly far behind to prevent unbounded growth
    
    Environment: SPOOL_AWARE_DEGRADED_MODE_ENABLED=true
    Default: True
    """
    
    spool_lag_threshold_seconds: float = _parse_float_env("SPOOL_LAG_THRESHOLD_SECONDS", 300.0)
    """
    Maximum acceptable lag (in seconds) before triggering frame skipping.
    
    When spool_aware_degraded_mode_enabled is True:
    - If spool lag < this threshold: Don't skip frames, let system catch up
    - If spool lag >= this threshold: Trigger smart skipping to prevent unbounded growth
    
    The lag is calculated as: (current_segment_being_recorded - current_segment_being_processed) * segment_duration
    
    Range: 60 - 600 seconds (1-10 minutes)
    - 60s: Aggressive, skip quickly if behind
    - 300s: Balanced, 5 minute buffer (default)
    - 600s: Conservative, allow up to 10 minutes of lag
    
    Environment: SPOOL_LAG_THRESHOLD_SECONDS=300.0
    Default: 300.0 (5 minutes)
    """
    
    spool_segment_duration_seconds: float = _parse_float_env("SPOOL_SEGMENT_DURATION_SECONDS", 5.0)
    """
    Average duration of a spool segment in seconds.
    
    Used to convert segment count lag to time-based lag:
        lag_seconds = segment_lag * spool_segment_duration_seconds
    
    This should match the actual segment duration configured in the spool recorder.
    
    Range: 1.0 - 30.0 seconds
    Default: 5.0 seconds (typical segment duration)
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
        
        # Anti-double-counting suppression - frame-based
        suppression_distance_px=tracking_config.suppression_distance_px,
        suppression_duration_ms=tracking_config.suppression_duration_ms,
        suppression_duration_frames=tracking_config.suppression_duration_frames,
        suppression_require_box_overlap=tracking_config.suppression_require_box_overlap,
        suppression_iou_threshold=tracking_config.suppression_iou_threshold,
        # Size-adaptive suppression (Issue #4 fix)
        suppression_use_adaptive_distance=tracking_config.suppression_use_adaptive_distance,
        suppression_diagonal_multiplier=tracking_config.suppression_diagonal_multiplier,
        suppression_min_distance_px=tracking_config.suppression_min_distance_px,
        suppression_max_distance_px=tracking_config.suppression_max_distance_px,
        
        # Temporal cooldown for new event creation - frame-based with ms migration
        min_event_creation_interval_ms=tracking_config.min_event_creation_interval_ms,
        temporal_cooldown_frames=tracking_config.temporal_cooldown_frames,
        temporal_cooldown_distance_px=tracking_config.temporal_cooldown_distance_px,
        
        # Active event spatial exclusion
        active_event_exclusion_distance_px=tracking_config.active_event_exclusion_distance_px,
        active_event_exclusion_iou=tracking_config.active_event_exclusion_iou,
        
        # Detection clustering
        detection_cluster_distance_px=tracking_config.detection_cluster_distance_px,
        
        # State transition timing - frame-based with new min_open_duration_ms
        min_open_duration_ms=tracking_config.min_open_duration_ms,
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

        # Disambiguate
        disambiguation_small_threshold=tracking_config.disambiguation_small_threshold,
        disambiguation_regular_threshold=tracking_config.disambiguation_regular_threshold,
        penalty_for_roi_in_gray_zone=tracking_config.penalty_for_roi_in_gray_zone,
        disambiguation_gray_zone_penalty_homography=tracking_config.disambiguation_gray_zone_penalty_homography,
        disambiguation_gray_zone_penalty_pixel=tracking_config.disambiguation_gray_zone_penalty_pixel,
        
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
        
        # V6 Performance & Reliability Optimization Parameters
        # Adaptive Ghost Timeout
        adaptive_ghost_timeout_enabled=tracking_config.adaptive_ghost_timeout_enabled,
        adaptive_ghost_velocity_factor=tracking_config.adaptive_ghost_velocity_factor,
        adaptive_ghost_min_timeout_frames=tracking_config.adaptive_ghost_min_timeout_frames,
        adaptive_ghost_max_timeout_frames=tracking_config.adaptive_ghost_max_timeout_frames,
        
        # Temporal Decimation
        temporal_decimation_enabled=tracking_config.temporal_decimation_enabled,
        temporal_decimation_area_epsilon=tracking_config.temporal_decimation_area_epsilon,
        temporal_decimation_centroid_delta_px=tracking_config.temporal_decimation_centroid_delta_px,
        temporal_decimation_confidence_epsilon=tracking_config.temporal_decimation_confidence_epsilon,
        temporal_decimation_max_skip_frames=tracking_config.temporal_decimation_max_skip_frames,
        
        # Multi-Stage Matching Early Rejection
        early_rejection_enabled=tracking_config.early_rejection_enabled,
        early_rejection_area_ratio_min=tracking_config.early_rejection_area_ratio_min,
        early_rejection_area_ratio_max=tracking_config.early_rejection_area_ratio_max,
        
        # Spatial Zones
        spatial_zones_enabled=tracking_config.spatial_zones_enabled,
        entry_zone_margin_px=tracking_config.entry_zone_margin_px,
        exit_zone_margin_px=tracking_config.exit_zone_margin_px,
        
        # Retention Safety
        retention_safety_enabled=tracking_config.retention_safety_enabled,
        
        # Velocity Stability Gate for ROI Collection
        velocity_stability_gate_enabled=tracking_config.velocity_stability_gate_enabled,
        velocity_stability_threshold=tracking_config.velocity_stability_threshold,
        velocity_stability_min_duration_ms=tracking_config.velocity_stability_min_duration_ms,
        
        # Spin Detection for ROI Collection
        spin_detection_min_boxes=tracking_config.spin_detection_min_boxes,
        spin_detection_ar_variance_threshold=tracking_config.spin_detection_ar_variance_threshold,
        spin_detection_box_history_size=tracking_config.spin_detection_box_history_size,
    )
