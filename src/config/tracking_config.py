"""
Centralized configuration for detection and tracking parameters.

This module contains all tunable parameters for the bag detection and tracking system.
Adjust these values to tune the system's sensitivity and behavior.

V3 Performance Optimization Notes:
- min_roi_size reduced to 100 (from 300) to avoid blocking the pipeline
- min_roi_sharpness reduced to 300 (from 400) to accept more samples
- Parameters tuned for 25fps throughput at 720p resolution
"""

from dataclasses import dataclass


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
    lockout_window: int = 20  # V3: Reduced from 25 for faster event recovery


    # ============================================================================
    # State Transition Thresholds
    # ============================================================================
    
    min_open_frames: int = 4  # V3: Reduced from 5 for faster state transitions
    """
    Minimum consecutive frames a bag must be detected as "open" before allowing 
    transition to "closed" state.
    
    Range: 1 - 20
    - Lower values: Faster state transitions, more responsive
    - Higher values: More stable, reduces noise-induced transitions
    
    Default: 4 (V3: reduced from 5)
    """
    
    min_closed_frames: int = 2  # V3: Reduced from 3 for faster counting
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
    
    expiry_detecting_open: int = 8  # V3: Reduced from 10 for faster cleanup
    """
    Frames without update before expiring an event in 'detecting_open' state.
    
    Range: 5 - 30
    - Lower values: Faster cleanup of lost tracks
    - Higher values: More persistent tracking through occlusions
    
    Default: 8 (V3: reduced from 10)
    """
    
    expiry_detecting_closed: int = 8  # V3: Reduced from 10 for faster cleanup
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
    
    min_roi_sharpness: float = 300  # V3: Reduced from 400 for more accepted samples
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


# Global configuration instance
tracking_config = TrackingConfig()
