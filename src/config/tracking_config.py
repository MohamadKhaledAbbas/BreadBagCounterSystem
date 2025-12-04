"""
Centralized configuration for detection and tracking parameters.

This module contains all tunable parameters for the bag detection and tracking system.
Adjust these values to tune the system's sensitivity and behavior.
"""

from dataclasses import dataclass


@dataclass
class TrackingConfig:
    """
    Configuration for bag detection and tracking system.
    
    These parameters control how the system detects, tracks, and classifies bread bags.
    Adjust these values to tune sensitivity and accuracy.
    """
    
    # ============================================================================
    # IoU Matching Parameters
    # ============================================================================
    
    iou_threshold: float = 0.35
    """
    IoU (Intersection over Union) threshold for matching detections to existing events.
    
    Range: 0.0 - 1.0
    - Lower values (e.g., 0.3): More lenient matching, good for fast-moving bags
    - Higher values (e.g., 0.5): Stricter matching, reduces false associations
    
    Default: 0.35 (relaxed from 0.45 to improve detection sensitivity)
    """
    
    # ============================================================================
    # State Transition Thresholds
    # ============================================================================
    
    min_open_frames: int = 5
    """
    Minimum consecutive frames a bag must be detected as "open" before allowing 
    transition to "closed" state.
    
    Range: 1 - 20
    - Lower values: Faster state transitions, more responsive
    - Higher values: More stable, reduces noise-induced transitions
    
    Default: 5
    """
    
    min_closed_frames: int = 2
    """
    Minimum consecutive frames a bag must be detected as "closed" to trigger 
    classification and counting.
    
    Range: 1 - 10
    - Lower values: Faster counting, may catch partial closures
    - Higher values: More reliable closed detection, may miss quick closures
    
    Default: 2
    """
    
    # ============================================================================
    # Detection Confidence Thresholds
    # ============================================================================
    
    min_conf_threshold: float = 0.2
    """
    Minimum confidence score for creating new tracking events from detections.
    
    Range: 0.0 - 1.0
    - Lower values (e.g., 0.1): Catch more potential bags, more false positives
    - Higher values (e.g., 0.4): Only high-confidence detections, may miss some bags
    
    Default: 0.2 (relaxed from 0.3 to catch more detections)
    """
    
    # ============================================================================
    # Event Management
    # ============================================================================
    
    max_active_events: int = 50
    """
    Maximum number of concurrent tracking events to prevent memory issues.
    
    Range: 10 - 100
    If this limit is reached, new detections will be ignored until events expire.
    
    Default: 50
    """
    
    # ============================================================================
    # State-Aware Expiry Timeouts
    # ============================================================================
    
    expiry_detecting_open: int = 12
    """
    Frames without update before expiring an event in 'detecting_open' state.
    
    Range: 5 - 30
    - Lower values: Faster cleanup of lost tracks
    - Higher values: More persistent tracking through occlusions
    
    Default: 12 (relaxed from 8 to handle temporary occlusions better)
    """
    
    expiry_detecting_closed: int = 18
    """
    Frames without update before expiring an event in 'detecting_closed' state.
    
    Range: 10 - 30
    - Lower values: Faster cleanup
    - Higher values: Give more time for closed detection to stabilize
    
    Default: 18 (relaxed from 15 to allow more time for closed detection)
    """
    
    expiry_counted: int = 5
    """
    Frames without update before expiring an event in 'counted' state.
    
    Range: 3 - 15
    Events in counted state are ready for cleanup and should expire quickly.
    
    Default: 5
    """
    
    # ============================================================================
    # ROI Collection Parameters (BagEvent)
    # ============================================================================
    
    max_open_samples: int = 8
    """
    Maximum number of ROI samples to collect during the 'open' phase.
    
    Range: 4 - 15
    More samples provide better classification but use more memory.
    
    Default: 8
    """
    
    max_closed_samples: int = 4
    """
    Maximum number of ROI samples to collect during the 'closed' phase.
    
    Range: 2 - 10
    More samples provide better classification but use more memory.
    
    Default: 4
    """
    
    # ============================================================================
    # ROI Quality Validation
    # ============================================================================
    
    min_roi_size: int = 80
    """
    Minimum width/height (in pixels) for a valid ROI.
    
    Range: 50 - 200
    ROIs smaller than this are rejected as too small for reliable classification.
    
    Default: 80
    """
    
    min_roi_sharpness: float = 30.0
    """
    Minimum sharpness score (Laplacian variance) for a valid ROI.
    
    Range: 10 - 100
    - Lower values: Accept more blurry images, more samples
    - Higher values: Only sharp images, fewer samples but better quality
    
    Default: 30.0 (relaxed from 50.0 to accept more samples)
    """


# Global configuration instance
tracking_config = TrackingConfig()
