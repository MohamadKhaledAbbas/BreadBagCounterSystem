"""
Event-Centric State Monitor for BreadBag Counter System.

This module provides a drop-in replacement for BagStateMonitor that uses
the new event-centric tracking system.

KEY DIFFERENCES FROM BagStateMonitor:
1. Uses centroid distance instead of IoU for association
2. Uses millisecond-based timing instead of frame counts
3. Counting occurs after exit timeout, not at closure moment
4. Events survive detection loss through ghost state
5. State transitions require temporal stability

Usage:
    Replace BagStateMonitor with EventCentricStateMonitor in BagCounterApp
    
    monitor = EventCentricStateMonitor(open_cls_id=1, closed_cls_id=0)
    ready_events = monitor.update(detections, frame_dict)
"""

import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from src.tracking.EventCentricTracker import (
    EventCentricTracker, 
    EventConfig, 
    EventState
)
from src.config.tracking_config import tracking_config, get_event_config
from src.utils.AppLogging import logger, structured_logger
from src.utils.PipelineMetrics import pipeline_metrics


class EventCentricStateMonitor:
    """
    Event-centric state monitor for bread bag tracking and counting.
    
    This is a drop-in replacement for BagStateMonitor that implements:
    - Centroid-based association (no IoU)
    - Millisecond-based timing (no frame counts)
    - Ghost event handling for detection gaps
    - Exit-boundary-based counting
    
    The interface is compatible with BagStateMonitor.update() for seamless integration.
    """
    
    def __init__(self, 
                 open_cls_id: int, 
                 closed_cls_id: int,
                 config: Optional[EventConfig] = None,
                 fps: float = 25.0):
        """
        Initialize the event-centric state monitor.
        
        Args:
            open_cls_id: Class ID for open bag detections
            closed_cls_id: Class ID for closed bag detections
            config: EventConfig instance (creates from tracking_config if None)
            fps: Video frame rate for timestamp calculation
        """
        self.open_id = open_cls_id
        self.closed_id = closed_cls_id
        self.fps = fps
        self.frame_duration_ms = 1000.0 / fps

        # Get configuration
        if config is None:
            config = get_event_config()

        self.use_frame_timestamps = config.use_frame_timestamps
        
        # Create the event-centric tracker
        self.tracker = EventCentricTracker(
            config=config,
            open_class_id=open_cls_id,
            closed_class_id=closed_cls_id
        )
        
        # Track start time for timestamp calculation
        self.start_time_ms = time.perf_counter() * 1000
        
        # Statistics
        self.total_events_created = 0
        self.total_events_counted = 0
        self.total_events_expired = 0
        self.total_events_suppressed = 0
        
        # For visualization compatibility
        self.active_events = []  # List of active event info
        
        logger.info(
            f"[EventCentricStateMonitor] Initialized: "
            f"open_id={open_cls_id}, closed_id={closed_cls_id}, "
            f"fps={fps}, frame_duration={self.frame_duration_ms:.1f}ms, "
            f"use_frame_timestamps={config.use_frame_timestamps}"
        )
    
    def update(self, 
               detections: List[Dict[str, Any]], 
               frame_dict: Dict[str, Any]) -> List[Tuple[int, List, Any, Dict]]:
        """
        Update the monitor with new detections.
        
        This method provides interface compatibility with BagStateMonitor.update().
        
        Args:
            detections: List of detection dicts with keys:
                - box: bounding box [x1, y1, x2, y2]
                - class_id: detection class ID
                - conf: confidence score
            frame_dict: Dictionary with keys:
                - frame_count: current frame number
                - frame: the frame image (numpy array)
                
        Returns:
            List of tuples: (event_id, candidates, box, stats)
            Compatible with BagStateMonitor output format.
        """
        frame_count = frame_dict['frame_count']
        frame_img = frame_dict['frame']
        
        # Calculate timestamp
        if self.use_frame_timestamps:
            # Deterministic frame-based timing (recommended for offline/testing)
            # This ensures consistent timing regardless of processing speed
            current_time_ms = frame_count * self.frame_duration_ms
        else:
            # Wall clock time for production (real-time processing)
            current_time_ms = time.perf_counter() * 1000 - self.start_time_ms
        
        # Update the event-centric tracker
        ready_events = self.tracker.update(
            detections=detections,
            timestamp_ms=current_time_ms,
            frame_img=frame_img,
            frame_index=frame_count
        )
        
        # Update statistics from tracker
        tracker_stats = self.tracker.get_tracker_stats()
        self.total_events_created = tracker_stats['events_created']
        self.total_events_counted = tracker_stats['events_committed']
        self.total_events_expired = tracker_stats['events_expired']
        self.total_events_suppressed = tracker_stats['events_suppressed']
        
        # Update active events for visualization
        self._update_active_events()
        
        # Convert ready events to BagStateMonitor output format
        result = []
        for event_data in ready_events:
            event_id = event_data['event_id']
            candidates = event_data['candidates']
            box = event_data['box']
            stats = event_data['stats']
            
            # Log the commit for metrics
            pipeline_metrics.record_event_counted(
                stats.get('open_hits', 0),
                stats.get('closed_hits', 0)
            )
            
            result.append((event_id, candidates, box, stats))
        
        return result
    
    def _update_active_events(self):
        """Update active_events list for visualization compatibility."""
        events_info = self.tracker.get_active_events_info()
        
        # Convert to format expected by Visualizer
        self.active_events = []
        for info in events_info:
            # Create a minimal event object for visualization
            event_proxy = _EventProxy(
                id=info['id'],
                box=info['box'],
                state=info['state'],
                open_hits=info['open_count'],
                closed_hits=info['closed_count'],
                centroid=info.get('centroid'),
                roi_count=info.get('roi_count', 0),
            )
            self.active_events.append(event_proxy)
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """Return overall monitor statistics for monitoring."""
        tracker_stats = self.tracker.get_tracker_stats()
        
        return {
            "total_events_created": tracker_stats['events_created'],
            "total_events_counted": tracker_stats['events_committed'],
            "total_events_expired": tracker_stats['events_expired'],
            "total_events_suppressed": tracker_stats['events_suppressed'],
            "active_events": tracker_stats['active_events'],
            "recently_counted": tracker_stats['recently_committed'],
            "completion_rate": tracker_stats['completion_rate'],
            "total_detections_processed": tracker_stats['total_detections_processed'],
        }


class _EventProxy:
    """
    Minimal event proxy for visualization compatibility.
    
    Provides the attributes expected by Visualizer.render_all().
    """
    
    def __init__(self, id: int, box: tuple, state: str, open_hits: int, closed_hits: int,
                 centroid: tuple = None, roi_count: int = 0):
        self.id = id
        self.box = box
        self.state = state  # Keep original state name for proper visualization
        self.open_hits = open_hits
        self.closed_hits = closed_hits
        self.frames_since_update = 0
        self.last_centroid = centroid
        self.roi_count = roi_count


def create_event_centric_monitor(open_cls_id: int, closed_cls_id: int) -> EventCentricStateMonitor:
    """
    Factory function to create an EventCentricStateMonitor.
    
    Uses configuration from tracking_config.
    
    Args:
        open_cls_id: Class ID for open bag detections
        closed_cls_id: Class ID for closed bag detections
        
    Returns:
        Configured EventCentricStateMonitor instance
    """
    config = get_event_config()
    return EventCentricStateMonitor(
        open_cls_id=open_cls_id,
        closed_cls_id=closed_cls_id,
        config=config
    )
