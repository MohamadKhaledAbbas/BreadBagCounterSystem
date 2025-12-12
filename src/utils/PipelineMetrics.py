"""
Pipeline Metrics Module for BreadBag Counter System.

Provides comprehensive monitoring and KPI tracking for the detection,
classification, and counting pipeline to achieve 99.9% accuracy.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
import statistics

from src.utils.AppLogging import logger


@dataclass
class DetectionMetrics:
    """Metrics for detection stage."""
    total_detections: int = 0
    open_detections: int = 0
    closed_detections: int = 0
    low_confidence_filtered: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    processing_time_ms: float = 0.0


@dataclass 
class EventMetrics:
    """Metrics for event lifecycle."""
    events_created: int = 0
    events_counted: int = 0
    events_expired: int = 0
    events_suppressed: int = 0
    avg_open_hits: float = 0.0
    avg_closed_hits: float = 0.0


@dataclass
class ClassificationMetrics:
    """Metrics for classification stage."""
    total_classified: int = 0
    unknown_count: int = 0
    low_confidence_count: int = 0
    voting_used: int = 0
    confidence_only_used: int = 0
    avg_confidence: float = 0.0
    avg_candidates_per_event: float = 0.0


@dataclass
class QualityMetrics:
    """ROI quality metrics."""
    rois_accepted: int = 0
    rois_rejected_size: int = 0
    rois_rejected_sharpness: int = 0
    rois_rejected_brightness: int = 0
    avg_sharpness: float = 0.0


class PipelineMetrics:
    """
    Centralized metrics tracking for the BreadBag Counter pipeline.
    
    Tracks KPIs for detection, ROI quality, classification, and event management.
    Provides real-time insights for achieving 99.9% accuracy target.
    """
    
    # Target KPIs for 99.9% accuracy
    # NOTE: TARGET_DETECTION_CONFIDENCE (0.7) is intentionally higher than the 
    # min_conf_threshold (0.4) in tracking_config.py. The min_conf_threshold is a
    # permissive filter to avoid missing bags, while TARGET_DETECTION_CONFIDENCE
    # represents the desired average confidence for high-quality detection.
    # Warnings are triggered when average confidence drops below target, indicating
    # potential model degradation or environmental issues.
    TARGET_DETECTION_CONFIDENCE = 0.7
    TARGET_CLASSIFICATION_CONFIDENCE = 0.5
    TARGET_EVENT_COMPLETION_RATE = 0.95  # Events that complete vs expire
    TARGET_ROI_ACCEPTANCE_RATE = 0.8
    
    def __init__(self, log_interval_seconds: float = 30.0):
        self.log_interval = log_interval_seconds
        self.last_log_time = time.perf_counter()
        self._lock = threading.Lock()
        
        # Running metrics
        self.detection = DetectionMetrics()
        self.events = EventMetrics()
        self.classification = ClassificationMetrics()
        self.quality = QualityMetrics()
        
        # Sliding window for recent performance (last 100 events)
        self._recent_detection_confs: deque = deque(maxlen=100)
        self._recent_classification_confs: deque = deque(maxlen=100)
        self._recent_processing_times: deque = deque(maxlen=100)
        self._recent_sharpness_values: deque = deque(maxlen=100)
        
        # Anomaly detection
        self._consecutive_low_conf_detections = 0
        self._consecutive_unknown_classifications = 0
        
        logger.info(
            f"[PipelineMetrics] Initialized with log_interval={log_interval_seconds}s, "
            f"targets: det_conf>={self.TARGET_DETECTION_CONFIDENCE}, "
            f"cls_conf>={self.TARGET_CLASSIFICATION_CONFIDENCE}"
        )
    
    def record_detection(self, detections: List[Dict[str, Any]], 
                        processing_time_ms: float,
                        open_cls_id: int, closed_cls_id: int):
        """Record detection stage metrics."""
        with self._lock:
            self.detection.processing_time_ms = processing_time_ms
            
            for det in detections:
                conf = det.get('conf', 0.0)
                cls_id = det.get('class_id')
                
                self.detection.total_detections += 1
                self._recent_detection_confs.append(conf)
                
                if cls_id == open_cls_id:
                    self.detection.open_detections += 1
                elif cls_id == closed_cls_id:
                    self.detection.closed_detections += 1
                
                self.detection.min_confidence = min(self.detection.min_confidence, conf)
                self.detection.max_confidence = max(self.detection.max_confidence, conf)
                
                # Track low confidence
                if conf < self.TARGET_DETECTION_CONFIDENCE:
                    self._consecutive_low_conf_detections += 1
                else:
                    self._consecutive_low_conf_detections = 0
            
            # Update average
            if self._recent_detection_confs:
                self.detection.avg_confidence = statistics.mean(self._recent_detection_confs)
            
            self._recent_processing_times.append(processing_time_ms)
            
            # Anomaly alert
            if self._consecutive_low_conf_detections >= 10:
                logger.warning(
                    f"[PipelineMetrics] ANOMALY: {self._consecutive_low_conf_detections} "
                    f"consecutive low-confidence detections (< {self.TARGET_DETECTION_CONFIDENCE})"
                )
    
    def record_detection_filtered(self, reason: str = "low_confidence"):
        """Record when a detection is filtered out."""
        with self._lock:
            if reason == "low_confidence":
                self.detection.low_confidence_filtered += 1
    
    def record_event_created(self):
        """Record new event creation."""
        with self._lock:
            self.events.events_created += 1
    
    def record_event_counted(self, open_hits: int, closed_hits: int):
        """Record event that reached counted state."""
        with self._lock:
            self.events.events_counted += 1
            
            # Update running averages
            total = self.events.events_counted
            self.events.avg_open_hits = (
                (self.events.avg_open_hits * (total - 1) + open_hits) / total
            )
            self.events.avg_closed_hits = (
                (self.events.avg_closed_hits * (total - 1) + closed_hits) / total
            )
    
    def record_event_expired(self, state: str):
        """Record event expiration."""
        with self._lock:
            self.events.events_expired += 1
            
            # Check completion rate
            total_events = self.events.events_created
            if total_events > 0:
                completion_rate = self.events.events_counted / total_events
                if completion_rate < self.TARGET_EVENT_COMPLETION_RATE and total_events >= 10:
                    logger.warning(
                        f"[PipelineMetrics] Low event completion rate: "
                        f"{completion_rate:.2%} (target: {self.TARGET_EVENT_COMPLETION_RATE:.0%})"
                    )
    
    def record_event_suppressed(self):
        """Record when event creation is suppressed."""
        with self._lock:
            self.events.events_suppressed += 1
    
    def record_classification(self, label: str, confidence: float, 
                             candidates_count: int, used_voting: bool):
        """Record classification result."""
        with self._lock:
            self.classification.total_classified += 1
            self._recent_classification_confs.append(confidence)
            
            if label == "Unknown":
                self.classification.unknown_count += 1
                self._consecutive_unknown_classifications += 1
            else:
                self._consecutive_unknown_classifications = 0
            
            if confidence < self.TARGET_CLASSIFICATION_CONFIDENCE:
                self.classification.low_confidence_count += 1
            
            if used_voting:
                self.classification.voting_used += 1
            else:
                self.classification.confidence_only_used += 1
            
            # Update averages
            if self._recent_classification_confs:
                self.classification.avg_confidence = statistics.mean(
                    self._recent_classification_confs
                )
            
            total = self.classification.total_classified
            self.classification.avg_candidates_per_event = (
                (self.classification.avg_candidates_per_event * (total - 1) + candidates_count) / total
            )
            
            # Anomaly alert
            if self._consecutive_unknown_classifications >= 5:
                logger.warning(
                    f"[PipelineMetrics] ANOMALY: {self._consecutive_unknown_classifications} "
                    f"consecutive Unknown classifications"
                )
    
    def record_roi_quality(self, accepted: bool, sharpness: float, 
                          reject_reason: Optional[str] = None):
        """Record ROI quality check result."""
        with self._lock:
            if accepted:
                self.quality.rois_accepted += 1
                self._recent_sharpness_values.append(sharpness)
                if self._recent_sharpness_values:
                    self.quality.avg_sharpness = statistics.mean(self._recent_sharpness_values)
            else:
                if reject_reason == "size":
                    self.quality.rois_rejected_size += 1
                elif reject_reason == "sharpness":
                    self.quality.rois_rejected_sharpness += 1
                elif reject_reason == "brightness":
                    self.quality.rois_rejected_brightness += 1
    
    def maybe_log_summary(self, force: bool = False):
        """Log metrics summary if interval has passed."""
        current_time = time.perf_counter()
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return
        
        with self._lock:
            self.last_log_time = current_time
            
            # Calculate derived KPIs
            total_rois = (self.quality.rois_accepted + 
                         self.quality.rois_rejected_size + 
                         self.quality.rois_rejected_sharpness +
                         self.quality.rois_rejected_brightness)
            roi_acceptance_rate = (
                self.quality.rois_accepted / total_rois if total_rois > 0 else 0.0
            )
            
            unknown_rate = (
                self.classification.unknown_count / self.classification.total_classified
                if self.classification.total_classified > 0 else 0.0
            )
            
            event_completion_rate = (
                self.events.events_counted / self.events.events_created
                if self.events.events_created > 0 else 0.0
            )
            
            # Log summary
            logger.info(
                f"[PipelineMetrics] === PIPELINE SUMMARY ===\n"
                f"  Detection: total={self.detection.total_detections}, "
                f"open={self.detection.open_detections}, "
                f"closed={self.detection.closed_detections}, "
                f"avg_conf={self.detection.avg_confidence:.3f}\n"
                f"  Events: created={self.events.events_created}, "
                f"counted={self.events.events_counted}, "
                f"expired={self.events.events_expired}, "
                f"suppressed={self.events.events_suppressed}, "
                f"completion_rate={event_completion_rate:.1%}\n"
                f"  Classification: total={self.classification.total_classified}, "
                f"unknown={self.classification.unknown_count} ({unknown_rate:.1%}), "
                f"avg_conf={self.classification.avg_confidence:.3f}\n"
                f"  ROI Quality: accepted={self.quality.rois_accepted}, "
                f"rejected_size={self.quality.rois_rejected_size}, "
                f"rejected_sharp={self.quality.rois_rejected_sharpness}, "
                f"acceptance_rate={roi_acceptance_rate:.1%}"
            )
            
            # KPI Alerts
            if event_completion_rate < self.TARGET_EVENT_COMPLETION_RATE and self.events.events_created >= 10:
                logger.warning(
                    f"[PipelineMetrics] KPI ALERT: Event completion rate "
                    f"{event_completion_rate:.1%} < target {self.TARGET_EVENT_COMPLETION_RATE:.0%}"
                )
            
            if roi_acceptance_rate < self.TARGET_ROI_ACCEPTANCE_RATE and total_rois >= 10:
                logger.warning(
                    f"[PipelineMetrics] KPI ALERT: ROI acceptance rate "
                    f"{roi_acceptance_rate:.1%} < target {self.TARGET_ROI_ACCEPTANCE_RATE:.0%}"
                )
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Return metrics as dictionary for external monitoring."""
        with self._lock:
            total_rois = (self.quality.rois_accepted + 
                         self.quality.rois_rejected_size + 
                         self.quality.rois_rejected_sharpness +
                         self.quality.rois_rejected_brightness)
            
            return {
                "detection": {
                    "total": self.detection.total_detections,
                    "open": self.detection.open_detections,
                    "closed": self.detection.closed_detections,
                    "avg_confidence": self.detection.avg_confidence,
                    "low_confidence_filtered": self.detection.low_confidence_filtered,
                },
                "events": {
                    "created": self.events.events_created,
                    "counted": self.events.events_counted,
                    "expired": self.events.events_expired,
                    "suppressed": self.events.events_suppressed,
                    "completion_rate": (
                        self.events.events_counted / self.events.events_created
                        if self.events.events_created > 0 else 0.0
                    ),
                },
                "classification": {
                    "total": self.classification.total_classified,
                    "unknown": self.classification.unknown_count,
                    "unknown_rate": (
                        self.classification.unknown_count / self.classification.total_classified
                        if self.classification.total_classified > 0 else 0.0
                    ),
                    "avg_confidence": self.classification.avg_confidence,
                },
                "quality": {
                    "rois_accepted": self.quality.rois_accepted,
                    "rois_rejected": total_rois - self.quality.rois_accepted,
                    "acceptance_rate": (
                        self.quality.rois_accepted / total_rois if total_rois > 0 else 0.0
                    ),
                    "avg_sharpness": self.quality.avg_sharpness,
                },
            }


# Global metrics instance
pipeline_metrics = PipelineMetrics()
