"""
Enhanced Logging Module for BreadBag Counter System V2.

Provides:
- Colored console output for development
- Rotating file-based logging for production debugging
- Structured JSON logging for log analysis and pattern detection
- Context-rich logging with event correlation
- Performance metrics logging

Log files are written to data/logs/ directory:
- app.log: Human-readable logs with rotation
- app.json.log: Structured JSON logs for analysis
"""

import logging
import logging.handlers
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps


# ============================================================================
# Configuration
# ============================================================================

# Log directory - ensure it exists
LOG_DIR = os.environ.get("LOG_DIR", "data/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Read environment log level
log_level_name = os.environ.get("LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL = getattr(logging, log_level_name, logging.DEBUG)

# File logging configuration
LOG_FILE_MAX_BYTES = int(os.environ.get("LOG_FILE_MAX_BYTES", 50 * 1024 * 1024))  # 50MB default
LOG_FILE_BACKUP_COUNT = int(os.environ.get("LOG_FILE_BACKUP_COUNT", 25))  # Keep 10 backup files
ENABLE_JSON_LOGGING = os.environ.get("ENABLE_JSON_LOGGING", "true").lower() == "true"


# ============================================================================
# Custom Formatters
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""

    COLORS = {
        'ERROR': '\033[91m',      # Red
        'WARNING': '\033[93m',    # Yellow
        'INFO': '\033[92m',       # Green
        'DEBUG': '\033[96m',      # Cyan
        'CRITICAL': '\033[95m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        result = super().format(record)
        record.levelname = levelname  # restore
        return result


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter for log analysis and pattern detection.
    
    Outputs logs in JSON format with:
    - timestamp in ISO format
    - log level
    - logger name
    - message
    - extra context fields
    - exception info if present
    """
    
    def format(self, record: logging.LogRecord) -> str:
        from datetime import timezone
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present (for structured logging)
        if hasattr(record, 'extra_data') and record.extra_data:
            log_entry["data"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add any custom attributes from the record
        for key in ['event_id', 'track_id', 'frame_id', 'component', 'duration_ms']:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        
        return json.dumps(log_entry, default=str)


class FileFormatter(logging.Formatter):
    """Standard file formatter with full timestamp and thread info."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


# ============================================================================
# Logger Setup
# ============================================================================

# Console formatter
console_formatter = ColoredFormatter(
    fmt='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
root_logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Rotating file handler for human-readable logs
file_handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "app.log"),
    maxBytes=LOG_FILE_MAX_BYTES,
    backupCount=LOG_FILE_BACKUP_COUNT,
    encoding='utf-8'
)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(FileFormatter())
root_logger.addHandler(file_handler)

# JSON file handler for structured logs (for log analysis)
json_file_handler = None
if ENABLE_JSON_LOGGING:
    json_file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "app.json.log"),
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    json_file_handler.setLevel(LOG_LEVEL)
    json_file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(json_file_handler)

# Application logger
logger = logging.getLogger("BreadCounter")
logger.setLevel(LOG_LEVEL)

# Quiet third-party components
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("rclpy").setLevel(logging.WARNING)


# ============================================================================
# Structured Logging Helpers
# ============================================================================

class StructuredLogger:
    """
    Enhanced logger with structured logging support for easier debugging
    and pattern detection.
    
    Usage:
        structured_logger.event_created(event_id=123, confidence=0.95, box=[10, 20, 100, 200])
        structured_logger.classification_result(track_id=456, label="Bran", confidence=0.87)
    """
    
    def __init__(self, base_logger: logging.Logger):
        self._logger = base_logger
    
    def _log_structured(self, level: int, message: str, component: str, **kwargs):
        """Internal method to log with structured data."""
        record = self._logger.makeRecord(
            self._logger.name, level, "", 0, message, (), None
        )
        record.extra_data = kwargs
        record.component = component
        self._logger.handle(record)
    
    def event_created(self, event_id: int, confidence: float, box: list, frame_index: int = 0, **kwargs):
        """Log event creation with structured data."""
        self._log_structured(
            logging.INFO,
            f"[EVENT_CREATED] id={event_id}, conf={confidence:.3f}, frame={frame_index}",
            component="BagStateMonitor",
            event_id=event_id,
            confidence=confidence,
            box=box,
            frame_index=frame_index,
            **kwargs
        )
    
    def event_counted(self, event_id: int, label: str, confidence: float, 
                      open_hits: int, closed_hits: int, total_frames: int = 0, 
                      track_duration: int = 0, **kwargs):
        """Log successful event counting with structured data."""
        msg = (
            f"[EVENT_COUNTED] id={event_id}, label={label}, conf={confidence:.3f}, "
            f"open_hits={open_hits}, closed_hits={closed_hits}, frames={total_frames}"
        )
        self._log_structured(
            logging.INFO,
            msg,
            component="BagStateMonitor",
            event_id=event_id,
            label=label,
            confidence=confidence,
            open_hits=open_hits,
            closed_hits=closed_hits,
            total_frames=total_frames,
            track_duration=track_duration,
            **kwargs
        )
    
    def event_expired(self, event_id: int, state: str, frames_tracked: int, 
                     open_hits: int = 0, closed_hits: int = 0, 
                     frames_since_update: int = 0, **kwargs):
        """Log event expiration with structured data."""
        # Use WARNING only for events that had significant progress (likely under-counting)
        # Use DEBUG for events that naturally expired without much progress
        has_progress = (open_hits >= 3 or closed_hits >= 2)
        level = logging.WARNING if has_progress else logging.DEBUG
        
        msg = (
            f"[EVENT_EXPIRED] id={event_id}, state={state}, frames={frames_tracked}, "
            f"open_hits={open_hits}, closed_hits={closed_hits}, idle={frames_since_update}"
        )
        self._log_structured(
            level,
            msg,
            component="BagStateMonitor",
            event_id=event_id,
            state=state,
            frames_tracked=frames_tracked,
            open_hits=open_hits,
            closed_hits=closed_hits,
            frames_since_update=frames_since_update,
            **kwargs
        )
    
    def classification_result(self, track_id: int, label: str, confidence: float,
                             candidates: int, used_voting: bool, 
                             rejection_reason: str = None, evidence_scores: dict = None,
                             winner_ratio: float = None, **kwargs):
        """Log classification result with structured data."""
        msg_parts = [f"[CLASSIFICATION] track={track_id}, label={label}, conf={confidence:.3f}"]
        if rejection_reason:
            if kwargs["has_previous_label_reused"]:
                msg_parts.append(f"reused_reason={rejection_reason}")
            else:
                msg_parts.append(f"reason={rejection_reason}")
        if winner_ratio is not None:
            msg_parts.append(f"ratio={winner_ratio:.2f}")
        
        level = logging.WARNING if label == "Unknown" else logging.INFO
        
        self._log_structured(
            level,
            ", ".join(msg_parts),
            component="ClassifierService",
            track_id=track_id,
            label=label,
            confidence=confidence,
            candidates=candidates,
            used_voting=used_voting,
            rejection_reason=rejection_reason,
            evidence_scores=evidence_scores,
            winner_ratio=winner_ratio,
            **kwargs
        )
    
    def detection_frame(self, frame_id: int, open_count: int, closed_count: int,
                        processing_time_ms: float, **kwargs):
        """Log detection results for a frame."""
        self._log_structured(
            logging.DEBUG,
            f"[DETECTION] frame={frame_id}, open={open_count}, closed={closed_count}, time={processing_time_ms:.1f}ms",
            component="Detector",
            frame_id=frame_id,
            open_count=open_count,
            closed_count=closed_count,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
    
    def roi_quality(self, accepted: bool, sharpness: float, reject_reason: Optional[str] = None, **kwargs):
        """Log ROI quality check result."""
        status = "ACCEPTED" if accepted else f"REJECTED({reject_reason})"
        self._log_structured(
            logging.DEBUG,
            f"[ROI_QUALITY] {status}, sharpness={sharpness:.1f}",
            component="BagEvent",
            accepted=accepted,
            sharpness=sharpness,
            reject_reason=reject_reason,
            **kwargs
        )
    
    def anomaly_detected(self, anomaly_type: str, details: str, **kwargs):
        """Log anomaly detection."""
        self._log_structured(
            logging.WARNING,
            f"[ANOMALY] type={anomaly_type}, details={details}",
            component="PipelineMetrics",
            anomaly_type=anomaly_type,
            details=details,
            **kwargs
        )
    
    def pipeline_summary(self, metrics: Dict[str, Any]):
        """Log pipeline metrics summary."""
        self._log_structured(
            logging.INFO,
            f"[PIPELINE_SUMMARY] {json.dumps(metrics, default=str)}",
            component="PipelineMetrics",
            **metrics
        )
    
    def health_check(self, status: str, details: Dict[str, Any]):
        """Log health check status."""
        self._log_structured(
            logging.INFO if status == "healthy" else logging.WARNING,
            f"[HEALTH_CHECK] status={status}",
            component="HealthCheck",
            status=status,
            **details
        )
    
    def event_state_transition(self, event_id: int, old_state: str, new_state: str, 
                               trigger: str, **kwargs):
        """Log event state transitions for pipeline flow tracking."""
        msg = f"[STATE_TRANSITION] id={event_id}, {old_state} -> {new_state}, trigger={trigger}"
        self._log_structured(
            logging.INFO,
            msg,
            component="BagStateMonitor",
            event_id=event_id,
            old_state=old_state,
            new_state=new_state,
            trigger=trigger,
            **kwargs
        )
    
    def event_suppressed(self, event_id: int, reason: str, iou: float = 0.0, 
                        conflicting_event_id: int = None, **kwargs):
        """Log event suppression with detailed context."""
        msg = (
            f"[EVENT_SUPPRESSED] new_id={event_id}, reason={reason}, iou={iou:.2f}, "
            f"conflict_with={conflicting_event_id}"
        )
        self._log_structured(
            logging.INFO,
            msg,
            component="BagStateMonitor",
            event_id=event_id,
            reason=reason,
            iou=iou,
            conflicting_event_id=conflicting_event_id,
            **kwargs
        )
    
    def event_forced_close(self, event_id: int, state: str, reason: str, **kwargs):
        """Log forced event closure (stuck event fail-safe)."""
        msg = f"[EVENT_FORCED_CLOSE] id={event_id}, state={state}, reason={reason}"
        self._log_structured(
            logging.WARNING,
            msg,
            component="EventCentricTracker",
            event_id=event_id,
            state=state,
            forced_close_reason=reason,
            **kwargs
        )
    
    def roi_added(self, event_id: int, is_open: bool, sharpness: float, 
                  frame_index: int, confidence: float, total_rois: int, **kwargs):
        """Log ROI addition with quality metrics."""
        roi_type = "OPEN" if is_open else "CLOSED"
        msg = (
            f"[ROI_ADDED] event={event_id}, type={roi_type}, sharpness={sharpness:.1f}, "
            f"frame={frame_index}, conf={confidence:.2f}, total={total_rois}"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="BagEvent",
            event_id=event_id,
            roi_type=roi_type,
            is_open=is_open,
            sharpness=sharpness,
            frame_index=frame_index,
            confidence=confidence,
            total_rois=total_rois,
            **kwargs
        )
    
    def roi_rejected(self, event_id: int, reason: str, sharpness: float = 0.0, 
                    dimensions: tuple = None, brightness: float = 0.0, **kwargs):
        """Log ROI rejection with detailed reasons."""
        msg = (
            f"[ROI_REJECTED] event={event_id}, reason={reason}, sharpness={sharpness:.1f}, "
            f"dims={dimensions}, brightness={brightness:.1f}"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="BagEvent",
            event_id=event_id,
            reason=reason,
            sharpness=sharpness,
            dimensions=dimensions,
            brightness=brightness,
            **kwargs
        )
    
    def classification_candidate(self, track_id: int, candidate_idx: int, 
                                label: str, confidence: float, sharpness: float,
                                relative_time: float, contribution: float, **kwargs):
        """Log individual candidate classification in evidence accumulation."""
        msg = (
            f"[CANDIDATE] track={track_id}, idx={candidate_idx}, label={label}, "
            f"conf={confidence:.3f}, sharpness={sharpness:.1f}, time={relative_time:.2f}, "
            f"contrib={contribution:.3f}"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="ClassifierService",
            track_id=track_id,
            candidate_idx=candidate_idx,
            label=label,
            confidence=confidence,
            sharpness=sharpness,
            relative_time=relative_time,
            contribution=contribution,
            **kwargs
        )
    
    def classification_history_vote(self, track_id: int, current_label: str, current_confidence: float,
                                   history_label: str, history_confidence: float, vote_count: int, 
                                   history_size: int, history_buffer: list, **kwargs):
        """Log when classification uses history vote instead of current classification."""
        msg = (
            f"[HISTORY_VOTE] track={track_id}, current={current_label}({current_confidence:.2f}), "
            f"history={history_label}({history_confidence:.2f}), votes={vote_count}/{history_size}"
        )
        self._log_structured(
            logging.INFO,
            msg,
            component="ClassifierService",
            track_id=track_id,
            current_label=current_label,
            current_confidence=current_confidence,
            history_label=history_label,
            history_confidence=history_confidence,
            vote_count=vote_count,
            history_size=history_size,
            history_buffer=history_buffer,
            **kwargs
        )
    
    def label_reuse_override(self, track_id: int, prev_label: str, new_label: str, 
                            new_confidence: float, streak_len: int, 
                            dominance_label: str = None, dominance_ratio: float = None,
                            candidate_tops: list = None, reason: str = None, **kwargs):
        """Log when previous label is reused instead of current low-confidence classification."""
        msg_parts = [
            f"[LABEL_REUSE] track={track_id}",
            f"prev={prev_label}",
            f"new={new_label}({new_confidence:.2f})",
            f"streak={streak_len}"
        ]
        
        if dominance_label and dominance_ratio is not None:
            msg_parts.append(f"dom={dominance_label}({dominance_ratio:.2f})")
        
        if reason:
            msg_parts.append(f"reason={reason}")
        
        self._log_structured(
            logging.INFO,
            ", ".join(msg_parts),
            component="ClassifierService",
            track_id=track_id,
            prev_label=prev_label,
            new_label=new_label,
            new_confidence=new_confidence,
            streak_len=streak_len,
            dominance_label=dominance_label,
            dominance_ratio=dominance_ratio,
            candidate_tops=candidate_tops,
            reuse_reason=reason,
            **kwargs
        )
    
    def label_volatility_flag(self, track_id: int, label_changes: int, lifespan: int,
                             volatility_score: float, label_history: list, **kwargs):
        """Log high-volatility track detection."""
        msg = (
            f"[HIGH_VOLATILITY] track={track_id}, changes={label_changes}, "
            f"lifespan={lifespan}, volatility={volatility_score:.3f}"
        )
        self._log_structured(
            logging.WARNING,
            msg,
            component="ClassifierService",
            track_id=track_id,
            label_changes=label_changes,
            lifespan=lifespan,
            volatility_score=volatility_score,
            label_history=label_history,
            **kwargs
        )
    
    def count_updated(self, bag_type: str, new_count: int, track_id: int, 
                     confidence: float, phash: str = None, **kwargs):
        """Log count updates for bag types."""
        msg = (
            f"[COUNT_UPDATE] type={bag_type}, count={new_count}, track={track_id}, "
            f"conf={confidence:.3f}"
        )
        self._log_structured(
            logging.INFO,
            msg,
            component="BagCounterApp",
            bag_type=bag_type,
            new_count=new_count,
            track_id=track_id,
            confidence=confidence,
            phash=phash,
            **kwargs
        )
    
    def frame_processed(self, frame_id: int, detection_time_ms: float, 
                       monitor_time_ms: float, total_time_ms: float,
                       detections_count: int, events_ready: int = 0,
                       queue_sizes: dict = None, **kwargs):
        """Log frame processing with performance metrics."""
        msg = (
            f"[FRAME] id={frame_id}, detect={detection_time_ms:.1f}ms, "
            f"monitor={monitor_time_ms:.1f}ms, total={total_time_ms:.1f}ms, "
            f"dets={detections_count}, ready={events_ready}"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="BagCounterApp",
            frame_id=frame_id,
            detection_time_ms=detection_time_ms,
            monitor_time_ms=monitor_time_ms,
            total_time_ms=total_time_ms,
            detections_count=detections_count,
            events_ready=events_ready,
            queue_sizes=queue_sizes,
            **kwargs
        )
    
    def queue_backpressure(self, queue_name: str, utilization: float, 
                          drops: int, action: str, **kwargs):
        """Log queue backpressure and adaptive actions."""
        msg = (
            f"[BACKPRESSURE] queue={queue_name}, util={utilization:.1%}, "
            f"drops={drops}, action={action}"
        )
        self._log_structured(
            logging.WARNING,
            msg,
            component="BagCounterApp",
            queue_name=queue_name,
            utilization=utilization,
            drops=drops,
            action=action,
            **kwargs
        )
    
    def pipeline_error(self, component: str, operation: str, error_type: str,
                      error_message: str, affected_ids: list = None,
                      context: dict = None, **kwargs):
        """Log pipeline errors with full context for debugging."""
        msg = (
            f"[ERROR] component={component}, op={operation}, type={error_type}, "
            f"msg={error_message}, affected={affected_ids}"
        )
        self._log_structured(
            logging.ERROR,
            msg,
            component=component,
            operation=operation,
            error_type=error_type,
            error_message=error_message,
            affected_ids=affected_ids,
            upstream_context=context,
            **kwargs
        )
    
    # ==========================================================================
    # V5 Event-Centric Tracking Logs
    # ==========================================================================
    
    def event_committed(self, event_id: int, lifespan_ms: float, state: str,
                       open_evidence: int, closed_evidence: int,
                       roi_count: int, commit_reason: str,
                       detection_gaps: list = None, **kwargs):
        """Log event commit with full debug information for analysis."""
        msg = (
            f"[EVENT_COMMITTED] id={event_id}, lifespan={lifespan_ms:.0f}ms, "
            f"open_ev={open_evidence}, closed_ev={closed_evidence}, "
            f"rois={roi_count}, reason={commit_reason}"
        )
        self._log_structured(
            logging.INFO,
            msg,
            component="EventCentricTracker",
            event_id=event_id,
            lifespan_ms=lifespan_ms,
            state=state,
            open_evidence=open_evidence,
            closed_evidence=closed_evidence,
            roi_count=roi_count,
            commit_reason=commit_reason,
            detection_gaps=detection_gaps,
            **kwargs
        )
    
    def event_association(self, event_id: int, detection_centroid: tuple,
                         event_centroid: tuple, distance_px: float,
                         time_gap_ms: float, associated: bool, 
                         rejection_reason: str = None, **kwargs):
        """Log detection-to-event association decisions for debugging."""
        status = "ASSOCIATED" if associated else f"REJECTED ({rejection_reason})"
        msg = (
            f"[ASSOCIATION] event={event_id}, dist={distance_px:.1f}px, "
            f"gap={time_gap_ms:.0f}ms, {status}"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="EventCentricTracker",
            event_id=event_id,
            detection_centroid=detection_centroid,
            event_centroid=event_centroid,
            distance_px=distance_px,
            time_gap_ms=time_gap_ms,
            associated=associated,
            rejection_reason=rejection_reason,
            **kwargs
        )
    
    def hybrid_association_attempt(self, event_id: int, detection_centroid: tuple,
                                   event_centroid: tuple, distance_px: float,
                                   distance_threshold: float, iou_value: float,
                                   iou_threshold: float, time_gap_ms: float,
                                   centroid_match: bool, iou_match: bool,
                                   associated: bool, match_type: str, **kwargs):
        """
        Log parallel hybrid association attempts with detailed metrics.
        
        This logs every association attempt with both centroid distance and IoU values,
        regardless of whether the association succeeded or failed. This is crucial for
        debugging flip/spin scenarios where centroid distance fails but IoU succeeds.
        
        Args:
            event_id: Event ID being matched against
            detection_centroid: (x, y) of detection centroid
            event_centroid: (x, y) of event's last known centroid
            distance_px: Computed centroid distance in pixels
            distance_threshold: Active distance threshold (may be velocity-scaled)
            iou_value: Computed IoU value (0.0-1.0)
            iou_threshold: Configured IoU threshold
            time_gap_ms: Time gap between detection and last event update
            centroid_match: True if centroid criterion was met
            iou_match: True if IoU criterion was met
            associated: True if detection was associated with event
            match_type: One of 'both_match', 'centroid_match', 'iou_match', 'no_match', 'time_exceeded'
        """
        status = "SUCCESS" if associated else "REJECTED"
        msg = (
            f"[HYBRID_ASSOCIATION] event={event_id}, {status} ({match_type}) | "
            f"dist={distance_px:.1f}px (thresh={distance_threshold:.1f}px, match={centroid_match}), "
            f"iou={iou_value:.2f} (thresh={iou_threshold}, match={iou_match}), "
            f"gap={time_gap_ms:.0f}ms"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="EventCentricTracker",
            event_id=event_id,
            detection_centroid=detection_centroid,
            event_centroid=event_centroid,
            distance_px=distance_px,
            distance_threshold=distance_threshold,
            iou_value=iou_value,
            iou_threshold=iou_threshold,
            time_gap_ms=time_gap_ms,
            centroid_match=centroid_match,
            iou_match=iou_match,
            associated=associated,
            match_type=match_type,
            **kwargs
        )
    
    def ghost_state_update(self, event_id: int, time_since_detection_ms: float,
                          state: str, near_exit: bool = False, **kwargs):
        """Log ghost state updates for detection gap analysis."""
        msg = (
            f"[GHOST_UPDATE] event={event_id}, idle={time_since_detection_ms:.0f}ms, "
            f"state={state}, near_exit={near_exit}"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="EventCentricTracker",
            event_id=event_id,
            time_since_detection_ms=time_since_detection_ms,
            state=state,
            near_exit=near_exit,
            **kwargs
        )
    
    def event_debug_summary(self, event_id: int, debug_info: dict, **kwargs):
        """Log comprehensive debug summary for event analysis."""
        msg = (
            f"[EVENT_DEBUG] id={event_id}, lifespan={debug_info.get('lifespan_ms', 0):.0f}ms, "
            f"gaps={len(debug_info.get('detection_gaps', []))}, "
            f"transitions={len(debug_info.get('state_transitions', []))}"
        )
        self._log_structured(
            logging.DEBUG,
            msg,
            component="EventCentricTracker",
            event_id=event_id,
            **debug_info,
            **kwargs
        )


# Create structured logger instance
structured_logger = StructuredLogger(logger)


# ============================================================================
# Performance Logging Decorator
# ============================================================================

def log_performance(component: str):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_performance("Classifier")
        def classify_image(self, image):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"[{component}] {func.__name__} completed in {elapsed:.2f}ms")
            return result
        return wrapper
    return decorator


# ============================================================================
# Public API
# ============================================================================

def set_log_level(level_name: str):
    """Change logging level dynamically at runtime."""
    level = getattr(logging, level_name.upper(), None)
    if level is None:
        logger.warning(f"[AppLogging] Invalid level: {level_name}")
        return

    root_logger.setLevel(level)
    console_handler.setLevel(level)
    file_handler.setLevel(level)
    if json_file_handler:
        json_file_handler.setLevel(level)
    logger.setLevel(level)
    logger.info(f"[AppLogging] Log level changed to {level_name.upper()}")


def get_log_file_paths() -> Dict[str, str]:
    """Return paths to log files for external access."""
    paths = {
        "app_log": os.path.join(LOG_DIR, "app.log"),
    }
    if ENABLE_JSON_LOGGING:
        paths["json_log"] = os.path.join(LOG_DIR, "app.json.log")
    return paths


def flush_logs():
    """Flush all log handlers - useful before shutdown."""
    for handler in root_logger.handlers:
        handler.flush()


# Log initialization
logger.info(f"[AppLogging] Initialized - level={log_level_name}, log_dir={LOG_DIR}, json_logging={ENABLE_JSON_LOGGING}")
