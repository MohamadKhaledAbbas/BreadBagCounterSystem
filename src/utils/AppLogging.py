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
LOG_FILE_MAX_BYTES = int(os.environ.get("LOG_FILE_MAX_BYTES", 10 * 1024 * 1024))  # 10MB default
LOG_FILE_BACKUP_COUNT = int(os.environ.get("LOG_FILE_BACKUP_COUNT", 5))  # Keep 5 backup files
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
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
    
    def event_created(self, event_id: int, confidence: float, box: list, **kwargs):
        """Log event creation with structured data."""
        self._log_structured(
            logging.INFO,
            f"[EVENT_CREATED] id={event_id}, conf={confidence:.3f}",
            component="BagStateMonitor",
            event_id=event_id,
            confidence=confidence,
            box=box,
            **kwargs
        )
    
    def event_counted(self, event_id: int, label: str, confidence: float, 
                      open_hits: int, closed_hits: int, **kwargs):
        """Log successful event counting with structured data."""
        self._log_structured(
            logging.INFO,
            f"[EVENT_COUNTED] id={event_id}, label={label}, conf={confidence:.3f}",
            component="BagStateMonitor",
            event_id=event_id,
            label=label,
            confidence=confidence,
            open_hits=open_hits,
            closed_hits=closed_hits,
            **kwargs
        )
    
    def event_expired(self, event_id: int, state: str, frames_tracked: int, **kwargs):
        """Log event expiration with structured data."""
        self._log_structured(
            logging.DEBUG,
            f"[EVENT_EXPIRED] id={event_id}, state={state}, frames={frames_tracked}",
            component="BagStateMonitor",
            event_id=event_id,
            state=state,
            frames_tracked=frames_tracked,
            **kwargs
        )
    
    def classification_result(self, track_id: int, label: str, confidence: float,
                             candidates: int, used_voting: bool, **kwargs):
        """Log classification result with structured data."""
        self._log_structured(
            logging.INFO,
            f"[CLASSIFICATION] track={track_id}, label={label}, conf={confidence:.3f}",
            component="ClassifierService",
            track_id=track_id,
            label=label,
            confidence=confidence,
            candidates=candidates,
            used_voting=used_voting,
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
