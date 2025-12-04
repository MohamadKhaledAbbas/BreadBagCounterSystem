import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# ANSI color codes for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    COLORS = {
        'ERROR': '\033[91m',    # Red
        'WARNING': '\033[93m',  # Yellow
        'INFO': '\033[92m',     # Green
        'DEBUG': '\033[96m',    # Cyan
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        result = super().format(record)
        
        # Reset levelname to original (for file handlers)
        record.levelname = levelname
        
        return result


# Get log level from environment variable, default to DEBUG for troubleshooting
log_level_name = os.environ.get("LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL = getattr(logging, log_level_name, logging.DEBUG)

# Console formatter with colors and milliseconds
console_formatter = ColoredFormatter(
    fmt='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# File formatter (no colors)
file_formatter = logging.Formatter(
    fmt='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
root_logger.handlers.clear()

# Console handler with colored output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# File handler with rotation (10MB max, 3 backup files)
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "breadcounter.log")

file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3
)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# App logger
logger = logging.getLogger("BreadCounter")
logger.setLevel(LOG_LEVEL)

# Quiet third-party loggers
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("rclpy").setLevel(logging.WARNING)

def set_log_level(level_name: str):
    """
    Change the log level at runtime.
    
    Args:
        level_name: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, level_name.upper(), None)
    if level is None:
        logger.warning(f"[AppLogging] Invalid log level: {level_name}")
        return
    
    root_logger.setLevel(level)
    console_handler.setLevel(level)
    file_handler.setLevel(level)
    logger.setLevel(level)
    logger.info(f"[AppLogging] Log level changed to: {level_name.upper()}")