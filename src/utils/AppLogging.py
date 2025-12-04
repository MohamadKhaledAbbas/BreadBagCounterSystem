import logging
import sys
import os


# ANSI color codes for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""

    COLORS = {
        'ERROR': '\033[91m',  # Red
        'WARNING': '\033[93m',  # Yellow
        'INFO': '\033[92m',  # Green
        'DEBUG': '\033[96m',  # Cyan
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        result = super().format(record)
        record.levelname = levelname  # restore
        return result


# Read environment log level
log_level_name = os.environ.get("LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL = getattr(logging, log_level_name, logging.DEBUG)

# Console formatter
console_formatter = ColoredFormatter(
    fmt='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
root_logger.handlers.clear()

# Console handler only
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Application logger
logger = logging.getLogger("BreadCounter")
logger.setLevel(LOG_LEVEL)

# Quiet third-party components
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("rclpy").setLevel(logging.WARNING)


def set_log_level(level_name: str):
    """Change logging level dynamically at runtime."""
    level = getattr(logging, level_name.upper(), None)
    if level is None:
        logger.warning(f"[AppLogging] Invalid level: {level_name}")
        return

    root_logger.setLevel(level)
    console_handler.setLevel(level)
    logger.setLevel(level)
    logger.info(f"[AppLogging] Log level changed to {level_name.upper()}")
