import logging
import sys
import os

# Get log level from environment variable, default to DEBUG for troubleshooting
log_level_name = os.environ. get("LOG_LEVEL", "DEBUG"). upper()
LOG_LEVEL = getattr(logging, log_level_name, logging.DEBUG)

# Formatter
formatter = logging. Formatter(
    fmt='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
root_logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# App logger
logger = logging.getLogger("BreadCounter")
logger.setLevel(LOG_LEVEL)

# Quiet third-party loggers
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("rclpy").setLevel(logging.WARNING)