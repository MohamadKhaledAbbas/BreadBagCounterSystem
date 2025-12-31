#!/usr/bin/env python3
"""
ROS2 Message Utility Functions.

Minimal utility functions for ROS2 message handling.
"""

import time
import uuid


def generate_session_id() -> str:
    """Generate a unique session ID using UUID4."""
    return str(uuid.uuid4())


def get_current_time_ros() -> tuple:
    """
    Get current time in ROS2 Time format (seconds, nanoseconds).
    
    Returns:
        Tuple of (seconds: int, nanoseconds: int)
    """
    current = time.time()
    sec = int(current)
    nsec = int((current - sec) * 1e9)
    return sec, nsec

