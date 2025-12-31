"""
Utility functions for spool operations.

Provides helper functions for:
- CRC32 checksum calculation for frame traceability
- State file I/O for persisting processor progress
- Structured logging helpers
"""

import os
import json
import time
import zlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from src.utils.AppLogging import logger


@dataclass
class ProcessorState:
    """
    Persistent state for the spool processor.
    
    This state is saved to disk to enable restart continuity,
    preventing replay or skip of frames after restart.
    
    Attributes:
        last_published_index: Last frame index successfully published
        last_published_segment: Segment number of last published frame
        session_id: Session ID when state was saved
        timestamp: Unix timestamp when state was saved
    """
    last_published_index: int
    last_published_segment: int
    session_id: str
    timestamp: float


def calculate_crc32(data: bytes) -> int:
    """
    Calculate CRC32 checksum for frame data.
    
    This provides a lightweight checksum for frame traceability
    without the overhead of cryptographic hashes.
    
    Args:
        data: Frame data bytes
        
    Returns:
        CRC32 checksum as unsigned 32-bit integer
    """
    return zlib.crc32(data) & 0xFFFFFFFF


def save_processor_state(state_path: str, state: ProcessorState) -> bool:
    """
    Save processor state to disk atomically.
    
    Uses atomic write pattern (write to .tmp, then rename) to prevent
    corruption during crashes.
    
    Args:
        state_path: Path to state file
        state: ProcessorState object to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        state_dict = asdict(state)
        tmp_path = state_path + ".tmp"
        
        # Write to temporary file
        with open(tmp_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        # Atomic rename
        os.replace(tmp_path, state_path)
        
        logger.debug(f"[SpoolUtils] Saved processor state: "
                    f"index={state.last_published_index}, "
                    f"segment={state.last_published_segment}")
        return True
        
    except Exception as e:
        logger.error(f"[SpoolUtils] Failed to save processor state: {e}")
        return False


def load_processor_state(state_path: str) -> Optional[ProcessorState]:
    """
    Load processor state from disk.
    
    Args:
        state_path: Path to state file
        
    Returns:
        ProcessorState object if successful, None if file doesn't exist or is invalid
    """
    if not os.path.exists(state_path):
        logger.info(f"[SpoolUtils] No processor state file found at {state_path}")
        return None
    
    try:
        with open(state_path, 'r') as f:
            state_dict = json.load(f)
        
        state = ProcessorState(
            last_published_index=state_dict['last_published_index'],
            last_published_segment=state_dict['last_published_segment'],
            session_id=state_dict['session_id'],
            timestamp=state_dict['timestamp']
        )
        
        logger.info(f"[SpoolUtils] Loaded processor state: "
                   f"index={state.last_published_index}, "
                   f"segment={state.last_published_segment}, "
                   f"session={state.session_id[:8]}")
        return state
        
    except Exception as e:
        logger.error(f"[SpoolUtils] Failed to load processor state: {e}")
        return None


def format_structured_log(message: str, **fields) -> str:
    """
    Format a log message with structured key=value fields.
    
    This makes logs machine-parsable while still human-readable.
    
    Args:
        message: Base log message
        **fields: Key-value pairs to include
        
    Returns:
        Formatted log string
        
    Example:
        >>> format_structured_log("Frame published", index=100, seq=42, crc32=0xABCD1234)
        "Frame published: index=100 seq=42 crc32=0xabcd1234"
    """
    if not fields:
        return message
    
    field_strs = []
    for key, value in fields.items():
        # Format hex values nicely
        if isinstance(value, int) and key.lower().endswith(('crc32', 'checksum', 'hash')):
            field_strs.append(f"{key}=0x{value:08x}")
        else:
            field_strs.append(f"{key}={value}")
    
    return f"{message}: {' '.join(field_strs)}"


def throttled_log(
    logger_func,
    message: str,
    key: str,
    throttle_dict: Dict[str, float],
    min_interval: float = 1.0
) -> bool:
    """
    Log a message with throttling to prevent spam.
    
    Args:
        logger_func: Logger function to call (e.g., logger.warning)
        message: Message to log
        key: Unique key for this log type (for throttling)
        throttle_dict: Dictionary to track last log times
        min_interval: Minimum seconds between logs of same type
        
    Returns:
        True if message was logged, False if throttled
    """
    current_time = time.time()
    last_time = throttle_dict.get(key, 0.0)
    
    if current_time - last_time >= min_interval:
        logger_func(message)
        throttle_dict[key] = current_time
        return True
    
    return False
