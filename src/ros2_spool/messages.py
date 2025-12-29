#!/usr/bin/env python3
"""
ROS2 Message Definitions for Accuracy Mode Spool Processing.

Since this repository doesn't use colcon/ament, we define message structures
using Python dataclasses and provide conversion utilities to/from ROS2 messages.

These messages replace the simple UInt32 ACK with structured, session-aware protocol.
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessingAck:
    """
    Structured ACK message for frame processing acknowledgment.
    
    Attributes:
        frame_index: Frame index being acknowledged (uint32)
        session_id: UUID of the current processing session (string)
        seq: Monotonic sequence number assigned by Processor (uint64)
        sent_time_sec: Timestamp seconds when frame was published (int64)
        sent_time_nsec: Timestamp nanoseconds when frame was published (uint32)
        segment_num: Optional segment number if available (int32, -1 if not set)
    """
    frame_index: int
    session_id: str
    seq: int
    sent_time_sec: int
    sent_time_nsec: int
    segment_num: int = -1
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingAck':
        """Create ProcessingAck from dictionary."""
        return cls(
            frame_index=data['frame_index'],
            session_id=data['session_id'],
            seq=data['seq'],
            sent_time_sec=data['sent_time_sec'],
            sent_time_nsec=data['sent_time_nsec'],
            segment_num=data.get('segment_num', -1)
        )
    
    def to_dict(self) -> dict:
        """Convert ProcessingAck to dictionary."""
        return {
            'frame_index': self.frame_index,
            'session_id': self.session_id,
            'seq': self.seq,
            'sent_time_sec': self.sent_time_sec,
            'sent_time_nsec': self.sent_time_nsec,
            'segment_num': self.segment_num
        }


@dataclass
class ProcessingReady:
    """
    READY message published by BagCounterApp on startup.
    
    Signals that the consumer is ready to process frames for a given session.
    
    Attributes:
        session_id: UUID of the session the consumer is ready to serve (string)
        ready_time_sec: Timestamp seconds when ready (int64)
        ready_time_nsec: Timestamp nanoseconds when ready (uint32)
    """
    session_id: str
    ready_time_sec: int
    ready_time_nsec: int
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingReady':
        """Create ProcessingReady from dictionary."""
        return cls(
            session_id=data['session_id'],
            ready_time_sec=data['ready_time_sec'],
            ready_time_nsec=data['ready_time_nsec']
        )
    
    def to_dict(self) -> dict:
        """Convert ProcessingReady to dictionary."""
        return {
            'session_id': self.session_id,
            'ready_time_sec': self.ready_time_sec,
            'ready_time_nsec': self.ready_time_nsec
        }


@dataclass
class FrameMetadata:
    """
    Metadata for a published frame, sent alongside encoded frame.
    
    This travels on /spool/current_frame_metadata and provides context
    for the consumer to construct proper ACKs.
    
    Attributes:
        frame_index: Frame index (uint32)
        session_id: UUID of current session (string)
        seq: Monotonic sequence number (uint64)
        sent_time_sec: Timestamp seconds (int64)
        sent_time_nsec: Timestamp nanoseconds (uint32)
        segment_num: Optional segment number (int32, -1 if not set)
    """
    frame_index: int
    session_id: str
    seq: int
    sent_time_sec: int
    sent_time_nsec: int
    segment_num: int = -1
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FrameMetadata':
        """Create FrameMetadata from dictionary."""
        return cls(
            frame_index=data['frame_index'],
            session_id=data['session_id'],
            seq=data['seq'],
            sent_time_sec=data['sent_time_sec'],
            sent_time_nsec=data['sent_time_nsec'],
            segment_num=data.get('segment_num', -1)
        )
    
    def to_dict(self) -> dict:
        """Convert FrameMetadata to dictionary."""
        return {
            'frame_index': self.frame_index,
            'session_id': self.session_id,
            'seq': self.seq,
            'sent_time_sec': self.sent_time_sec,
            'sent_time_nsec': self.sent_time_nsec,
            'segment_num': self.segment_num
        }


# Utility functions

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


# ROS2 message conversion utilities
# These functions convert between our dataclasses and ROS2 String messages
# We use String messages with JSON encoding as a workaround since we can't
# define custom message types without colcon/ament


def processing_ack_to_ros_string(ack: ProcessingAck) -> str:
    """
    Convert ProcessingAck to JSON string for ROS2 String message.
    
    Args:
        ack: ProcessingAck instance
        
    Returns:
        JSON string representation
    """
    return json.dumps(ack.to_dict())


def processing_ack_from_ros_string(data: str) -> ProcessingAck:
    """
    Parse ProcessingAck from ROS2 String message data.
    
    Args:
        data: JSON string from ROS2 String message
        
    Returns:
        ProcessingAck instance
    """
    return ProcessingAck.from_dict(json.loads(data))


def processing_ready_to_ros_string(ready: ProcessingReady) -> str:
    """
    Convert ProcessingReady to JSON string for ROS2 String message.
    
    Args:
        ready: ProcessingReady instance
        
    Returns:
        JSON string representation
    """
    return json.dumps(ready.to_dict())


def processing_ready_from_ros_string(data: str) -> ProcessingReady:
    """
    Parse ProcessingReady from ROS2 String message data.
    
    Args:
        data: JSON string from ROS2 String message
        
    Returns:
        ProcessingReady instance
    """
    return ProcessingReady.from_dict(json.loads(data))


def frame_metadata_to_ros_string(metadata: FrameMetadata) -> str:
    """
    Convert FrameMetadata to JSON string for ROS2 String message.
    
    Args:
        metadata: FrameMetadata instance
        
    Returns:
        JSON string representation
    """
    return json.dumps(metadata.to_dict())


def frame_metadata_from_ros_string(data: str) -> FrameMetadata:
    """
    Parse FrameMetadata from ROS2 String message data.
    
    Args:
        data: JSON string from ROS2 String message
        
    Returns:
        FrameMetadata instance
    """
    return FrameMetadata.from_dict(json.loads(data))
