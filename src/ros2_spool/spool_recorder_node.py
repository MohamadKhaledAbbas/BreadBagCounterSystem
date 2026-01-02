#!/usr/bin/env python3
"""
Spool Recorder Node for Accuracy Mode.

This node subscribes to H.264 encoded frames from the RTSP client
and writes them to disk in chunked segment files for later replay.

Features:
- Subscribes to /rtsp_image_ch_0 (img_msgs/msg/H26XFrame)
- Non-blocking callback with bounded queue
- Background writer thread for disk I/O
- Segment rotation with IDR alignment
- SPS/PPS caching for segment boundaries
- Automatic retention policy enforcement

Usage:
    python -m src.ros2_spool.spool_recorder_node

Configuration (via database config table):
    spool_dir: Directory for spool files (default: /home/sunrise/BreadCounting/data/spool)
    spool_segment_duration: Target segment duration in seconds (default: 5.0)
    spool_retention_seconds: Maximum segment age before deletion (default: 180)
"""

import os
import sys
import time
import queue
import signal
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.config.settings import AppConfig

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK
from src.spool.h264_nal import extract_sps_pps, is_idr_frame
from src.spool.segment_io import SegmentWriter, FrameRecord
from src.spool.retention import RetentionPolicy, cleanup_stale_tmp_files
from src.spool.spool_utils import format_structured_log, throttled_log
from src.logging.Database import DatabaseManager
from src import constants

# ROS2 imports (only on RDK platform)
if IS_RDK:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from img_msgs.msg import H26XFrame
else:
    # Stub for non-RDK development
    class Node:
        def __init__(self, name): pass
        def get_logger(self): return logger
        def create_subscription(self, *args, **kwargs): pass
        def destroy_node(self): pass


# Default configuration values
DEFAULT_SPOOL_DIR = "/home/sunrise/BreadCounting/data/spool"
DEFAULT_SEGMENT_DURATION = 5.0
DEFAULT_MAX_SEGMENT_DURATION = 10.0
DEFAULT_RETENTION_SECONDS = 180.0
DEFAULT_QUEUE_SIZE = 100
DEFAULT_STATS_INTERVAL = 10.0
DEFAULT_DROP_LOG_THROTTLE = 5.0  # Seconds between drop warning logs
DEFAULT_MAX_SPOOL_SIZE_BYTES = 2_147_483_648  # 2GB hard limit to prevent SD card fill


@dataclass
class SpoolConfig:
    """Configuration for the spool recorder."""
    spool_dir_path: str = DEFAULT_SPOOL_DIR
    segment_duration: float = DEFAULT_SEGMENT_DURATION
    max_segment_duration: float = DEFAULT_MAX_SEGMENT_DURATION
    retention_seconds: float = DEFAULT_RETENTION_SECONDS
    queue_size: int = DEFAULT_QUEUE_SIZE
    stats_interval: float = DEFAULT_STATS_INTERVAL
    drop_log_throttle: float = DEFAULT_DROP_LOG_THROTTLE
    enable_backpressure_hook: bool = False  # Future: enable backpressure on drops
    max_spool_size_bytes: int = DEFAULT_MAX_SPOOL_SIZE_BYTES  # Maximum spool directory size


def load_default_config() -> SpoolConfig:
    """Load spool configuration from database config table."""
    return SpoolConfig(
        spool_dir_path=DEFAULT_SPOOL_DIR,
        segment_duration=DEFAULT_SEGMENT_DURATION,
        retention_seconds=DEFAULT_RETENTION_SECONDS,
    )


class SpoolRecorderNode(Node):
    """
    ROS2 Node that records H.264 frames to disk.
    
    The node uses a two-stage architecture:
    1. ROS2 callback enqueues frames to bounded memory queue (non-blocking)
    2. Writer thread flushes queue to disk, handles rotation and retention
    
    This design ensures the ROS2 callback never blocks on disk I/O.
    """
    
    def __init__(self, config: Optional[SpoolConfig] = None):
        super().__init__('spool_recorder')
        
        # Load configuration from database if not provided
        self.config = config or load_default_config()

        spool_dir = Path(self.config.spool_dir_path)
        spool_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[SpoolRecorder] Initializing with config: "
                   f"spool_dir={self.config.spool_dir_path}, "
                   f"segment_duration={self.config.segment_duration}s, "
                   f"retention={self.config.retention_seconds}s")
        
        # Initialize components
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_size)
        self._writer: Optional[SegmentWriter] = None
        self._retention: Optional[RetentionPolicy] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Statistics
        self._frames_received = 0
        self._frames_dropped = 0
        self._frames_written = 0
        self._ingress_drop_events = 0  # Number of distinct drop events
        self._last_stats_time = time.time()
        self._stats_lock = threading.Lock()
        self._drop_log_throttle_dict = {}  # For throttled drop logging
        
        # SPS/PPS cache
        self._cached_sps: Optional[bytes] = None
        self._cached_pps: Optional[bytes] = None
        
        # Create QoS profile for H26X subscription
        if IS_RDK:
            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            # Subscribe to H26X frames from RTSP client
            self._subscription = self.create_subscription(
                H26XFrame,
                '/rtsp_image_ch_0',
                self._frame_callback,
                qos
            )
            logger.info("[SpoolRecorder] Subscribed to /rtsp_image_ch_0")
    
    def start(self):
        """Start the recorder (writer thread and retention)."""
        if self._running:
            return
        
        logger.info("[SpoolRecorder] Starting...")

        # Clean up stale tmp files from previous crashes
        cleanup_stale_tmp_files(self.config.spool_dir_path)
        
        # Initialize segment writer
        self._writer = SegmentWriter(
            spool_dir=self.config.spool_dir_path,
            segment_duration=self.config.segment_duration,
            max_segment_duration=self.config.max_segment_duration,
            write_metadata=True
        )
        self._writer.start()
        
        # Initialize and start retention policy
        self._retention = RetentionPolicy(
            spool_dir=self.config.spool_dir_path,
            retention_seconds=self.config.retention_seconds,
            cleanup_interval=10.0,
            min_segments_to_keep=2,
            retention_safety_enabled=True,  # Enable processor state awareness
            max_spool_size_bytes=self.config.max_spool_size_bytes,  # Enforce 2GB limit
            delete_processed_segments=False  # Only delete by age/size, not immediate
        )
        self._retention.start()
        
        logger.info(format_structured_log(
            "[SpoolRecorder] Retention policy configured",
            retention_seconds=self.config.retention_seconds,
            max_spool_size_mb=self.config.max_spool_size_bytes / 1024 / 1024,
            min_segments_to_keep=2
        ))
        
        # Start writer thread
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="SpoolWriter"
        )
        self._writer_thread.start()
        
        logger.info("[SpoolRecorder] Started")
    
    def stop(self):
        """Stop the recorder gracefully with proper cleanup."""
        if not self._running:
            return
        
        logger.info("[SpoolRecorder] Stopping...")
        self._running = False
        
        # Signal writer thread to stop and wait for it to finish
        if self._writer_thread and self._writer_thread.is_alive():
            logger.info("[SpoolRecorder] Waiting for writer thread to finish...")
            self._writer_thread.join(timeout=10.0)
            if self._writer_thread.is_alive():
                logger.error("[SpoolRecorder] ⚠ Writer thread did not stop gracefully within timeout")
                # Try to flush remaining queue items
                remaining = self._frame_queue.qsize()
                if remaining > 0:
                    logger.warning(f"[SpoolRecorder] {remaining} frames still in queue, attempting flush...")
            else:
                logger.info("[SpoolRecorder] Writer thread stopped successfully")
        
        # Stop retention
        if self._retention:
            logger.info("[SpoolRecorder] Stopping retention policy...")
            self._retention.stop()
        
        # Close writer (ensures flush)
        if self._writer:
            logger.info("[SpoolRecorder] Closing segment writer...")
            self._writer.close()
        
        # Log final stats with structured format
        with self._stats_lock:
            final_msg = format_structured_log(
                "[SpoolRecorder] Final stats",
                frames_received=self._frames_received,
                frames_written=self._frames_written,
                frames_dropped=self._frames_dropped,
                drop_events=self._ingress_drop_events,
                queue_remaining=self._frame_queue.qsize()
            )
            logger.info(final_msg)
            
            # Escalate alert if sustained drops
            if self._ingress_drop_events > 10:
                logger.error(
                    f"[SpoolRecorder] 🔴 CRITICAL: Sustained ingress drops detected! "
                    f"drop_events={self._ingress_drop_events}, "
                    f"total_dropped={self._frames_dropped}"
                )
        
        logger.info("[SpoolRecorder] Stopped")
    
    def _frame_callback(self, msg):
        """
        ROS2 callback for H26X frames.
        
        This callback is designed to be non-blocking. It simply
        enqueues the frame data for the writer thread to process.
        """
        with self._stats_lock:
            self._frames_received += 1
        
        # Create frame record from ROS message
        # Handle encoding field that might be numpy array, bytes, or string
        encoding = msg.encoding
        if hasattr(encoding, 'tobytes'):
            # numpy array
            encoding = encoding.tobytes().decode('utf-8', errors='replace').rstrip('\x00')
        elif isinstance(encoding, bytes):
            encoding = encoding.decode('utf-8', errors='replace').rstrip('\x00')
        elif not isinstance(encoding, str):
            # Fallback for other array-like types
            try:
                encoding = bytes(encoding).decode('utf-8', errors='replace').rstrip('\x00')
            except (TypeError, ValueError):
                encoding = "H264"  # Default encoding
        
        record = FrameRecord(
            index=msg.index,
            width=msg.width,
            height=msg.height,
            dts_sec=msg.dts.sec,
            dts_nsec=msg.dts.nanosec,
            pts_sec=msg.pts.sec,
            pts_nsec=msg.pts.nanosec,
            encoding=encoding,
            data=bytes(msg.data)
        )
        
        # Extract SPS/PPS if present (for segment boundary insertion)
        sps, pps = extract_sps_pps(record.data)
        if sps:
            self._cached_sps = sps
        if pps:
            self._cached_pps = pps
        
        # Try to enqueue (non-blocking)
        try:
            self._frame_queue.put_nowait(record)
        except queue.Full:
            # Queue overflow - ingress drop detected
            with self._stats_lock:
                self._frames_dropped += 1
                self._ingress_drop_events += 1
            
            # Emit throttled high-severity structured log
            msg = format_structured_log(
                "🔴 INGRESS DROP: Queue overflow",
                frame_index=record.index,
                queue_size=self.config.queue_size,
                drops_total=self._frames_dropped,
                drop_events=self._ingress_drop_events
            )
            throttled_log(
                logger.error,
                msg,
                key="ingress_drop",
                throttle_dict=self._drop_log_throttle_dict,
                min_interval=self.config.drop_log_throttle
            )
            
            # Backpressure hook (for future use)
            if self.config.enable_backpressure_hook:
                logger.warning("[SpoolRecorder] Backpressure hook enabled but not implemented yet")
            
            # Try to drop oldest frame and retry
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(record)
            except (queue.Empty, queue.Full):
                pass
    
    def _writer_loop(self):
        """
        Background thread that writes frames to disk.
        
        This thread consumes frames from the queue and writes them
        to segment files. It handles segment rotation and ensures
        disk I/O doesn't block the ROS2 callback.
        """
        logger.info("[SpoolRecorder] Writer thread started")
        
        while self._running:
            try:
                # Get frame from queue with timeout
                record = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                # Log stats periodically
                self._maybe_log_stats()
                continue
            
            try:
                # Check for IDR frame
                has_idr = is_idr_frame(record.data)
                
                # Update SPS/PPS cache in writer
                if self._writer and (self._cached_sps or self._cached_pps):
                    self._writer.update_sps_pps(self._cached_sps, self._cached_pps)
                
                # Write frame to segment
                if self._writer and self._writer.write_frame(record, has_idr):
                    with self._stats_lock:
                        self._frames_written += 1
                else:
                    logger.warning(f"[SpoolRecorder] Failed to write frame {record.index}")
                
            except Exception as e:
                logger.error(f"[SpoolRecorder] Error writing frame: {e}")
            
            # Log stats periodically
            self._maybe_log_stats()
        
        logger.info("[SpoolRecorder] Writer thread stopped")
    
    def _maybe_log_stats(self):
        """Log statistics periodically."""
        current_time = time.time()
        if current_time - self._last_stats_time >= self.config.stats_interval:
            with self._stats_lock:
                queue_size = self._frame_queue.qsize()
                queue_util = (queue_size / self.config.queue_size * 100) if self.config.queue_size > 0 else 0
                
                # Structured stats logging
                stats_msg = format_structured_log(
                    "[SpoolRecorder] Stats",
                    frames_received=self._frames_received,
                    frames_written=self._frames_written,
                    frames_dropped=self._frames_dropped,
                    drop_events=self._ingress_drop_events,
                    queue_size=queue_size,
                    queue_max=self.config.queue_size,
                    queue_util_pct=f"{queue_util:.1f}"
                )
                logger.info(stats_msg)
                
                # Log retention stats
                if self._retention:
                    ret_stats = self._retention.get_stats()
                    ret_msg = format_structured_log(
                        "[SpoolRecorder] Retention",
                        total_segments=ret_stats['total_segments'],
                        total_size_mb=f"{ret_stats['total_size_mb']:.1f}",
                        oldest_age_sec=f"{ret_stats['oldest_segment_age_seconds']:.1f}"
                    )
                    logger.info(ret_msg)
            
            self._last_stats_time = current_time


def main():
    """Main entry point for the spool recorder node."""
    logger.info("=" * 60)
    logger.info("  Spool Recorder Node - Accuracy Mode")
    logger.info("=" * 60)
    
    if not IS_RDK:
        logger.error("[SpoolRecorder] This node requires RDK platform with ROS2")
        logger.info("[SpoolRecorder] Running in stub mode for testing")
    
    # Initialize ROS2
    if IS_RDK:
        rclpy.init()
    
    # Create and start node
    node = SpoolRecorderNode()
    node.start()
    
    # Setup signal handlers for clean shutdown
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"[SpoolRecorder] Received signal {signum}, shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Spin ROS2
    if IS_RDK:
        try:
            while not shutdown_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            node.stop()
            node.destroy_node()
            rclpy.shutdown()
    else:
        # For testing without ROS2
        logger.info("[SpoolRecorder] Running in test mode. Press Ctrl+C to exit.")
        try:
            shutdown_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            node.stop()
    
    logger.info("[SpoolRecorder] Shutdown complete")


if __name__ == '__main__':
    main()
