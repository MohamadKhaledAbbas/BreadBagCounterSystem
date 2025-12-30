"""
Retention Policy Module for Spool Management.

Provides safe deletion policies for segment files to manage disk space
while maintaining data integrity during 24/7 operation.

Features:
- Age-based retention with configurable duration
- Safe deletion (only completed .bin files, never .tmp)
- Atomic cleanup operations
- Startup cleanup for stale temporary files
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
import threading

from src.utils.AppLogging import logger


def cleanup_stale_tmp_files(spool_dir: str, max_age_seconds: float = 60.0) -> int:
    """
    Clean up stale .tmp files on startup.
    
    Temporary files may be left behind if the recorder crashes during
    segment writing. This function safely removes old .tmp files.
    
    Args:
        spool_dir: Directory containing spool files
        max_age_seconds: Maximum age of .tmp files before cleanup
        
    Returns:
        Number of files cleaned up
    """
    spool_path = Path(spool_dir)
    if not spool_path.exists():
        return 0
    
    cleaned = 0
    current_time = time.time()
    
    for tmp_file in spool_path.glob("seg_*.tmp"):
        try:
            # Check file age
            file_mtime = tmp_file.stat().st_mtime
            age = current_time - file_mtime
            
            if age > max_age_seconds:
                tmp_file.unlink()
                logger.info(f"[Retention] Cleaned up stale tmp file: {tmp_file.name} "
                           f"(age: {age:.1f}s)")
                cleaned += 1
        except Exception as e:
            logger.warning(f"[Retention] Error cleaning up {tmp_file.name}: {e}")
    
    if cleaned > 0:
        logger.info(f"[Retention] Cleaned up {cleaned} stale tmp files")
    
    return cleaned


class RetentionPolicy:
    """
    Manages retention of segment files based on age.
    
    Features:
    - Configurable retention duration
    - Safe deletion (only closed .bin files)
    - Background cleanup thread option
    - Statistics tracking
    - V6: Processor progress awareness (retention safety)
    
    Usage:
        policy = RetentionPolicy('/path/to/spool', retention_seconds=180)
        policy.start()  # Start background cleanup
        
        # Or manual cleanup:
        deleted = policy.cleanup_once()
        
        # V6: Set processor progress to prevent deleting unprocessed data
        policy.set_last_processed_frame(frame_index)
        
        policy.stop()
    """
    
    def __init__(
        self,
        spool_dir: str,
        retention_seconds: float = 180.0,
        cleanup_interval: float = 10.0,
        min_segments_to_keep: int = 2,
        retention_safety_enabled: bool = True
    ):
        """
        Initialize the retention policy.
        
        Args:
            spool_dir: Directory containing spool files
            retention_seconds: Maximum age of segments before deletion
            cleanup_interval: Interval between cleanup checks
            min_segments_to_keep: Minimum segments to always keep
            retention_safety_enabled: V6 - Enable processor progress awareness
        """
        self.spool_dir = Path(spool_dir)
        self.retention_seconds = retention_seconds
        self.cleanup_interval = cleanup_interval
        self.min_segments_to_keep = min_segments_to_keep
        self.retention_safety_enabled = retention_safety_enabled
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # V6: Processor progress tracking for retention safety
        self._last_processed_frame: int = 0
        self._progress_lock = threading.Lock()
        
        # Statistics
        self.segments_deleted: int = 0
        self.bytes_recovered: int = 0
        self.last_cleanup_time: float = 0.0
        self.segments_protected_by_progress: int = 0  # V6: Track protected segments
    
    def set_last_processed_frame(self, frame_index: int):
        """
        V6: Update the last processed frame index.
        
        Retention will never delete segments containing frames beyond this index.
        This ensures processor progress is respected and unprocessed data is preserved.
        
        Args:
            frame_index: Frame index of the last fully processed frame
        """
        with self._progress_lock:
            if frame_index > self._last_processed_frame:
                self._last_processed_frame = frame_index
    
    def get_last_processed_frame(self) -> int:
        """Get the last processed frame index."""
        with self._progress_lock:
            return self._last_processed_frame
    
    def list_segments(self) -> List[Tuple[int, Path, float, int]]:
        """
        List all segments with metadata.
        
        Returns:
            List of tuples (segment_num, path, mtime, size)
        """
        segments = []
        for f in self.spool_dir.glob("seg_*.bin"):
            try:
                num = int(f.stem.split('_')[1])
                stat = f.stat()
                segments.append((num, f, stat.st_mtime, stat.st_size))
            except (IndexError, ValueError, OSError):
                continue
        return sorted(segments, key=lambda x: x[0])
    
    def _get_segment_frame_range(self, segment_path: Path) -> Optional[Tuple[int, int]]:
        """
        V6: Get the frame range for a segment from its metadata.
        
        Returns:
            Tuple of (start_frame, end_frame) or None if not available
        """
        meta_path = segment_path.with_suffix('.meta.json')
        if not meta_path.exists():
            return None
        
        try:
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            return (meta.get('start_frame', 0), meta.get('end_frame', 0))
        except Exception:
            return None
    
    def get_expired_segments(self) -> List[Tuple[int, Path, int]]:
        """
        Get list of segments that have exceeded retention.
        
        V6: Respects processor progress - segments containing unprocessed frames
        are protected from deletion regardless of age.
        
        Returns:
            List of tuples (segment_num, path, size) for expired segments
        """
        current_time = time.time()
        segments = self.list_segments()
        
        # Always keep minimum segments
        if len(segments) <= self.min_segments_to_keep:
            return []
        
        # V6: Get last processed frame for safety check
        last_processed = self.get_last_processed_frame()
        
        # Find expired segments (exclude newest min_segments_to_keep)
        expired = []
        for seg_num, path, mtime, size in segments[:-self.min_segments_to_keep]:
            age = current_time - mtime
            if age > self.retention_seconds:
                # V6: Check if segment contains unprocessed frames
                if self.retention_safety_enabled and last_processed > 0:
                    frame_range = self._get_segment_frame_range(path)
                    if frame_range is not None:
                        start_frame, end_frame = frame_range
                        if end_frame > last_processed:
                            # Segment contains unprocessed frames - protect it
                            logger.debug(
                                f"[Retention] Protected segment {seg_num}: contains unprocessed frames "
                                f"(segment_end={end_frame}, last_processed={last_processed})"
                            )
                            self.segments_protected_by_progress += 1
                            continue
                
                expired.append((seg_num, path, size))
        
        return expired
    
    def cleanup_once(self) -> Tuple[int, int]:
        """
        Perform a single cleanup pass.
        
        Returns:
            Tuple of (segments_deleted, bytes_recovered)
        """
        expired = self.get_expired_segments()
        
        deleted = 0
        bytes_freed = 0
        
        for seg_num, path, size in expired:
            try:
                # Delete segment file
                path.unlink()
                deleted += 1
                bytes_freed += size
                
                # Also delete metadata file if exists
                meta_path = path.with_suffix('.meta.json')
                if meta_path.exists():
                    try:
                        meta_size = meta_path.stat().st_size
                        meta_path.unlink()
                        bytes_freed += meta_size
                    except OSError:
                        pass
                
                logger.info(f"[Retention] Deleted expired segment {seg_num} "
                           f"(size: {size / 1024:.1f} KB)")
                
            except Exception as e:
                logger.warning(f"[Retention] Error deleting segment {seg_num}: {e}")
        
        self.segments_deleted += deleted
        self.bytes_recovered += bytes_freed
        self.last_cleanup_time = time.time()
        
        return deleted, bytes_freed
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        logger.info(f"[Retention] Background cleanup started "
                   f"(interval: {self.cleanup_interval}s, retention: {self.retention_seconds}s)")
        
        while not self._stop_event.wait(self.cleanup_interval):
            try:
                deleted, bytes_freed = self.cleanup_once()
                if deleted > 0:
                    logger.info(f"[Retention] Cleanup pass: deleted {deleted} segments, "
                               f"freed {bytes_freed / 1024:.1f} KB")
            except Exception as e:
                logger.error(f"[Retention] Error in cleanup loop: {e}")
        
        logger.info("[Retention] Background cleanup stopped")
    
    def start(self):
        """Start background cleanup thread."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="RetentionCleanup"
        )
        self._thread.start()
    
    def stop(self):
        """Stop background cleanup thread."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("[Retention] Cleanup thread did not stop in time")
        
        self._thread = None
    
    def get_stats(self) -> dict:
        """
        Get retention statistics.
        
        Returns:
            Dictionary with retention statistics
        """
        segments = self.list_segments()
        total_size = sum(s[3] for s in segments)
        
        oldest_age = 0.0
        if segments:
            oldest_age = time.time() - segments[0][2]
        
        return {
            'total_segments': len(segments),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_segment_age_seconds': oldest_age,
            'retention_seconds': self.retention_seconds,
            'segments_deleted': self.segments_deleted,
            'bytes_recovered': self.bytes_recovered,
            'bytes_recovered_mb': self.bytes_recovered / (1024 * 1024),
            'last_cleanup_time': self.last_cleanup_time,
            # V6: Retention safety stats
            'retention_safety_enabled': self.retention_safety_enabled,
            'last_processed_frame': self.get_last_processed_frame(),
            'segments_protected_by_progress': self.segments_protected_by_progress,
        }


def get_spool_disk_usage(spool_dir: str) -> dict:
    """
    Get disk usage statistics for the spool directory.
    
    Args:
        spool_dir: Path to spool directory
        
    Returns:
        Dictionary with disk usage statistics
    """
    spool_path = Path(spool_dir)
    
    if not spool_path.exists():
        return {
            'exists': False,
            'total_bytes': 0,
            'segment_count': 0,
            'tmp_count': 0,
        }
    
    segment_files = list(spool_path.glob("seg_*.bin"))
    tmp_files = list(spool_path.glob("seg_*.tmp"))
    meta_files = list(spool_path.glob("seg_*.meta.json"))
    
    segment_size = sum(f.stat().st_size for f in segment_files)
    tmp_size = sum(f.stat().st_size for f in tmp_files)
    meta_size = sum(f.stat().st_size for f in meta_files)
    
    total_size = segment_size + tmp_size + meta_size
    
    return {
        'exists': True,
        'total_bytes': total_size,
        'total_mb': total_size / (1024 * 1024),
        'segment_count': len(segment_files),
        'segment_bytes': segment_size,
        'segment_mb': segment_size / (1024 * 1024),
        'tmp_count': len(tmp_files),
        'tmp_bytes': tmp_size,
        'meta_count': len(meta_files),
        'meta_bytes': meta_size,
    }
