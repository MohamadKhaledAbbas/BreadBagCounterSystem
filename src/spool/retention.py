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
        retention_seconds: float = 300.0,  # V7.1: Increased from 180s to 300s (5 minutes)
        cleanup_interval: float = 10.0,
        min_segments_to_keep: int = 2,
        retention_safety_enabled: bool = True,
        max_spool_size_bytes: int = 2_147_483_648,  # V7.1: 2GB size limit
        delete_processed_segments: bool = True  # V7.1: Delete immediately after processing
    ):
        """
        Initialize the retention policy.
        
        Args:
            spool_dir: Directory containing spool files
            retention_seconds: Maximum age of segments before deletion (default: 300s = 5min)
            cleanup_interval: Interval between cleanup checks
            min_segments_to_keep: Minimum segments to always keep
            retention_safety_enabled: V6 - Enable processor progress awareness
            max_spool_size_bytes: V7.1 - Maximum total spool size (default: 2GB)
            delete_processed_segments: V7.1 - Delete segments immediately after processing
        """
        self.spool_dir = Path(spool_dir)
        self.retention_seconds = retention_seconds
        self.cleanup_interval = cleanup_interval
        self.min_segments_to_keep = min_segments_to_keep
        self.retention_safety_enabled = retention_safety_enabled
        self.max_spool_size_bytes = max_spool_size_bytes
        self.delete_processed_segments = delete_processed_segments
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # V6: Processor progress tracking for retention safety
        self._last_processed_frame: int = 0
        self._last_processed_segment: int = -1  # V7.1: Track processed segment for immediate deletion
        self._progress_lock = threading.Lock()
        
        # Statistics
        self.segments_deleted: int = 0
        self.bytes_recovered: int = 0
        self.last_cleanup_time: float = 0.0
        self.segments_protected_by_progress: int = 0  # V6: Track protected segments
        self.segments_deleted_by_processing: int = 0  # V7.1: Track immediate deletions
    
    def set_last_processed_segment(self, segment_num: int):
        """
        V7.1: Update the last processed segment number.
        
        When delete_processed_segments is enabled, this will trigger immediate
        deletion of the segment that was just fully processed.
        
        Args:
            segment_num: Segment number that was just fully processed
        """
        with self._progress_lock:
            if segment_num > self._last_processed_segment:
                old_segment = self._last_processed_segment
                self._last_processed_segment = segment_num
                
                # If enabled, delete the old segment immediately
                if self.delete_processed_segments and old_segment >= 0:
                    self._delete_processed_segment(old_segment)
    
    def _delete_processed_segment(self, segment_num: int):
        """Delete a processed segment immediately."""
        try:
            segment_file = self.spool_dir / f"seg_{segment_num:010d}.bin"
            if segment_file.exists():
                size = segment_file.stat().st_size
                segment_file.unlink()
                self.bytes_recovered += size
                self.segments_deleted += 1
                self.segments_deleted_by_processing += 1
                logger.info(f"[Retention] Deleted processed segment {segment_num} ({size / 1024 / 1024:.2f}MB)")
        except Exception as e:
            logger.warning(f"[Retention] Error deleting processed segment {segment_num}: {e}")
    
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
        
        V7.1: Now includes size-based cleanup in addition to age-based.
        
        Returns:
            Tuple of (segments_deleted, bytes_recovered)
        """
        deleted_count = 0
        bytes_freed = 0
        
        # V7.1: Check total spool size
        total_size = self._get_total_spool_size()
        size_exceeded = total_size > self.max_spool_size_bytes
        
        if size_exceeded:
            logger.warning(f"[Retention] Spool size ({total_size / 1024 / 1024:.2f}MB) exceeds limit "
                          f"({self.max_spool_size_bytes / 1024 / 1024:.2f}MB) - aggressive cleanup")
        
        # Get expired segments (age-based)
        expired = self.get_expired_segments()
        
        # V7.1: If size exceeded, also delete oldest processed segments even if not expired
        if size_exceeded:
            expired.extend(self._get_oldest_processed_segments(exclude=expired))
        
        # Delete expired segments
        for seg_num, path, size in expired:
            try:
                path.unlink()
                deleted_count += 1
                bytes_freed += size
                self.segments_deleted += 1
                self.bytes_recovered += bytes_freed
                
                logger.info(
                    f"[Retention] Deleted segment {seg_num} "
                    f"({size / 1024 / 1024:.2f}MB, age: {time.time() - path.stat().st_mtime:.1f}s)"
                )
                
                # Check if we've freed enough space
                if size_exceeded and total_size - bytes_freed <= self.max_spool_size_bytes:
                    break
                    
            except Exception as e:
                logger.warning(f"[Retention] Error deleting segment {seg_num}: {e}")
        
        self.last_cleanup_time = time.time()
        return deleted_count, bytes_freed
    
    def _get_total_spool_size(self) -> int:
        """V7.1: Calculate total size of all segment files."""
        total = 0
        for seg_num, path, mtime, size in self.list_segments():
            total += size
        return total
    
    def _get_oldest_processed_segments(self, exclude: List[Tuple[int, Path, int]]) -> List[Tuple[int, Path, int]]:
        """
        V7.1: Get oldest segments that have been processed (for size-based cleanup).
        
        Args:
            exclude: List of segments already marked for deletion
            
        Returns:
            List of tuples (segment_num, path, size) for oldest processed segments
        """
        excluded_nums = {seg_num for seg_num, _, _ in exclude}
        last_processed_seg = self._last_processed_segment
        
        # Get all segments older than last processed
        segments = self.list_segments()
        processed_segments = []
        
        for seg_num, path, mtime, size in segments:
            if seg_num < last_processed_seg and seg_num not in excluded_nums:
                processed_segments.append((seg_num, path, size))
        
        # Sort by age (oldest first)
        processed_segments.sort(key=lambda x: x[1].stat().st_mtime)
        
        return processed_segments
    
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
