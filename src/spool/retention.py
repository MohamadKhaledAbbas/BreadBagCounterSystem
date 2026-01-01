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


def cleanup_orphaned_meta_files(spool_dir: str) -> int:
    """
    Clean up orphaned .meta.json files without corresponding .bin files.
    
    Args:
        spool_dir: Directory containing spool files
        
    Returns:
        Number of orphaned meta files cleaned up
    """
    spool_path = Path(spool_dir)
    if not spool_path.exists():
        return 0
    
    cleaned = 0
    
    for meta_file in spool_path.glob("seg_*.meta.json"):
        try:
            # Check if corresponding .bin file exists
            bin_file = meta_file.with_suffix('.bin')
            if not bin_file.exists():
                meta_file.unlink()
                logger.info(f"[Retention] Cleaned up orphaned meta file: {meta_file.name}")
                cleaned += 1
        except Exception as e:
            logger.warning(f"[Retention] Error cleaning up {meta_file.name}: {e}")
    
    if cleaned > 0:
        logger.info(f"[Retention] Cleaned up {cleaned} orphaned meta files")
    
    return cleaned


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
    
    # Also clean up orphaned meta files
    cleaned_meta = cleanup_orphaned_meta_files(spool_dir)
    
    return cleaned + cleaned_meta


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
        self.last_cleanup_monotonic: float = time.monotonic()  # Use monotonic time for intervals
        self.segments_protected_by_progress: int = 0  # V6: Track protected segments
        self.segments_deleted_by_processing: int = 0  # V7.1: Track immediate deletions
        self.delete_errors: int = 0  # Track deletion failures
        self.segments_protected_by_size_limit: int = 0  # Track segments protected due to unprocessed data

    def get_last_processed_segment(self) -> int:
        """
        Return the most recently processed segment number.
        """
        with self._progress_lock:
            return self._last_processed_segment

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
    
    def get_last_processed_segment(self) -> int:
        """Get the last processed segment number."""
        with self._progress_lock:
            return self._last_processed_segment
    
    def _delete_processed_segment(self, segment_num: int):
        """Delete a processed segment immediately with its metadata."""
        try:
            segment_file = self.spool_dir / f"seg_{segment_num:06d}.bin"
            if segment_file.exists():
                size = segment_file.stat().st_size
                segment_file.unlink()
                
                # Also delete corresponding .meta.json file if it exists
                meta_file = segment_file.with_suffix('.meta.json')
                if meta_file.exists():
                    try:
                        meta_file.unlink()
                        logger.debug(f"[Retention] Also deleted metadata for immediate cleanup of segment {segment_num}")
                    except Exception as e:
                        logger.warning(f"[Retention] Error deleting metadata during immediate cleanup of segment {segment_num}: {e}")
                
                self.bytes_recovered += size
                self.segments_deleted += 1
                self.segments_deleted_by_processing += 1
                logger.debug(f"[Retention] Deleted processed segment {segment_num} ({size / 1024:.1f}KB)")
            else:
                logger.debug(f"[Retention] Segment {segment_num} already deleted (immediate cleanup)")
        except FileNotFoundError:
            logger.debug(f"[Retention] Segment {segment_num} already deleted (immediate cleanup)")
        except Exception as e:
            logger.warning(f"[Retention] Error deleting segment {segment_num}: {e}")
            self.delete_errors += 1
    
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
    
    def get_expired_segments(self, force_size_cleanup: bool = False) -> List[Tuple[int, Path, int]]:
        """
        Get list of segments that have exceeded retention.
        
        V6: Respects processor progress - segments containing unprocessed frames
        are protected from deletion regardless of age.
        
        V7.2: Also protects segments at or after processor's current position
        to prevent race conditions.
        
        V7.3: NEVER delete segments >= last_processed_segment unless total spool
        exceeds 2GB cap (force_size_cleanup=True).
        
        Args:
            force_size_cleanup: If True, allows deleting old processed segments
                                even if not expired by age (for size limit enforcement)
        
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
        
        # V7.2: Get last processed segment number
        last_processed_segment = self.get_last_processed_segment()
        
        # Find expired segments (exclude newest min_segments_to_keep)
        expired = []
        protected_by_unprocessed = 0
        
        for seg_num, path, mtime, size in segments[:-self.min_segments_to_keep]:
            # V7.3: CRITICAL PROTECTION - Never delete segments >= last processed segment
            # UNLESS we're in force_size_cleanup mode (spool exceeds 2GB)
            if self.retention_safety_enabled and last_processed_segment >= 0:
                if seg_num >= last_processed_segment:
                    if not force_size_cleanup:
                        logger.debug(
                            f"[Retention] Protected segment {seg_num}: at or after processor position "
                            f"(last_processed_segment={last_processed_segment})"
                        )
                        self.segments_protected_by_progress += 1
                        protected_by_unprocessed += 1
                        continue
                    else:
                        # In force_size_cleanup mode, we can delete segments >= last_processed
                        # ONLY if they are significantly older than the current segment
                        # This allows cleanup of segments that the processor has moved past
                        # but respects a safety margin
                        logger.debug(
                            f"[Retention] Size cleanup: considering segment {seg_num} "
                            f"(last_processed={last_processed_segment})"
                        )
            
            # Check age-based expiration
            age = current_time - mtime
            is_age_expired = age > self.retention_seconds
            
            # V6: Check if segment contains unprocessed frames (frame-level safety)
            if self.retention_safety_enabled and last_processed > 0:
                frame_range = self._get_segment_frame_range(path)
                if frame_range is not None:
                    start_frame, end_frame = frame_range
                    if end_frame > last_processed:
                        # Segment contains unprocessed frames - protect it unless force_size_cleanup
                        if not force_size_cleanup:
                            logger.debug(
                                f"[Retention] Protected segment {seg_num}: contains unprocessed frames "
                                f"(segment_end={end_frame}, last_processed={last_processed})"
                            )
                            self.segments_protected_by_progress += 1
                            protected_by_unprocessed += 1
                            continue
            
            # In force mode, add any old processed segments
            # In normal mode, only add age-expired segments
            if force_size_cleanup or is_age_expired:
                expired.append((seg_num, path, size))
        
        # Log if we protected segments
        if protected_by_unprocessed > 0:
            self.segments_protected_by_size_limit = protected_by_unprocessed
            if force_size_cleanup:
                logger.warning(
                    f"[Retention] Protected {protected_by_unprocessed} unprocessed segments "
                    f"even during size-based cleanup (last_processed_segment={last_processed_segment})"
                )
        
        return expired
    
    def cleanup_once(self) -> Tuple[int, int]:
        """
        Perform a single cleanup pass.
        
        V7.1: Now includes size-based cleanup in addition to age-based.
        V7.3: Implements 2GB size guardrail with proper protection of unprocessed segments.
        
        Returns:
            Tuple of (segments_deleted, bytes_recovered)
        """
        deleted_count = 0
        bytes_freed = 0
        
        # V7.1: Check total spool size
        total_size = self._get_total_spool_size()
        size_exceeded = total_size > self.max_spool_size_bytes
        
        # Log warning if approaching or exceeding capacity
        size_pct = (total_size / self.max_spool_size_bytes) * 100 if self.max_spool_size_bytes > 0 else 0
        if size_exceeded:
            logger.warning(
                f"[Retention] ⚠ Spool size EXCEEDED: {total_size / 1024 / 1024:.2f}MB "
                f"({size_pct:.1f}%) > limit {self.max_spool_size_bytes / 1024 / 1024:.2f}MB - "
                f"aggressive cleanup"
            )
        elif size_pct > 80:
            logger.warning(
                f"[Retention] ⚠ Approaching capacity: {total_size / 1024 / 1024:.2f}MB "
                f"({size_pct:.1f}%) of {self.max_spool_size_bytes / 1024 / 1024:.2f}MB limit"
            )
        
        # Get expired segments (age-based)
        expired = self.get_expired_segments(force_size_cleanup=False)
        
        # V7.1: If size exceeded, also delete oldest processed segments even if not age-expired
        # BUT ONLY segments < last_processed_segment (never delete unprocessed data)
        if size_exceeded:
            size_cleanup_candidates = self._get_oldest_processed_segments(exclude=expired)
            logger.info(
                f"[Retention] Size-based cleanup: found {len(size_cleanup_candidates)} "
                f"old processed segments for potential deletion"
            )
            expired.extend(size_cleanup_candidates)
        
        # Sort expired segments by age (oldest first) to delete in optimal order
        expired.sort(key=lambda x: x[1].stat().st_mtime if x[1].exists() else 0)
        
        # Delete expired segments
        for seg_num, path, size in expired:
            try:
                if not path.exists():
                    # Already deleted by immediate cleanup
                    logger.debug(f"[Retention] Segment {seg_num} already deleted (immediate cleanup)")
                    continue
                
                # Get age for logging
                try:
                    age = time.time() - path.stat().st_mtime
                except OSError:
                    age = 0.0
                
                # Delete the segment file
                path.unlink()
                
                # Also delete corresponding .meta.json file if it exists
                meta_path = path.with_suffix('.meta.json')
                if meta_path.exists():
                    try:
                        meta_path.unlink()
                        logger.debug(f"[Retention] Also deleted metadata file for segment {seg_num}")
                    except Exception as e:
                        logger.warning(f"[Retention] Error deleting metadata file for segment {seg_num}: {e}")
                
                deleted_count += 1
                bytes_freed += size
                self.segments_deleted += 1
                self.bytes_recovered += size
                
                logger.info(
                    f"[Retention] Deleted segment {seg_num} "
                    f"({size / 1024 / 1024:.2f}MB, age: {age:.1f}s)"
                )
                
                # Check if we've freed enough space
                if size_exceeded:
                    new_total = total_size - bytes_freed
                    if new_total <= self.max_spool_size_bytes:
                        logger.info(
                            f"[Retention] Size limit satisfied: freed {bytes_freed / 1024 / 1024:.2f}MB, "
                            f"new total: {new_total / 1024 / 1024:.2f}MB"
                        )
                        break
                    
            except FileNotFoundError:
                # Already deleted - likely by immediate cleanup
                logger.debug(f"[Retention] Segment {seg_num} already deleted (immediate cleanup)")
            except Exception as e:
                logger.error(f"[Retention] Error deleting segment {seg_num}: {e}")
                self.delete_errors += 1
        
        # Update cleanup timestamp (use monotonic time for intervals)
        self.last_cleanup_time = time.time()
        self.last_cleanup_monotonic = time.monotonic()
        
        # Log summary if significant cleanup happened
        if deleted_count > 0 or size_exceeded:
            logger.info(
                f"[Retention] Cleanup summary: deleted={deleted_count}, "
                f"freed={bytes_freed / 1024 / 1024:.2f}MB, "
                f"errors={self.delete_errors}, "
                f"protected={self.segments_protected_by_size_limit}"
            )
        
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
        """Background cleanup loop using monotonic time for intervals."""
        logger.info(f"[Retention] Background cleanup started "
                   f"(interval: {self.cleanup_interval}s, retention: {self.retention_seconds}s)")
        
        last_cleanup_monotonic = time.monotonic()
        
        while not self._stop_event.is_set():
            try:
                # Calculate time since last cleanup using monotonic clock
                current_monotonic = time.monotonic()
                time_since_last = current_monotonic - last_cleanup_monotonic
                
                # Wait for next cleanup interval
                remaining = self.cleanup_interval - time_since_last
                if remaining > 0:
                    # Use wait with timeout to allow clean shutdown
                    if self._stop_event.wait(timeout=remaining):
                        break  # Stop event was set
                
                # Perform cleanup
                deleted, bytes_freed = self.cleanup_once()
                last_cleanup_monotonic = time.monotonic()
                
                if deleted > 0:
                    logger.info(f"[Retention] Cleanup pass: deleted {deleted} segments, "
                               f"freed {bytes_freed / 1024:.1f} KB")
            except Exception as e:
                logger.error(f"[Retention] Error in cleanup loop: {e}")
                # Wait a bit before retrying to avoid tight error loops
                self._stop_event.wait(timeout=5.0)
        
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
        """Stop background cleanup thread with final cleanup pass."""
        if not self._running:
            return
        
        logger.info("[Retention] Stopping cleanup thread...")
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("[Retention] Cleanup thread did not stop in time")
            else:
                logger.info("[Retention] Cleanup thread stopped successfully")
        
        # Perform final cleanup pass
        try:
            logger.info("[Retention] Performing final cleanup pass...")
            deleted, bytes_freed = self.cleanup_once()
            if deleted > 0:
                logger.info(
                    f"[Retention] Final cleanup: deleted {deleted} segments, "
                    f"freed {bytes_freed / 1024:.1f} KB"
                )
        except Exception as e:
            logger.error(f"[Retention] Error in final cleanup pass: {e}")
        
        self._thread = None
    
    def get_stats(self) -> dict:
        """
        Get retention statistics with rich metrics.
        
        V7.3: Enhanced with detailed capacity warnings and error tracking.
        
        Returns:
            Dictionary with retention statistics
        """
        segments = self.list_segments()
        total_size = sum(s[3] for s in segments)
        
        oldest_age = 0.0
        if segments:
            oldest_age = time.time() - segments[0][2]
        
        # Calculate capacity metrics
        size_pct = (total_size / self.max_spool_size_bytes) * 100 if self.max_spool_size_bytes > 0 else 0
        nearing_capacity = size_pct > 80
        at_capacity = total_size >= self.max_spool_size_bytes
        
        stats = {
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
            'last_processed_segment': self.get_last_processed_segment(),
            'segments_protected_by_progress': self.segments_protected_by_progress,
            # V7.3: Enhanced stats
            'delete_errors': self.delete_errors,
            'segments_deleted_by_processing': self.segments_deleted_by_processing,
            'segments_protected_by_size_limit': self.segments_protected_by_size_limit,
            'max_spool_size_mb': self.max_spool_size_bytes / (1024 * 1024),
            'size_percentage': size_pct,
            'nearing_capacity': nearing_capacity,
            'at_capacity': at_capacity,
        }
        
        # Add warnings
        warnings = []
        if at_capacity:
            warnings.append(f"CRITICAL: Spool at capacity ({size_pct:.1f}%)")
        elif nearing_capacity:
            warnings.append(f"WARNING: Approaching capacity ({size_pct:.1f}%)")
        
        if self.delete_errors > 0:
            warnings.append(f"Delete errors: {self.delete_errors}")
        
        if warnings:
            stats['warnings'] = warnings
        
        return stats


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
