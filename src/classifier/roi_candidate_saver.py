"""
ROI Candidate Saver Module for Debug/Analysis.

This module provides functionality to save all ROI candidates with metadata
for post-analysis and model improvement.

PROBLEM:
No visibility into why classification fails or what ROIs were collected:
- Can't analyze "Rejected" or "Uncertain" results
- Can't verify ROI quality metrics
- Can't improve model with real production data
- Blind to data quality issues

SOLUTION:
Save all ROI candidates with metadata for post-analysis.

Directory structure:
    data/roi_candidates/
    ├── Brown_Orange_Small/
    │   ├── track_12345_roi_0_quality_0.85.jpg
    │   ├── track_12345_roi_1_quality_0.78.jpg
    │   └── track_12345_metadata.json
    ├── Brown_Orange_Large/
    │   └── ...
    ├── Rejected/
    │   └── track_67890_roi_0_quality_0.45.jpg
    └── Uncertain/
        └── track_11111_roi_0_quality_0.62.jpg

Configuration (environment variables):
    SAVE_ROI_CANDIDATES=true              # Enable/disable ROI candidate saving
    ROI_CANDIDATES_DIR="data/roi_candidates"  # Directory for saved ROI candidates
    SAVE_REJECTED_TRACKS=true             # Save rejected/uncertain tracks
    SAVE_UNCERTAIN_TRACKS=true

BENEFITS:
✅ Debug "Uncertain" results: See exactly what ROIs were collected
✅ Analyze "Rejected" cases: Understand why classifier rejects
✅ Model improvement: Use real production data for retraining
✅ Quality verification: Validate sharpness/quality metrics
✅ Production monitoring: Track data quality over time
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.utils.AppLogging import logger


def _parse_bool_env(env_var: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    value = os.getenv(env_var)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def _parse_str_env(env_var: str, default: str) -> str:
    """Parse string from environment variable."""
    return os.getenv(env_var, default)


class ROICandidateSaverConfig:
    """Configuration for ROI candidate saving."""
    
    def __init__(self):
        # Enable/disable ROI candidate saving (production: false, debug: true)
        self.enabled = _parse_bool_env('SAVE_ROI_CANDIDATES', False)
        
        # Directory for saved ROI candidates
        self.output_dir = _parse_str_env('ROI_CANDIDATES_DIR', 'data/roi_candidates')
        
        # Save rejected tracks (debug analysis)
        self.save_rejected_tracks = _parse_bool_env('SAVE_REJECTED_TRACKS', True)
        
        # Save uncertain tracks (debug analysis)
        self.save_uncertain_tracks = _parse_bool_env('SAVE_UNCERTAIN_TRACKS', True)
        
        # Maximum ROI candidates to save per track (limit disk usage)
        self.max_rois_per_track = int(os.getenv('MAX_ROIS_PER_TRACK', '20'))
        
        # Save images in JPEG with configurable quality (balance size vs quality)
        self.jpeg_quality = int(os.getenv('ROI_JPEG_QUALITY', '85'))
        
        # Enable homography info in metadata if available
        self.include_homography_info = _parse_bool_env('ROI_INCLUDE_HOMOGRAPHY', True)


class ROICandidateSaver:
    """
    Saves ROI candidates with metadata for post-analysis.
    
    This class is responsible for:
    1. Organizing ROIs by classification result (directories per class)
    2. Saving ROI images with quality scores in filename
    3. Creating metadata JSON files per track
    4. Managing disk space with configurable limits
    """
    
    def __init__(self, config: Optional[ROICandidateSaverConfig] = None):
        """
        Initialize the ROI candidate saver.
        
        Args:
            config: Configuration object. If None, creates from environment.
        """
        self.config = config or ROICandidateSaverConfig()
        
        if self.config.enabled:
            self._ensure_output_directory()
            logger.info(
                f"[ROICandidateSaver] Initialized: "
                f"output_dir={self.config.output_dir}, "
                f"max_rois_per_track={self.config.max_rois_per_track}, "
                f"jpeg_quality={self.config.jpeg_quality}"
            )
        else:
            logger.info("[ROICandidateSaver] Disabled (set SAVE_ROI_CANDIDATES=true to enable)")
    
    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"[ROICandidateSaver] Failed to create output directory: {e}")
            self.config.enabled = False
    
    def _get_class_directory(self, classification: str) -> str:
        """Get or create directory for a specific classification."""
        # Sanitize class name for filesystem
        safe_name = classification.replace('/', '_').replace('\\', '_').replace(' ', '_')
        class_dir = os.path.join(self.config.output_dir, safe_name)
        
        try:
            os.makedirs(class_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"[ROICandidateSaver] Failed to create class directory: {e}")
            return self.config.output_dir
        
        return class_dir
    
    def _should_save_track(self, classification: str) -> bool:
        """Determine if a track should be saved based on classification."""
        if not self.config.enabled:
            return False
        
        # Check if rejected tracks should be saved
        if classification.lower() == 'rejected' and not self.config.save_rejected_tracks:
            return False
        
        # Check if uncertain tracks should be saved
        if classification.lower() in ('uncertain', 'unknown') and not self.config.save_uncertain_tracks:
            return False
        
        return True
    
    def save_track_candidates(
        self,
        track_id: int,
        classification: str,
        confidence: float,
        roi_candidates: List[Dict[str, Any]],
        homography_info: Optional[Dict[str, Any]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Save all ROI candidates for a track with metadata.
        
        Args:
            track_id: Unique track identifier
            classification: Final classification result
            confidence: Classification confidence
            roi_candidates: List of ROI candidate dictionaries, each containing:
                - roi: numpy array (BGR image)
                - sharpness: float
                - quality: float
                - size: Tuple[int, int] (width, height)
                - frame_index: int
                - confidence: float
                - state: str ('open' or 'closed')
                - bbox: Tuple[float, float, float, float] (x1, y1, x2, y2)
            homography_info: Optional homography calibration info for size measurements
            additional_metadata: Any additional metadata to include
            
        Returns:
            Path to metadata file if saved, None otherwise
        """
        if not self._should_save_track(classification):
            return None
        
        if not CV2_AVAILABLE:
            logger.warning("[ROICandidateSaver] cv2 not available, cannot save ROIs")
            return None
        
        class_dir = self._get_class_directory(classification)
        timestamp = datetime.now().isoformat()
        frame_range = self._compute_frame_range(roi_candidates)
        
        # Build metadata structure
        metadata = {
            'track_id': track_id,
            'final_classification': classification,
            'confidence': confidence,
            'timestamp': timestamp,
            'frame_range': frame_range,
            'total_roi_count': len(roi_candidates),
            'roi_candidates': [],
        }
        
        # Add homography info if available
        if homography_info and self.config.include_homography_info:
            metadata['homography_info'] = homography_info
        
        # Add additional metadata
        if additional_metadata:
            metadata['additional'] = additional_metadata
        
        # Save each ROI candidate (up to limit)
        saved_count = 0
        for idx, candidate in enumerate(roi_candidates[:self.config.max_rois_per_track]):
            roi_image = candidate.get('roi')
            if roi_image is None:
                # Handle lazy ROI - try to get it
                if hasattr(candidate.get('roi_candidate'), 'get_roi'):
                    roi_image = candidate['roi_candidate'].get_roi()
            
            if roi_image is None or not isinstance(roi_image, np.ndarray):
                continue
            
            quality = candidate.get('quality', 0.0)
            sharpness = candidate.get('sharpness', 0.0)
            
            # Generate filename with quality score
            filename = f"track_{track_id}_roi_{idx}_quality_{quality:.2f}.jpg"
            filepath = os.path.join(class_dir, filename)
            
            try:
                # Save ROI image with configurable JPEG quality
                cv2.imwrite(
                    filepath, 
                    roi_image, 
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
                saved_count += 1
            except Exception as e:
                logger.error(f"[ROICandidateSaver] Failed to save ROI {idx}: {e}")
                continue
            
            # Build ROI metadata
            size_px = candidate.get('size', (0, 0))
            bbox = candidate.get('bbox')
            
            roi_metadata = {
                'roi_index': idx,
                'filename': filename,
                'frame_index': candidate.get('frame_index', 0),
                'sharpness': sharpness,
                'quality': quality,
                'size_px': list(size_px) if size_px else None,
                'confidence': candidate.get('confidence', 0.0),
                'state': candidate.get('state', 'unknown'),
                'bbox': list(bbox) if bbox else None,
            }
            
            # Add homography-based size if available
            if homography_info and bbox:
                size_cm = homography_info.get('size_cm')
                area_cm2 = homography_info.get('area_cm2')
                if size_cm:
                    roi_metadata['size_cm'] = size_cm
                if area_cm2:
                    roi_metadata['area_cm2'] = area_cm2
            
            metadata['roi_candidates'].append(roi_metadata)
        
        # Save metadata JSON
        metadata_filename = f"track_{track_id}_metadata.json"
        metadata_filepath = os.path.join(class_dir, metadata_filename)
        
        try:
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.debug(
                f"[ROICandidateSaver] Saved track {track_id}: "
                f"{saved_count} ROIs, classification={classification}"
            )
            
            return metadata_filepath
            
        except Exception as e:
            logger.error(f"[ROICandidateSaver] Failed to save metadata: {e}")
            return None
    
    def _compute_frame_range(self, roi_candidates: List[Dict[str, Any]]) -> List[int]:
        """Compute frame range from ROI candidates."""
        frame_indices = [c.get('frame_index', 0) for c in roi_candidates if 'frame_index' in c]
        if not frame_indices:
            return [0, 0]
        return [min(frame_indices), max(frame_indices)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about saved ROI candidates."""
        if not self.config.enabled or not os.path.exists(self.config.output_dir):
            return {'enabled': False, 'total_tracks': 0, 'total_rois': 0}
        
        stats = {
            'enabled': True,
            'output_dir': self.config.output_dir,
            'classes': {},
            'total_tracks': 0,
            'total_rois': 0,
        }
        
        try:
            for class_name in os.listdir(self.config.output_dir):
                class_dir = os.path.join(self.config.output_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                files = os.listdir(class_dir)
                metadata_files = [f for f in files if f.endswith('_metadata.json')]
                roi_files = [f for f in files if f.endswith('.jpg')]
                
                stats['classes'][class_name] = {
                    'tracks': len(metadata_files),
                    'rois': len(roi_files),
                }
                stats['total_tracks'] += len(metadata_files)
                stats['total_rois'] += len(roi_files)
                
        except Exception as e:
            logger.error(f"[ROICandidateSaver] Error computing stats: {e}")
        
        return stats


# Global singleton for ROI candidate saver (lazy initialization)
_saver_instance: Optional[ROICandidateSaver] = None


def get_roi_candidate_saver() -> ROICandidateSaver:
    """
    Get the global ROI candidate saver instance.
    
    Returns:
        ROICandidateSaver instance
    """
    global _saver_instance
    
    if _saver_instance is None:
        _saver_instance = ROICandidateSaver()
    
    return _saver_instance


def save_track_roi_candidates(
    track_id: int,
    classification: str,
    confidence: float,
    roi_candidates: List[Dict[str, Any]],
    homography_info: Optional[Dict[str, Any]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Convenience function to save ROI candidates using global saver.
    
    Args:
        track_id: Unique track identifier
        classification: Final classification result
        confidence: Classification confidence
        roi_candidates: List of ROI candidate dictionaries
        homography_info: Optional homography calibration info
        additional_metadata: Any additional metadata to include
        
    Returns:
        Path to metadata file if saved, None otherwise
    """
    saver = get_roi_candidate_saver()
    return saver.save_track_candidates(
        track_id=track_id,
        classification=classification,
        confidence=confidence,
        roi_candidates=roi_candidates,
        homography_info=homography_info,
        additional_metadata=additional_metadata
    )
