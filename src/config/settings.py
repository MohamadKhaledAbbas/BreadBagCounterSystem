"""
Application Configuration Module V2.

V2 Enhancements:
- Model version tracking for experiment management
- Recording directory configuration
- Environment-based configuration overrides

V3 Enhancements:
- Testing mode for OpenCV frame source on slower machines
"""

import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from src.utils.platform import IS_RDK


def _parse_bool_env(env_var: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    value = os.getenv(env_var)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


@dataclass
class ModelInfo:
    """
    V2: Model metadata for version tracking and experiment management.
    
    Tracks model paths, versions, and checksums for reproducibility.
    """
    path: str
    version: str = "unknown"
    checksum: Optional[str] = None
    loaded_at: Optional[datetime] = None
    
    def compute_checksum(self) -> Optional[str]:
        """Compute MD5 checksum of model file for version verification."""
        if not os.path.exists(self.path):
            return None
        
        try:
            md5 = hashlib.md5()
            with open(self.path, 'rb') as f:
                # Read in chunks for large files
                for chunk in iter(lambda: f.read(8192), b''):
                    md5.update(chunk)
            self.checksum = md5.hexdigest()[:12]  # First 12 chars
            return self.checksum
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Return model info as dictionary for logging."""
        return {
            "path": self.path,
            "version": self.version,
            "checksum": self.checksum,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
        }


@dataclass
class AppConfig:
    """
    Application configuration with V2 model version tracking.
    """

    APP_VERSION: str = "2025-12-13-v1.0.0"

    video_path: str = os.getenv("VIDEO_PATH", "D:\\Recordings\\New_Recordings\\Brown_Orange_Overlay_20251128010005_20251128011157.mp4")

    # Platform-specific model paths
    # RDK uses .bin models optimized for BPU, Windows/other platforms use .pt or .onnx models
    detection_model: str = os.getenv(
        "DETECTION_MODEL",
        "data/model/detect_yolo_small_v5_bayese_640x640_nv12.bin" if IS_RDK
        else "data/model/detect_yolo_small_v5.pt"
    )
    classification_model: str = os.getenv(
        "CLASS_MODEL",
        "data/model/classify_yolo_small_v5_bayese_224x224_nv12.bin" if IS_RDK
        else "data/model/classify_yolo_small_v5.pt"
    )

    db_path: str = os.getenv("DB_PATH", "data/db/bag_events.db")
    
    # V2: Recording and snapshot directories
    recording_dir: str = os.getenv("RECORDING_DIR", "data/recordings")
    
    # V2: Model version identifiers
    detection_model_version: str = os.getenv("DETECTION_MODEL_VERSION", "v5.0")
    classification_model_version: str = os.getenv("CLASS_MODEL_VERSION", "v5.0")
    
    # V3: Testing mode for OpenCV frame source
    # When enabled, frames are read synchronously on-demand (no background thread)
    # This prevents frame drops and resource exhaustion on slower machines
    # Set OPENCV_TESTING_MODE=true to enable, or it auto-enables in development mode on non-RDK
    opencv_testing_mode: bool = field(default_factory=lambda: _parse_bool_env("OPENCV_TESTING_MODE", False))
    
    # Classifier class names
    classifier_classes: dict = None
    
    # Detector class names  
    detector_classes: dict = None
    
    # V2: Model info objects (populated in __post_init__)
    detection_model_info: Optional[ModelInfo] = None
    classification_model_info: Optional[ModelInfo] = None
    
    def __post_init__(self):
        if self.classifier_classes is None:
            self.classifier_classes = {
                0: 'Blue_Yellow', 
                1: 'Bran', 
                2: 'Brown_Orange_Overlay', 
                3: 'Brown_Orange_Small', 
                4: 'Green_Yellow', 
                5: 'Red_Yellow', 
                6: 'Wheatberry'
            }
        if self.detector_classes is None:
            self.detector_classes = {
                0: 'bread-bag-closed', 
                1: 'bread-bag-opened'
            }
        
        # V2: Initialize model info
        self.detection_model_info = ModelInfo(
            path=self.detection_model,
            version=self.detection_model_version,
            loaded_at=datetime.now()
        )
        self.detection_model_info.compute_checksum()
        
        self.classification_model_info = ModelInfo(
            path=self.classification_model,
            version=self.classification_model_version,
            loaded_at=datetime.now()
        )
        self.classification_model_info.compute_checksum()
    
    def get_model_versions(self) -> Dict[str, Any]:
        """V2: Return model version info for logging and tracking."""
        return {
            "detection": self.detection_model_info.to_dict() if self.detection_model_info else {},
            "classification": self.classification_model_info.to_dict() if self.classification_model_info else {},
            "platform": "RDK" if IS_RDK else "Standard",
        }
    
    def log_configuration(self):
        """V2: Log current configuration for debugging."""
        from src.utils.AppLogging import logger
        
        logger.info("[AppConfig] === Configuration Summary ===")
        logger.info(f"[AppConfig] App Version: {self.APP_VERSION}")
        logger.info(f"[AppConfig] Platform: {'RDK' if IS_RDK else 'Standard'}")
        logger.info(f"[AppConfig] Detection Model: {self.detection_model}")
        logger.info(f"[AppConfig] Detection Version: {self.detection_model_version}")
        if self.detection_model_info and self.detection_model_info.checksum:
            logger.info(f"[AppConfig] Detection Checksum: {self.detection_model_info.checksum}")
        logger.info(f"[AppConfig] Classification Model: {self.classification_model}")
        logger.info(f"[AppConfig] Classification Version: {self.classification_model_version}")
        if self.classification_model_info and self.classification_model_info.checksum:
            logger.info(f"[AppConfig] Classification Checksum: {self.classification_model_info.checksum}")
        logger.info(f"[AppConfig] Database: {self.db_path}")
        logger.info(f"[AppConfig] Recording Dir: {self.recording_dir}")
        logger.info(f"[AppConfig] OpenCV Testing Mode: {self.opencv_testing_mode}")


config = AppConfig()
