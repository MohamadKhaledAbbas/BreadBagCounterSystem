"""
Homography-Based Size Classification Module.

This module implements homography transformation for accurate size-based
classification of bread bags using the work table as a reference plane.

PROBLEM:
Current size-based classification (Small vs Large bread) relies on bounding box
dimensions from YOLO detections. This approach is unreliable because:
- Perspective distortion: Same object appears different sizes at different distances
- Camera angle variations affect perceived size
- No physical context - pixel size ≠ real-world size

SOLUTION:
Implement homography-based size estimation using the work table as a reference plane:
1. Calibrate table corner positions (one-time setup)
2. Compute homography transformation matrix
3. Transform bounding boxes to bird's-eye view (real-world coordinates)
4. Measure actual bread size in cm²
5. Classify based on physical thresholds (e.g., < 100 cm² = Small)

BENEFITS:
✅ Physically accurate: Measures actual bread size, not perception
✅ Perspective-invariant: Works at any camera distance/angle
✅ Debuggable: Real measurements (cm) vs arbitrary pixels
✅ Lightweight: Pure geometry, ~1ms overhead, no ML model needed
✅ Production-grade: Used in industrial inspection systems worldwide

Usage:
    from src.classifier.homography import HomographyTransform
    
    # Initialize with table corners and physical size
    transform = HomographyTransform(
        table_corners_px=[[150, 100], [950, 120], [980, 650], [120, 680]],
        table_size_cm=(80, 60)  # 80cm x 60cm table
    )
    
    # Get real-world size of a bounding box
    size_cm = transform.get_bbox_size_cm(bbox)
    area_cm2 = size_cm[0] * size_cm[1]
    
    # Classify based on physical threshold
    is_small = area_cm2 < 100  # cm²
"""

import numpy as np
from typing import Tuple, Optional, List
import os

from src.utils.AppLogging import logger


class HomographyTransform:
    """
    Homography transformation for converting pixel coordinates to real-world coordinates.
    
    Uses the work table as a reference plane to compute a perspective transformation
    that maps pixel coordinates to centimeter coordinates.
    """
    
    def __init__(
        self,
        table_corners_px: Optional[List[List[float]]] = None,
        table_size_cm: Optional[Tuple[float, float]] = None,
        enabled: bool = True
    ):
        """
        Initialize the homography transform.
        
        Args:
            table_corners_px: Four corners of the table in pixel coordinates,
                             in clockwise order starting from top-left:
                             [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            table_size_cm: Physical size of the table in centimeters (width, height)
            enabled: Whether homography transformation is enabled
        """
        self.enabled = enabled
        self.table_corners_px = table_corners_px
        self.table_size_cm = table_size_cm
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography_matrix: Optional[np.ndarray] = None
        
        # Pixels per centimeter (computed from calibration)
        self.px_per_cm: Optional[float] = None
        
        if enabled and table_corners_px is not None and table_size_cm is not None:
            self._compute_homography()
        elif enabled:
            logger.warning(
                "[Homography] Enabled but missing calibration data. "
                "Set table_corners_px and table_size_cm to enable transformation."
            )
            self.enabled = False
    
    def _compute_homography(self) -> None:
        """
        Compute the homography matrix from table corners.
        
        The homography maps the perspective-distorted table quadrilateral
        to a rectangle of known physical dimensions.
        """
        try:
            # Source points: table corners in pixel coordinates (perspective view)
            src_pts = np.array(self.table_corners_px, dtype=np.float32)
            
            # Destination points: rectangle in cm coordinates (bird's-eye view)
            # Top-left at origin, scaled by table size
            width_cm, height_cm = self.table_size_cm
            dst_pts = np.array([
                [0, 0],                    # Top-left
                [width_cm, 0],             # Top-right
                [width_cm, height_cm],     # Bottom-right
                [0, height_cm]             # Bottom-left
            ], dtype=np.float32)
            
            # Compute homography matrix using OpenCV if available, otherwise use numpy
            try:
                import cv2
                self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
                self.inverse_homography_matrix, _ = cv2.findHomography(dst_pts, src_pts)
            except ImportError:
                # Fallback to numpy-based computation
                self.homography_matrix = self._compute_homography_numpy(src_pts, dst_pts)
                self.inverse_homography_matrix = self._compute_homography_numpy(dst_pts, src_pts)
            
            # Compute approximate pixels per cm for fallback estimation
            table_diagonal_px = np.sqrt(
                (src_pts[2][0] - src_pts[0][0])**2 + 
                (src_pts[2][1] - src_pts[0][1])**2
            )
            table_diagonal_cm = np.sqrt(width_cm**2 + height_cm**2)
            self.px_per_cm = table_diagonal_px / table_diagonal_cm
            
            logger.info(
                f"[Homography] Calibration complete: "
                f"table_size={self.table_size_cm}cm, "
                f"px_per_cm≈{self.px_per_cm:.2f}"
            )
            
        except Exception as e:
            logger.error(f"[Homography] Failed to compute homography: {e}")
            self.enabled = False
            self.homography_matrix = None
    
    def _compute_homography_numpy(
        self, 
        src_pts: np.ndarray, 
        dst_pts: np.ndarray
    ) -> np.ndarray:
        """
        Compute homography matrix using numpy (fallback when OpenCV unavailable).
        
        Uses Direct Linear Transform (DLT) algorithm.
        """
        assert len(src_pts) == 4 and len(dst_pts) == 4
        
        A = []
        for i in range(4):
            x, y = src_pts[i]
            u, v = dst_pts[i]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        H = H / H[2, 2]  # Normalize
        
        return H
    
    def transform_point(self, point_px: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a point from pixel coordinates to real-world coordinates (cm).
        
        Args:
            point_px: Point in pixel coordinates (x, y)
            
        Returns:
            Point in centimeter coordinates (x_cm, y_cm)
        """
        if not self.enabled or self.homography_matrix is None:
            return point_px
        
        # Convert to homogeneous coordinates
        pt = np.array([point_px[0], point_px[1], 1.0])
        
        # Apply homography
        transformed = self.homography_matrix @ pt
        
        # Convert back from homogeneous
        x_cm = transformed[0] / transformed[2]
        y_cm = transformed[1] / transformed[2]
        
        return (x_cm, y_cm)
    
    def transform_bbox(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Transform a bounding box from pixel coordinates to real-world coordinates.
        
        Args:
            bbox: Bounding box in pixel coordinates (x1, y1, x2, y2)
            
        Returns:
            Bounding box in centimeter coordinates (x1_cm, y1_cm, x2_cm, y2_cm)
        """
        if not self.enabled or self.homography_matrix is None:
            return bbox
        
        x1, y1, x2, y2 = bbox
        
        # Transform all four corners
        corners = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]
        
        transformed_corners = [self.transform_point(c) for c in corners]
        
        # Get bounding box of transformed corners
        x_coords = [c[0] for c in transformed_corners]
        y_coords = [c[1] for c in transformed_corners]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_bbox_size_cm(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Get the real-world size of a bounding box in centimeters.
        
        Args:
            bbox: Bounding box in pixel coordinates (x1, y1, x2, y2)
            
        Returns:
            Size in centimeters (width_cm, height_cm)
        """
        if not self.enabled or self.homography_matrix is None:
            # Fallback: use approximate px_per_cm if available
            if self.px_per_cm is not None:
                x1, y1, x2, y2 = bbox
                width_px = abs(x2 - x1)
                height_px = abs(y2 - y1)
                return (width_px / self.px_per_cm, height_px / self.px_per_cm)
            return (0.0, 0.0)
        
        transformed_bbox = self.transform_bbox(bbox)
        x1_cm, y1_cm, x2_cm, y2_cm = transformed_bbox
        
        width_cm = abs(x2_cm - x1_cm)
        height_cm = abs(y2_cm - y1_cm)
        
        return (width_cm, height_cm)
    
    def get_bbox_area_cm2(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> float:
        """
        Get the real-world area of a bounding box in square centimeters.
        
        Args:
            bbox: Bounding box in pixel coordinates (x1, y1, x2, y2)
            
        Returns:
            Area in square centimeters
        """
        width_cm, height_cm = self.get_bbox_size_cm(bbox)
        return width_cm * height_cm
    
    def estimate_centroid_cm(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Get the centroid of a bounding box in real-world coordinates.
        
        Args:
            bbox: Bounding box in pixel coordinates (x1, y1, x2, y2)
            
        Returns:
            Centroid in centimeter coordinates (x_cm, y_cm)
        """
        x1, y1, x2, y2 = bbox
        centroid_px = ((x1 + x2) / 2, (y1 + y2) / 2)
        return self.transform_point(centroid_px)
    
    def is_calibrated(self) -> bool:
        """Check if homography is properly calibrated."""
        return self.enabled and self.homography_matrix is not None
    
    def get_calibration_info(self) -> dict:
        """Get calibration information for debugging/logging."""
        return {
            'enabled': self.enabled,
            'calibrated': self.is_calibrated(),
            'table_corners_px': self.table_corners_px,
            'table_size_cm': self.table_size_cm,
            'px_per_cm': self.px_per_cm
        }


# Global singleton for homography transform (lazy initialization)
_homography_instance: Optional[HomographyTransform] = None


def get_homography_transform() -> HomographyTransform:
    """
    Get the global homography transform instance.
    
    Initializes from environment variables if not already created.
    
    Environment Variables:
        HOMOGRAPHY_ENABLED: Enable/disable homography (default: false)

        # Homography Configuration
        # Add these lines to your .env file
        HOMOGRAPHY_ENABLED=true
        HOMOGRAPHY_TABLE_CORNERS='[[916.0, 184.0], [1159.0, 440.0], [388.0, 604.0], [390.0, 255.0]]'
        HOMOGRAPHY_TABLE_WIDTH_CM=140.0
        HOMOGRAPHY_TABLE_HEIGHT_CM=80.0
        HOMOGRAPHY_SMALL_THRESHOLD_CM2=100.0
        HOMOGRAPHY_LARGE_THRESHOLD_CM2=150.0

        \u2705 Saved calibration image:  data/calibration/calibration_image.jpg
        \u2705 Saved calibration data: data/calibration/calibration_data.json
        \u2705 Saved environment config: data/calibration/calibration.env

        ======================================================================
        \u2705 CALIBRATION COMPLETE!
        ======================================================================

        Next steps:
          1. Copy the environment variables above to your .env file
          2. Restart your application
          3. Verify homography is working with: python scripts/calibrate_homography.py --dry-run
        ======================================================================

    
    Returns:
        HomographyTransform instance
    """
    global _homography_instance
    
    if _homography_instance is None:
        import json
        
        enabled = os.getenv('HOMOGRAPHY_ENABLED', 'true').lower() in ('true', '1', 'yes')
        
        table_corners_px = None
        corners_str = os.getenv('HOMOGRAPHY_TABLE_CORNERS', '[[417.0, 289.0], [906.0, 286.0], [1068.0, 536.0], [363.0, 585.0]]')
        if corners_str:
            try:
                table_corners_px = json.loads(corners_str)
            except json.JSONDecodeError as e:
                logger.error(f"[Homography] Failed to parse HOMOGRAPHY_TABLE_CORNERS: {e}")
        
        table_width_cm = float(os.getenv('HOMOGRAPHY_TABLE_WIDTH_CM', '200'))
        table_height_cm = float(os.getenv('HOMOGRAPHY_TABLE_HEIGHT_CM', '100'))
        table_size_cm = (table_width_cm, table_height_cm)
        
        _homography_instance = HomographyTransform(
            table_corners_px=table_corners_px,
            table_size_cm=table_size_cm,
            enabled=enabled
        )
        
        if enabled:
            logger.info(
                f"[Homography] Initialized from environment: "
                f"enabled={enabled}, calibrated={_homography_instance.is_calibrated()}"
            )
    
    return _homography_instance


def classify_size_by_area_cm2(
    area_cm2: float,
    small_threshold_cm2: float = 760.0,
    large_threshold_cm2: float = 915.0
) -> Tuple[str, str]:
    """
    Classify size based on real-world area in square centimeters.
    
    Args:
        area_cm2: Area in square centimeters
        small_threshold_cm2: Maximum area for "Small" classification
        large_threshold_cm2: Minimum area for "Large" classification
        
    Returns:
        Tuple of (size_class, size_bin):
        - size_class: 'Small', 'Regular', or 'Large'
        - size_bin: More specific bin like 'very_small', 'small', 'medium', 'large'
    """
    if area_cm2 < small_threshold_cm2:
        if area_cm2 < small_threshold_cm2 * 0.5:
            return ('Small', 'very_small')
        return ('Small', 'small')
    elif area_cm2 > large_threshold_cm2:
        return ('Large', 'large')
    else:
        return ('Regular', 'medium')
