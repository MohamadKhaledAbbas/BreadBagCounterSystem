"""
Unit tests for homography-based size classification module.

Tests the HomographyTransform class and related functions.
"""

import pytest
import numpy as np
from typing import Tuple


class TestHomographyTransform:
    """Tests for HomographyTransform class."""
    
    def test_disabled_transform_returns_original(self):
        """Test that disabled transform returns original values."""
        from src.classifier.homography import HomographyTransform
        
        h = HomographyTransform(enabled=False)
        
        assert not h.enabled
        assert not h.is_calibrated()
        
        # Transform should return original point
        point = (100.0, 100.0)
        transformed = h.transform_point(point)
        assert transformed == point
        
        # Transform should return original bbox
        bbox = (50.0, 50.0, 150.0, 150.0)
        transformed_bbox = h.transform_bbox(bbox)
        assert transformed_bbox == bbox
    
    def test_enabled_without_calibration_returns_original(self):
        """Test that enabled but uncalibrated transform returns original values."""
        from src.classifier.homography import HomographyTransform
        
        h = HomographyTransform(enabled=True, table_corners_px=None, table_size_cm=None)
        
        # Should be disabled due to missing calibration
        assert not h.enabled
        assert not h.is_calibrated()
    
    def test_calibrated_transform_computes_homography(self):
        """Test that calibrated transform computes homography matrix."""
        from src.classifier.homography import HomographyTransform
        
        # Simple square table: 100x100 pixels -> 50x50 cm
        table_corners_px = [[0, 0], [100, 0], [100, 100], [0, 100]]
        table_size_cm = (50.0, 50.0)
        
        h = HomographyTransform(
            table_corners_px=table_corners_px,
            table_size_cm=table_size_cm,
            enabled=True
        )
        
        assert h.enabled
        assert h.is_calibrated()
        assert h.homography_matrix is not None
        assert h.px_per_cm is not None
    
    def test_transform_point_converts_coordinates(self):
        """Test that transform_point converts pixel coords to cm coords."""
        from src.classifier.homography import HomographyTransform
        
        # Simple 1:1 mapping for easy testing
        # 100x100 pixels -> 100x100 cm (1 px = 1 cm)
        table_corners_px = [[0, 0], [100, 0], [100, 100], [0, 100]]
        table_size_cm = (100.0, 100.0)
        
        h = HomographyTransform(
            table_corners_px=table_corners_px,
            table_size_cm=table_size_cm,
            enabled=True
        )
        
        # Center of table should map to center in cm
        center_px = (50.0, 50.0)
        center_cm = h.transform_point(center_px)
        
        assert abs(center_cm[0] - 50.0) < 1.0
        assert abs(center_cm[1] - 50.0) < 1.0
    
    def test_get_bbox_size_cm(self):
        """Test that get_bbox_size_cm returns correct real-world dimensions."""
        from src.classifier.homography import HomographyTransform
        
        # 100x100 pixels -> 100x100 cm
        table_corners_px = [[0, 0], [100, 0], [100, 100], [0, 100]]
        table_size_cm = (100.0, 100.0)
        
        h = HomographyTransform(
            table_corners_px=table_corners_px,
            table_size_cm=table_size_cm,
            enabled=True
        )
        
        # 20x20 pixel bbox should be approximately 20x20 cm
        bbox = (10.0, 10.0, 30.0, 30.0)
        size_cm = h.get_bbox_size_cm(bbox)
        
        assert abs(size_cm[0] - 20.0) < 2.0  # Allow some tolerance
        assert abs(size_cm[1] - 20.0) < 2.0
    
    def test_get_bbox_area_cm2(self):
        """Test that get_bbox_area_cm2 returns correct area."""
        from src.classifier.homography import HomographyTransform
        
        # 100x100 pixels -> 100x100 cm
        table_corners_px = [[0, 0], [100, 0], [100, 100], [0, 100]]
        table_size_cm = (100.0, 100.0)
        
        h = HomographyTransform(
            table_corners_px=table_corners_px,
            table_size_cm=table_size_cm,
            enabled=True
        )
        
        # 20x20 pixel bbox should be approximately 400 cm²
        bbox = (10.0, 10.0, 30.0, 30.0)
        area_cm2 = h.get_bbox_area_cm2(bbox)
        
        assert abs(area_cm2 - 400.0) < 50.0  # Allow some tolerance
    
    def test_get_calibration_info(self):
        """Test that get_calibration_info returns correct metadata."""
        from src.classifier.homography import HomographyTransform
        
        table_corners_px = [[0, 0], [100, 0], [100, 100], [0, 100]]
        table_size_cm = (50.0, 50.0)
        
        h = HomographyTransform(
            table_corners_px=table_corners_px,
            table_size_cm=table_size_cm,
            enabled=True
        )
        
        info = h.get_calibration_info()
        
        assert info['enabled']
        assert info['calibrated']
        assert info['table_corners_px'] == table_corners_px
        assert info['table_size_cm'] == table_size_cm
        assert info['px_per_cm'] is not None


class TestClassifySizeByAreaCm2:
    """Tests for classify_size_by_area_cm2 function."""
    
    def test_small_classification(self):
        """Test that small areas are classified as Small."""
        from src.classifier.homography import classify_size_by_area_cm2
        
        size_class, size_bin = classify_size_by_area_cm2(
            area_cm2=50.0,
            small_threshold_cm2=100.0,
            large_threshold_cm2=150.0
        )
        
        assert size_class == 'Small'
        assert size_bin == 'small'
    
    def test_very_small_classification(self):
        """Test that very small areas are classified as very_small."""
        from src.classifier.homography import classify_size_by_area_cm2
        
        size_class, size_bin = classify_size_by_area_cm2(
            area_cm2=30.0,
            small_threshold_cm2=100.0,
            large_threshold_cm2=150.0
        )
        
        assert size_class == 'Small'
        assert size_bin == 'very_small'
    
    def test_regular_classification(self):
        """Test that medium areas are classified as Regular."""
        from src.classifier.homography import classify_size_by_area_cm2
        
        size_class, size_bin = classify_size_by_area_cm2(
            area_cm2=120.0,
            small_threshold_cm2=100.0,
            large_threshold_cm2=150.0
        )
        
        assert size_class == 'Regular'
        assert size_bin == 'medium'
    
    def test_large_classification(self):
        """Test that large areas are classified as Large."""
        from src.classifier.homography import classify_size_by_area_cm2
        
        size_class, size_bin = classify_size_by_area_cm2(
            area_cm2=200.0,
            small_threshold_cm2=100.0,
            large_threshold_cm2=150.0
        )
        
        assert size_class == 'Large'
        assert size_bin == 'large'


class TestGetHomographyTransform:
    """Tests for get_homography_transform singleton function."""
    
    def test_returns_singleton(self):
        """Test that get_homography_transform returns the same instance."""
        from src.classifier.homography import get_homography_transform
        import src.classifier.homography as homography_module
        
        # Reset singleton for test
        homography_module._homography_instance = None
        
        h1 = get_homography_transform()
        h2 = get_homography_transform()
        
        assert h1 is h2
    
    def test_default_disabled(self):
        """Test that default homography is disabled."""
        from src.classifier.homography import get_homography_transform
        import src.classifier.homography as homography_module
        
        # Reset singleton for test
        homography_module._homography_instance = None
        
        h = get_homography_transform()
        
        # Default should be disabled (env var not set)
        assert not h.enabled or not h.is_calibrated()
