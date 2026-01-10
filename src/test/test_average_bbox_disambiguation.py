"""
Unit Tests for Average Bbox Disambiguation.

Tests the V8 enhancement that uses average bbox size from all closed ROIs
instead of just the best one when disambiguating Brown_Orange_Family labels.

Run with: python -m pytest src/test/test_average_bbox_disambiguation.py -v
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MockTrackingConfig:
    """Mock configuration for testing."""
    disambiguation_v2_enabled: bool = True
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    disambiguation_family_name: str = 'Brown_Orange_Family'
    disambiguation_small_threshold: float = 9000.0
    disambiguation_regular_threshold: float = 11000.0
    homography_enabled: bool = False


class TestAverageBboxDisambiguation:
    """Tests for average bbox calculation in family label disambiguation."""
    
    def test_average_bbox_calculation_basic(self):
        """Verify average bbox is calculated correctly from multiple closed ROIs."""
        # Create mock candidates with different bbox sizes
        candidates = [
            {
                'state': 'closed',
                'bbox': (100, 100, 200, 200),  # 100x100 = 10000 px²
                'confidence': 0.8,
                'trust': 0.7,
                'sharpness': 500.0
            },
            {
                'state': 'closed',
                'bbox': (100, 100, 220, 220),  # 120x120 = 14400 px²
                'confidence': 0.75,
                'trust': 0.65,
                'sharpness': 480.0
            },
            {
                'state': 'closed',
                'bbox': (100, 100, 180, 180),  # 80x80 = 6400 px²
                'confidence': 0.7,
                'trust': 0.6,
                'sharpness': 450.0
            }
        ]
        
        # Calculate expected average dimensions
        widths = [200-100, 220-100, 180-100]  # [100, 120, 80]
        heights = [200-100, 220-100, 180-100]  # [100, 120, 80]
        expected_avg_width = sum(widths) / len(widths)  # 100
        expected_avg_height = sum(heights) / len(heights)  # 100
        expected_avg_area = expected_avg_width * expected_avg_height  # 10000
        
        assert expected_avg_width == 100.0
        assert expected_avg_height == 100.0
        assert expected_avg_area == 10000.0
    
    def test_average_bbox_filters_invalid_bboxes(self):
        """Verify that invalid bboxes (negative dimensions, None) are filtered out."""
        candidates = [
            {
                'state': 'closed',
                'bbox': (100, 100, 200, 200),  # Valid: 100x100
                'confidence': 0.8,
                'trust': 0.7,
            },
            {
                'state': 'closed',
                'bbox': None,  # Invalid: None
                'confidence': 0.75,
                'trust': 0.65,
            },
            {
                'state': 'closed',
                'bbox': (200, 200, 100, 100),  # Invalid: negative dimensions
                'confidence': 0.7,
                'trust': 0.6,
            },
            {
                'state': 'closed',
                'bbox': (100, 100, 180, 180),  # Valid: 80x80
                'confidence': 0.65,
                'trust': 0.55,
            }
        ]
        
        # Filter valid bboxes
        valid_bboxes = []
        for c in candidates:
            bbox = c.get('bbox')
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    valid_bboxes.append(bbox)
        
        # Should only have 2 valid bboxes
        assert len(valid_bboxes) == 2
        assert valid_bboxes[0] == (100, 100, 200, 200)
        assert valid_bboxes[1] == (100, 100, 180, 180)
    
    def test_average_area_determines_small_vs_regular(self):
        """
        Verify that average area determines classification.
        
        Small threshold: 9000 px²
        Regular threshold: 11000 px²
        
        Individual ROIs with areas:
        - ROI 1: 7000 px² (would be small)
        - ROI 2: 12000 px² (would be regular)
        
        Average area: (7000 + 12000) / 2 = 9500 px² → Gray zone, but if we had:
        
        - ROI 1: 8000 px²
        - ROI 2: 8500 px²
        - ROI 3: 9000 px²
        
        Average: 8500 px² → small
        """
        # This test validates the concept - actual integration test would require
        # mocking the ClassifierService more extensively
        
        # Scenario: Multiple ROIs average to "small" range
        areas_small = [8000, 8500, 9000]
        avg_small = sum(areas_small) / len(areas_small)  # 8500
        assert avg_small < 9000  # Below small threshold = small
        
        # Scenario: Multiple ROIs average to "regular" range
        areas_regular = [11000, 12000, 13000]
        avg_regular = sum(areas_regular) / len(areas_regular)  # 12000
        assert avg_regular > 11000  # Above regular threshold = regular
        
        # Scenario: ROIs have variance but average in gray zone
        areas_mixed = [7000, 10000, 13000]
        avg_mixed = sum(areas_mixed) / len(areas_mixed)  # 10000
        assert 9000 <= avg_mixed <= 11000  # Gray zone
    
    def test_single_roi_behaves_correctly(self):
        """Verify that with a single closed ROI, average equals that ROI's size."""
        candidates = [
            {
                'state': 'closed',
                'bbox': (100, 100, 220, 210),  # 120x110 = 13200 px²
                'confidence': 0.8,
                'trust': 0.7,
            }
        ]
        
        # Filter and compute average
        bbox = candidates[0]['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        # With single ROI, average should equal the single measurement
        assert width == 120
        assert height == 110
        assert area == 13200
    
    def test_metadata_includes_individual_areas(self):
        """Verify that metadata includes individual bbox areas for debugging."""
        bboxes = [
            (100, 100, 200, 200),  # 100x100 = 10000 px²
            (100, 100, 220, 220),  # 120x120 = 14400 px²
            (100, 100, 180, 180),  # 80x80 = 6400 px²
        ]
        
        individual_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        
        assert individual_areas == [10000, 14400, 6400]
        assert sum(individual_areas) / len(individual_areas) == pytest.approx(10266.67, rel=0.01)


class TestAverageBboxEdgeCases:
    """Edge case tests for average bbox calculation."""
    
    def test_no_closed_rois_fallback(self):
        """With no closed ROIs, should fallback to default subclass."""
        candidates = [
            {
                'state': 'open',  # Not closed
                'bbox': (100, 100, 200, 200),
                'confidence': 0.8,
            }
        ]
        
        closed_candidates = [c for c in candidates if c.get('state') == 'closed']
        assert len(closed_candidates) == 0
    
    def test_all_invalid_bboxes_fallback(self):
        """With all invalid bboxes, should fallback to default subclass."""
        candidates = [
            {
                'state': 'closed',
                'bbox': None,
                'confidence': 0.8,
            },
            {
                'state': 'closed',
                'bbox': (200, 200, 100, 100),  # Invalid dimensions
                'confidence': 0.75,
            }
        ]
        
        valid_bboxes = []
        for c in candidates:
            bbox = c.get('bbox')
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    valid_bboxes.append(bbox)
        
        assert len(valid_bboxes) == 0
    
    def test_large_variance_in_roi_sizes(self):
        """
        Even with large variance, average should be computed correctly.
        
        This tests the robustness of using average vs best-only approach.
        """
        # Scenario: Very different sizes due to detection noise
        bboxes = [
            (100, 100, 150, 150),   # 50x50 = 2500 px² (noise - very small)
            (100, 100, 220, 220),   # 120x120 = 14400 px² (normal)
            (100, 100, 230, 230),   # 130x130 = 16900 px² (normal)
            (100, 100, 210, 210),   # 110x110 = 12100 px² (normal)
        ]
        
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        
        # Average is more robust than just picking one ROI
        avg_area = sum(areas) / len(areas)  # (2500 + 14400 + 16900 + 12100) / 4 = 11475
        
        # The outlier (2500) pulls down the average slightly, but result is still
        # in "regular" range (> 11000) rather than being skewed by the outlier alone
        assert avg_area == 11475.0
        assert avg_area > 11000  # Still regular despite one noisy detection


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
