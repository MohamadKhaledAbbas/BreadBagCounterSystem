"""
Unit Tests for Disambiguation V2 Module (Production-Grade).

Tests cover:
1. Multi-threshold size bin logic
2. Aspect ratio and area validation
3. Gray zone resolution strategies
4. Confidence penalty mechanisms
5. Detailed metadata and logging
6. Edge cases and validation failures
7. Context tracking

Run with: python -m pytest src/test/test_disambiguation_v2.py -v
(Or run directly if pytest not installed: python src/test/test_disambiguation_v2.py)
"""

import sys
import os
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Handle pytest import gracefully
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    print("Warning: pytest not installed. Running in standalone mode.")
    PYTEST_AVAILABLE = False
    # Mock pytest decorators
    class pytest:
        @staticmethod
        def fixture(func):
            return func
        
        class mark:
            @staticmethod
            def parametrize(argnames, argvalues):
                def decorator(func):
                    return func
                return decorator
        
        @staticmethod
        def approx(value, rel=None):
            """Mock pytest.approx for standalone mode."""
            class Approx:
                def __init__(self, expected, rel):
                    self.expected = expected
                    self.rel = rel if rel is not None else 1e-6
                
                def __eq__(self, actual):
                    return abs(actual - self.expected) <= abs(self.expected * self.rel)
            return Approx(value, rel)

from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.classifier.disambiguation_v2 import (
    disambiguate_v2,
    disambiguate_batch_v2,
    validate_bbox,
    compute_size_bin,
    resolve_gray_zone,
    DisambiguationV2Result,
    ValidationResult
)


# =============================================================================
# Mock Configuration
# =============================================================================

@dataclass
class MockConfigV2:
    """Mock configuration for V2 testing."""
    # V2 enable flag
    disambiguation_v2_enabled: bool = True
    
    # V1 compatibility
    disambiguation_enabled: bool = True
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    disambiguation_small_threshold: float = 9000.0
    disambiguation_regular_threshold: float = 11000.0
    disambiguation_gray_zone_behavior: str = 'keep_original'
    disambiguation_family_name: str = 'Brown_Orange_Family'
    disambiguation_confidence_penalty: float = 0.9
    disambiguation_penalty_on_change_only: bool = False
    disambiguation_v2_debug_logging: bool = False
    
    # V2 validation parameters
    disambiguation_v2_min_aspect_ratio: float = 0.3
    disambiguation_v2_max_aspect_ratio: float = 3.0
    disambiguation_v2_aspect_ratio_penalty: float = 0.3
    disambiguation_v2_min_realistic_area: float = 1000.0
    disambiguation_v2_max_realistic_area: float = 100000.0
    disambiguation_v2_unrealistic_area_penalty: float = 0.5
    
    # V2 multi-threshold bins
    disambiguation_v2_very_small_threshold: float = 5000.0
    disambiguation_v2_large_threshold: float = 25000.0
    
    # V2 gray zone confidence strategy
    disambiguation_v2_gray_zone_confidence_threshold: float = 0.6


@pytest.fixture
def default_config_v2():
    """Default V2 mock configuration."""
    return MockConfigV2()


# =============================================================================
# Validation Tests
# =============================================================================

class TestBboxValidation:
    """Tests for bounding box validation."""
    
    def test_valid_bbox(self, default_config_v2):
        """Valid bbox should pass all checks."""
        bbox = (100, 100, 200, 200)  # 100x100 = 10,000 px²
        result = validate_bbox(bbox, default_config_v2)
        
        assert result.valid is True
        assert result.reason is None
        assert result.penalty_applied == 0.0
        assert result.metadata['area'] == 10000.0
        assert result.metadata['aspect_ratio'] == 1.0
    
    def test_degenerate_bbox_negative_width(self, default_config_v2):
        """Degenerate bbox with negative width should be rejected."""
        bbox = (200, 100, 100, 200)  # width = -100
        result = validate_bbox(bbox, default_config_v2)
        
        assert result.valid is False
        assert 'degenerate_bbox' in result.reason
        assert result.penalty_applied == 1.0
    
    def test_degenerate_bbox_zero_height(self, default_config_v2):
        """Degenerate bbox with zero height should be rejected."""
        bbox = (100, 100, 200, 100)  # height = 0
        result = validate_bbox(bbox, default_config_v2)
        
        assert result.valid is False
        assert 'degenerate_bbox' in result.reason
        assert result.penalty_applied == 1.0
    
    def test_suspicious_aspect_ratio_too_narrow(self, default_config_v2):
        """Bbox with aspect ratio < min should be penalized."""
        bbox = (100, 100, 120, 200)  # AR = 20/100 = 0.2 < 0.3
        result = validate_bbox(bbox, default_config_v2)
        
        assert result.valid is True  # Still valid but penalized
        assert 'suspicious_aspect_ratio' in result.reason
        assert result.penalty_applied == 0.3
        assert result.metadata['aspect_ratio'] == 0.2
    
    def test_suspicious_aspect_ratio_too_wide(self, default_config_v2):
        """Bbox with aspect ratio > max should be penalized."""
        bbox = (100, 100, 450, 200)  # AR = 350/100 = 3.5 > 3.0
        result = validate_bbox(bbox, default_config_v2)
        
        assert result.valid is True
        assert 'suspicious_aspect_ratio' in result.reason
        assert result.penalty_applied == 0.3
        assert result.metadata['aspect_ratio'] == 3.5
    
    def test_unrealistically_small_area(self, default_config_v2):
        """Bbox with area < min_realistic should be penalized."""
        bbox = (100, 100, 115, 150)  # 15x50 = 750 < 1000
        result = validate_bbox(bbox, default_config_v2)
        
        assert result.valid is True
        assert 'unrealistically_small_area' in result.reason
        assert result.penalty_applied == 0.5
    
    def test_unrealistically_large_area(self, default_config_v2):
        """Bbox with area > max_realistic should be penalized."""
        bbox = (0, 0, 500, 300)  # 500x300 = 150,000 > 100,000
        result = validate_bbox(bbox, default_config_v2)
        
        assert result.valid is True
        assert 'unrealistically_large_area' in result.reason
        assert result.penalty_applied == 0.5


# =============================================================================
# Size Bin Tests
# =============================================================================

class TestSizeBinComputation:
    """Tests for multi-threshold size bin logic."""
    
    def test_very_small_bin(self, default_config_v2):
        """Area < very_small_threshold should be 'very_small' bin."""
        raw_area = 3000.0
        bin_name, metadata = compute_size_bin(raw_area, default_config_v2)
        
        assert bin_name == 'very_small'
        assert metadata['raw_area'] == 3000.0
        assert metadata['bin'] == 'very_small'
    
    def test_small_bin(self, default_config_v2):
        """Area in [very_small, small) should be 'small' bin."""
        raw_area = 7000.0
        bin_name, metadata = compute_size_bin(raw_area, default_config_v2)
        
        assert bin_name == 'small'
    
    def test_gray_zone_bin(self, default_config_v2):
        """Area in [small, regular] should be 'gray_zone' bin."""
        raw_area = 10000.0
        bin_name, metadata = compute_size_bin(raw_area, default_config_v2)
        
        assert bin_name == 'gray_zone'
    
    def test_regular_bin(self, default_config_v2):
        """Area in (regular, large] should be 'regular' bin."""
        raw_area = 18000.0
        bin_name, metadata = compute_size_bin(raw_area, default_config_v2)
        
        assert bin_name == 'regular'
    
    def test_large_bin(self, default_config_v2):
        """Area > large_threshold should be 'large' bin."""
        raw_area = 30000.0
        bin_name, metadata = compute_size_bin(raw_area, default_config_v2)
        
        assert bin_name == 'large'


# =============================================================================
# Gray Zone Resolution Tests
# =============================================================================

class TestGrayZoneResolution:
    """Tests for simplified gray zone resolution (midpoint-based)."""
    
    def test_below_midpoint_pixel(self, default_config_v2):
        """Area below midpoint should resolve to small class (pixel mode)."""
        size_metadata = {
            'raw_area': 9500.0,  # Below midpoint of 10000
            'homography_used': False,
            'thresholds': {
                'small': 9000.0,
                'regular': 11000.0
            }
        }
        
        label, reason = resolve_gray_zone(
            original_label='Brown_Orange_Family',
            size_bin_metadata=size_metadata,
            target_classes=('Brown_Orange_Overlay', 'Brown_Orange_Small'),
            homography_used=False
        )
        
        assert label == 'Brown_Orange_Small'
        assert 'gray_zone_resolved_to_small' in reason
        assert '9500.0px²' in reason
    
    def test_above_midpoint_pixel(self, default_config_v2):
        """Area above midpoint should resolve to regular class (pixel mode)."""
        size_metadata = {
            'raw_area': 10500.0,  # Above midpoint of 10000
            'homography_used': False,
            'thresholds': {
                'small': 9000.0,
                'regular': 11000.0
            }
        }
        
        label, reason = resolve_gray_zone(
            original_label='Brown_Orange_Family',
            size_bin_metadata=size_metadata,
            target_classes=('Brown_Orange_Overlay', 'Brown_Orange_Small'),
            homography_used=False
        )
        
        assert label == 'Brown_Orange_Overlay'
        assert 'gray_zone_resolved_to_regular' in reason
        assert '10500.0px²' in reason
    
    def test_below_midpoint_homography(self, default_config_v2):
        """Area below midpoint should resolve to small class (homography mode)."""
        size_metadata = {
            'area_cm2': 120.0,  # Below midpoint of 125
            'homography_used': True,
            'thresholds_cm2': {
                'small': 100.0,
                'large': 150.0
            }
        }
        
        label, reason = resolve_gray_zone(
            original_label='Brown_Orange_Family',
            size_bin_metadata=size_metadata,
            target_classes=('Brown_Orange_Overlay', 'Brown_Orange_Small'),
            homography_used=True
        )
        
        assert label == 'Brown_Orange_Small'
        assert 'gray_zone_resolved_to_small' in reason
        assert '120.0cm²' in reason
    
    def test_above_midpoint_homography(self, default_config_v2):
        """Area above midpoint should resolve to regular class (homography mode)."""
        size_metadata = {
            'area_cm2': 130.0,  # Above midpoint of 125
            'homography_used': True,
            'thresholds_cm2': {
                'small': 100.0,
                'large': 150.0
            }
        }
        
        label, reason = resolve_gray_zone(
            original_label='Brown_Orange_Family',
            size_bin_metadata=size_metadata,
            target_classes=('Brown_Orange_Overlay', 'Brown_Orange_Small'),
            homography_used=True
        )
        
        assert label == 'Brown_Orange_Overlay'
        assert 'gray_zone_resolved_to_regular' in reason
        assert '130.0cm²' in reason



# =============================================================================
# Main Disambiguation V2 Tests
# =============================================================================

class TestDisambiguationV2:
    """Tests for main disambiguate_v2 function."""
    
    def test_disabled_v2_returns_original(self, default_config_v2):
        """When V2 is disabled, should return original label."""
        default_config_v2.disambiguation_v2_enabled = False
        
        result = disambiguate_v2(
            original_label='Brown_Orange_Overlay',
            confidence=0.8,
            bbox=(100, 100, 200, 200),
            is_open=False,
            config=default_config_v2
        )
        
        assert result.label == 'Brown_Orange_Overlay'
        assert result.confidence == 0.8
        assert result.disambiguated is False
        assert result.reason == 'disambiguation_v2_disabled'
    
    def test_skip_open_state(self, default_config_v2):
        """Open state ROIs should be skipped."""
        result = disambiguate_v2(
            original_label='Brown_Orange_Overlay',
            confidence=0.8,
            bbox=(100, 100, 200, 200),
            is_open=True,
            config=default_config_v2
        )
        
        assert result.label == 'Brown_Orange_Overlay'
        assert result.disambiguated is False
        assert result.reason == 'skipped_open_state'
        assert result.metadata['is_open'] is True
    
    def test_skip_non_target_family(self, default_config_v2):
        """Non-target classes should be skipped."""
        result = disambiguate_v2(
            original_label='Blue_Yellow',
            confidence=0.8,
            bbox=(100, 100, 200, 200),
            is_open=False,
            config=default_config_v2
        )
        
        assert result.label == 'Blue_Yellow'
        assert result.disambiguated is False
        assert result.reason == 'not_target_family'
    
    def test_very_small_area_to_small_class(self, default_config_v2):
        """Very small area should map to Small class."""
        result = disambiguate_v2(
            original_label='Brown_Orange_Overlay',  # Classifier wrong
            confidence=0.7,
            bbox=(100, 100, 170, 150),  # 70x50 = 3,500 px²
            is_open=False,
            config=default_config_v2
        )
        
        assert result.label == 'Brown_Orange_Small'
        assert result.disambiguated is True
        assert 'small' in result.reason.lower()
        assert result.confidence_tier == 'low'  # Pixel fallback
    
    def test_large_area_to_overlay_class(self, default_config_v2):
        """Large area should map to Overlay class."""
        result = disambiguate_v2(
            original_label='Brown_Orange_Small',  # Classifier wrong
            confidence=0.7,
            bbox=(100, 100, 300, 300),  # 200x200 = 40,000 px²
            is_open=False,
            config=default_config_v2
        )
        
        assert result.label == 'Brown_Orange_Overlay'
        assert result.disambiguated is True
        assert 'regular' in result.reason.lower()
        assert result.confidence_tier == 'low'  # Pixel fallback
    
    def test_gray_zone_resolution_by_midpoint(self, default_config_v2):
        """Gray zone should resolve by midpoint."""
        # Area = 10,000 px², midpoint = (9000 + 11000) / 2 = 10,000
        # Should resolve to regular (above or equal to midpoint)
        result = disambiguate_v2(
            original_label='Brown_Orange_Family',
            confidence=0.7,
            bbox=(100, 100, 200, 150),  # 100x50 = 10,000 px² (at midpoint)
            is_open=False,
            config=default_config_v2
        )
        
        assert result.disambiguated is True
        assert 'gray_zone' in result.reason
        assert result.confidence_tier == 'low'  # Gray zone always low
    
    def test_confidence_penalty_gray_zone_pixel_fallback(self, default_config_v2):
        """Confidence penalty only applied for pixel fallback + gray zone."""
        result = disambiguate_v2(
            original_label='Brown_Orange_Family',
            confidence=1.0,
            bbox=(100, 100, 200, 150),  # Gray zone
            is_open=False,
            config=default_config_v2
        )
        
        # Penalty only for gray zone with pixel fallback
        assert result.confidence == 0.9  # 1.0 * 0.9
    
    def test_no_penalty_clear_classification(self, default_config_v2):
        """No penalty when classification is clear (not gray zone)."""
        result = disambiguate_v2(
            original_label='Brown_Orange_Small',
            confidence=1.0,
            bbox=(100, 100, 170, 150),  # Very small area - clear
            is_open=False,
            config=default_config_v2
        )
        
        assert result.label == 'Brown_Orange_Small'
        assert result.confidence == 1.0  # No penalty for clear classification
    
    def test_context_included_in_metadata(self, default_config_v2):
        """Context should be included in result metadata."""
        context = {'track_id': 123, 'frame_index': 45}
        
        result = disambiguate_v2(
            original_label='Brown_Orange_Overlay',
            confidence=0.8,
            bbox=(100, 100, 200, 200),
            is_open=False,
            config=default_config_v2,
            context=context
        )
        
        assert result.metadata['context'] == context
    
    def test_family_name_recognition(self, default_config_v2):
        """Should recognize Brown_Orange_Family as family member."""
        result = disambiguate_v2(
            original_label='Brown_Orange_Family',
            confidence=0.7,
            bbox=(100, 100, 170, 150),  # Very small
            is_open=False,
            config=default_config_v2
        )
        
        assert result.label == 'Brown_Orange_Small'
        assert result.disambiguated is True
    
    def test_homography_metadata_present(self, default_config_v2):
        """Result metadata should include homography usage flag."""
        result = disambiguate_v2(
            original_label='Brown_Orange_Overlay',
            confidence=0.8,
            bbox=(100, 100, 200, 200),
            is_open=False,
            config=default_config_v2
        )
        
        # Should have homography_used in metadata
        assert 'homography_used' in result.metadata
        # Should be False since no calibration in test
        assert result.metadata['homography_used'] is False



# =============================================================================
# Batch Disambiguation Tests
# =============================================================================

class TestBatchDisambiguationV2:
    """Tests for batch disambiguation."""
    
    def test_batch_with_mixed_results(self, default_config_v2):
        """Batch should handle mix of skipped and disambiguated."""
        classifications = [
            {
                'label': 'Brown_Orange_Overlay',
                'confidence': 0.8,
                'bbox': (100, 100, 170, 150),  # Very small
                'is_open': False
            },
            {
                'label': 'Blue_Yellow',
                'confidence': 0.9,
                'bbox': (100, 100, 200, 200),
                'is_open': False
            },
            {
                'label': 'Brown_Orange_Small',
                'confidence': 0.7,
                'bbox': None,  # No bbox
                'is_open': False
            }
        ]
        
        results = disambiguate_batch_v2(
            classifications,
            default_config_v2,
            context={'track_id': 123}
        )
        
        assert len(results) == 3
        
        # First should be disambiguated to Small
        assert results[0]['label'] == 'Brown_Orange_Small'
        assert results[0]['disambiguation_v2']['applied'] is True
        
        # Second should be skipped (not family)
        assert results[1]['label'] == 'Blue_Yellow'
        assert results[1]['disambiguation_v2']['applied'] is False
        
        # Third should be skipped (no bbox)
        assert results[2]['label'] == 'Brown_Orange_Small'
        assert results[2]['disambiguation_v2']['applied'] is False
        assert results[2]['disambiguation_v2']['reason'] == 'no_bbox'


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    if PYTEST_AVAILABLE:
        pytest.main([__file__, '-v'])
    else:
        # Standalone mode - run tests manually
        print("\n" + "="*80)
        print("Running Disambiguation V2 Tests (Standalone Mode)")
        print("="*80 + "\n")
        
        # Create config
        config = MockConfigV2()
        
        # Track results
        test_results = {'passed': 0, 'failed': 0}
        
        def run_test(test_class, test_method_name):
            try:
                test_instance = test_class()
                test_method = getattr(test_instance, test_method_name)
                # Call test method with config as the fixture parameter
                test_method(default_config_v2=config)
                print(f"✓ {test_class.__name__}.{test_method_name}")
                test_results['passed'] += 1
            except AssertionError as e:
                error_msg = str(e) or "(assertion failed with no message)"
                tb = traceback.format_exc()
                print(f"✗ {test_class.__name__}.{test_method_name}: AssertionError: {error_msg}")
                if os.getenv('DEBUG_TESTS'):
                    print(tb)
                test_results['failed'] += 1
            except Exception as e:
                print(f"✗ {test_class.__name__}.{test_method_name}: {type(e).__name__}: {e}")
                test_results['failed'] += 1
        
        # Run all tests
        print("\n--- Bbox Validation Tests ---")
        run_test(TestBboxValidation, 'test_valid_bbox')
        run_test(TestBboxValidation, 'test_degenerate_bbox_negative_width')
        run_test(TestBboxValidation, 'test_degenerate_bbox_zero_height')
        run_test(TestBboxValidation, 'test_suspicious_aspect_ratio_too_narrow')
        run_test(TestBboxValidation, 'test_suspicious_aspect_ratio_too_wide')
        run_test(TestBboxValidation, 'test_unrealistically_small_area')
        run_test(TestBboxValidation, 'test_unrealistically_large_area')
        
        print("\n--- Size Bin Tests ---")
        run_test(TestSizeBinComputation, 'test_very_small_bin')
        run_test(TestSizeBinComputation, 'test_small_bin')
        run_test(TestSizeBinComputation, 'test_gray_zone_bin')
        run_test(TestSizeBinComputation, 'test_regular_bin')
        run_test(TestSizeBinComputation, 'test_large_bin')
        
        print("\n--- Gray Zone Resolution Tests ---")
        run_test(TestGrayZoneResolution, 'test_below_midpoint_pixel')
        run_test(TestGrayZoneResolution, 'test_above_midpoint_pixel')
        run_test(TestGrayZoneResolution, 'test_below_midpoint_homography')
        run_test(TestGrayZoneResolution, 'test_above_midpoint_homography')
        
        print("\n--- Main Disambiguation V2 Tests ---")
        run_test(TestDisambiguationV2, 'test_disabled_v2_returns_original')
        run_test(TestDisambiguationV2, 'test_skip_open_state')
        run_test(TestDisambiguationV2, 'test_skip_non_target_family')
        run_test(TestDisambiguationV2, 'test_very_small_area_to_small_class')
        run_test(TestDisambiguationV2, 'test_large_area_to_overlay_class')
        run_test(TestDisambiguationV2, 'test_gray_zone_resolution_by_midpoint')
        run_test(TestDisambiguationV2, 'test_confidence_penalty_gray_zone_pixel_fallback')
        run_test(TestDisambiguationV2, 'test_no_penalty_clear_classification')
        run_test(TestDisambiguationV2, 'test_context_included_in_metadata')
        run_test(TestDisambiguationV2, 'test_family_name_recognition')
        run_test(TestDisambiguationV2, 'test_homography_metadata_present')
        
        print("\n--- Batch Disambiguation Tests ---")
        run_test(TestBatchDisambiguationV2, 'test_batch_with_mixed_results')
        
        # Summary
        print("\n" + "="*80)
        print(f"Test Results: {test_results['passed']} passed, {test_results['failed']} failed")
        print("="*80)
        
        sys.exit(0 if test_results['failed'] == 0 else 1)
