"""
Unit Tests for Track-Level Disambiguation Logic.

Tests the refactored bag classification flow where disambiguation is applied
at track level after aggregation/voting, not per-ROI.

Key test scenarios:
1. Winner label is NOT a family label -> no disambiguation needed
2. Winner label is a family label with closed ROIs -> disambiguate once
3. Winner label is a family label with NO closed ROIs -> fallback to default
4. Multiple family members in ROIs -> voting determines winner, then disambiguate
5. Confidence tier assignment based on disambiguation results

Run with: python -m pytest src/test/test_track_disambiguation.py -v
"""

import sys
import os

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

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import unittest


# =============================================================================
# Mock Configuration
# =============================================================================

@dataclass
class MockTrackingConfig:
    """Mock tracking configuration for testing."""
    # Disambiguation settings
    disambiguation_enabled: bool = True
    disambiguation_v2_enabled: bool = True
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    disambiguation_family_name: str = 'Brown_Orange_Family'
    disambiguation_small_threshold: float = 9000.0
    disambiguation_regular_threshold: float = 11000.0
    disambiguation_gray_zone_behavior: str = 'keep_original'
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


# =============================================================================
# Mock ClassifierService Methods
# =============================================================================

class MockClassifierService:
    """Mock ClassifierService for testing track-level disambiguation."""
    
    def __init__(self, config):
        self.config = config
        self.disambiguation_enabled = config.disambiguation_enabled
        self.disambiguation_v2_enabled = config.disambiguation_v2_enabled
    
    def _is_family_label(self, label: str) -> bool:
        """Check if label is a family label."""
        family_name = self.config.disambiguation_family_name
        if label == family_name:
            return True
        
        target_classes = self.config.disambiguation_classes
        if label in target_classes:
            return True
        
        return False
    
    def _disambiguate_track_family_label(
        self,
        final_label: str,
        final_confidence: float,
        candidates: List[Dict[str, Any]],
        track_id: int
    ):
        """Simplified mock for track-level disambiguation."""
        # Import the real function for testing
        from src.classifier.disambiguation_v2 import disambiguate_v2
        
        # Filter to closed ROIs only
        closed_candidates = [c for c in candidates if c.get('state') == 'closed']
        
        if not closed_candidates:
            # No closed ROIs - fallback
            fallback_label = self.config.disambiguation_classes[0]
            return fallback_label, final_confidence, 'low', {
                'disambiguation_applied': True,
                'disambiguation_reason': 'no_closed_rois_fallback',
                'original_family_label': final_label,
                'fallback_used': True
            }
        
        # Select best closed ROI
        best_closed_roi = max(
            closed_candidates,
            key=lambda c: (c.get('trust', 0), c.get('confidence', 0), c.get('sharpness', 0))
        )
        
        bbox = best_closed_roi.get('bbox')
        if bbox is None:
            fallback_label = self.config.disambiguation_classes[0]
            return fallback_label, final_confidence, 'low', {
                'disambiguation_applied': True,
                'disambiguation_reason': 'no_bbox_fallback',
                'original_family_label': final_label,
                'fallback_used': True
            }
        
        # Run disambiguation
        result = disambiguate_v2(
            original_label=final_label,
            confidence=final_confidence,
            bbox=bbox,
            is_open=False,
            config=self.config,
            context={'track_id': track_id, 'track_level': True}
        )
        
        metadata = {
            'disambiguation_applied': True,
            'disambiguation_reason': result.reason,
            'original_family_label': final_label,
            'resolved_label': result.label,
            'resolved_confidence': result.confidence,
            'confidence_tier': result.confidence_tier,
            'best_closed_roi_trust': best_closed_roi.get('trust', 0),
            'total_closed_rois': len(closed_candidates)
        }
        
        return result.label, result.confidence, result.confidence_tier, metadata


# =============================================================================
# Test Cases
# =============================================================================

class TestTrackDisambiguation(unittest.TestCase):
    """Test track-level disambiguation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MockTrackingConfig()
        self.service = MockClassifierService(self.config)
    
    def test_is_family_label_explicit(self):
        """Test detection of explicit family label."""
        self.assertTrue(self.service._is_family_label('Brown_Orange_Family'))
    
    def test_is_family_label_member(self):
        """Test detection of family member labels."""
        self.assertTrue(self.service._is_family_label('Brown_Orange_Overlay'))
        self.assertTrue(self.service._is_family_label('Brown_Orange_Small'))
    
    def test_is_family_label_non_family(self):
        """Test non-family labels are not detected as family."""
        self.assertFalse(self.service._is_family_label('White_Bread'))
        self.assertFalse(self.service._is_family_label('Whole_Wheat'))
        self.assertFalse(self.service._is_family_label('Unknown'))
    
    def test_disambiguate_family_label_with_closed_rois_small(self):
        """Test disambiguation of family label with closed ROIs -> Small."""
        candidates = [
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.8,
                'state': 'closed',
                'bbox': (100, 100, 180, 200),  # Area = 8000 (< 9000 -> Small)
                'trust': 0.9,
                'sharpness': 500
            },
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.7,
                'state': 'open',
                'bbox': (100, 100, 200, 250),
                'trust': 0.7,
                'sharpness': 400
            }
        ]
        
        resolved_label, resolved_conf, tier, metadata = self.service._disambiguate_track_family_label(
            final_label='Brown_Orange_Family',
            final_confidence=0.75,
            candidates=candidates,
            track_id=1
        )
        
        # Should resolve to Small based on area
        self.assertEqual(resolved_label, 'Brown_Orange_Small')
        self.assertEqual(metadata['original_family_label'], 'Brown_Orange_Family')
        self.assertTrue(metadata['disambiguation_applied'])
        self.assertFalse(metadata.get('fallback_used', False))
    
    def test_disambiguate_family_label_with_closed_rois_overlay(self):
        """Test disambiguation of family label with closed ROIs -> Overlay."""
        candidates = [
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.8,
                'state': 'closed',
                'bbox': (100, 100, 220, 250),  # Area = 18000 (> 11000 -> Overlay)
                'trust': 0.9,
                'sharpness': 500
            },
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.7,
                'state': 'open',
                'bbox': (100, 100, 200, 250),
                'trust': 0.7,
                'sharpness': 400
            }
        ]
        
        resolved_label, resolved_conf, tier, metadata = self.service._disambiguate_track_family_label(
            final_label='Brown_Orange_Family',
            final_confidence=0.75,
            candidates=candidates,
            track_id=1
        )
        
        # Should resolve to Overlay based on area
        self.assertEqual(resolved_label, 'Brown_Orange_Overlay')
        self.assertEqual(metadata['original_family_label'], 'Brown_Orange_Family')
        self.assertTrue(metadata['disambiguation_applied'])
        self.assertFalse(metadata.get('fallback_used', False))
    
    def test_disambiguate_family_label_no_closed_rois(self):
        """Test disambiguation with no closed ROIs -> fallback."""
        candidates = [
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.8,
                'state': 'open',  # All open
                'bbox': (100, 100, 180, 200),
                'trust': 0.9,
                'sharpness': 500
            },
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.7,
                'state': 'open',  # All open
                'bbox': (100, 100, 200, 250),
                'trust': 0.7,
                'sharpness': 400
            }
        ]
        
        resolved_label, resolved_conf, tier, metadata = self.service._disambiguate_track_family_label(
            final_label='Brown_Orange_Family',
            final_confidence=0.75,
            candidates=candidates,
            track_id=1
        )
        
        # Should fallback to default (Overlay)
        self.assertEqual(resolved_label, 'Brown_Orange_Overlay')
        self.assertEqual(tier, 'low')  # Fallback is low confidence
        self.assertTrue(metadata['fallback_used'])
        self.assertEqual(metadata['disambiguation_reason'], 'no_closed_rois_fallback')
    
    def test_disambiguate_family_label_no_bbox(self):
        """Test disambiguation with closed ROIs but no bbox -> fallback."""
        candidates = [
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.8,
                'state': 'closed',
                'bbox': None,  # No bbox!
                'trust': 0.9,
                'sharpness': 500
            }
        ]
        
        resolved_label, resolved_conf, tier, metadata = self.service._disambiguate_track_family_label(
            final_label='Brown_Orange_Family',
            final_confidence=0.75,
            candidates=candidates,
            track_id=1
        )
        
        # Should fallback to default
        self.assertEqual(resolved_label, 'Brown_Orange_Overlay')
        self.assertEqual(tier, 'low')
        self.assertTrue(metadata['fallback_used'])
        self.assertEqual(metadata['disambiguation_reason'], 'no_bbox_fallback')
    
    def test_disambiguate_selects_best_closed_roi(self):
        """Test that disambiguation selects the best closed ROI."""
        candidates = [
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.6,
                'state': 'closed',
                'bbox': (100, 100, 180, 200),  # Area = 8000 (Small)
                'trust': 0.5,  # Low trust
                'sharpness': 300
            },
            {
                'label': 'Brown_Orange_Family',
                'confidence': 0.8,
                'state': 'closed',
                'bbox': (100, 100, 220, 250),  # Area = 18000 (Overlay)
                'trust': 0.9,  # High trust - should be selected
                'sharpness': 600
            }
        ]
        
        resolved_label, resolved_conf, tier, metadata = self.service._disambiguate_track_family_label(
            final_label='Brown_Orange_Family',
            final_confidence=0.75,
            candidates=candidates,
            track_id=1
        )
        
        # Should use the high-trust ROI (Overlay)
        self.assertEqual(resolved_label, 'Brown_Orange_Overlay')
        self.assertEqual(metadata['best_closed_roi_trust'], 0.9)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    if PYTEST_AVAILABLE:
        pytest.main([__file__, '-v'])
    else:
        # Run with unittest
        unittest.main()
