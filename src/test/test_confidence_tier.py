"""
Unit Tests for Confidence Tier Functionality.

Tests cover:
1. Gray zone classifications are marked as low confidence
2. Validation penalties trigger low confidence
3. Family label resolution triggers low confidence
4. Track-level confidence tier aggregation
5. All classes (not just Brown_Orange_Family) are properly flagged

Run with: python src/test/test_confidence_tier.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.classifier.disambiguation_v2 import (
    disambiguate_v2,
    resolve_gray_zone,
    DisambiguationV2Result,
)


# =============================================================================
# Mock Configuration
# =============================================================================

@dataclass
class MockConfig:
    """Mock configuration for testing."""
    disambiguation_v2_enabled: bool = True
    disambiguation_enabled: bool = True
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    disambiguation_small_threshold: float = 9000.0
    disambiguation_regular_threshold: float = 11000.0
    disambiguation_gray_zone_behavior: str = 'keep_original'
    disambiguation_family_name: str = 'Brown_Orange_Family'
    disambiguation_confidence_penalty: float = 0.9
    disambiguation_penalty_on_change_only: bool = False
    disambiguation_v2_debug_logging: bool = False
    disambiguation_v2_min_aspect_ratio: float = 0.3
    disambiguation_v2_max_aspect_ratio: float = 3.0
    disambiguation_v2_aspect_ratio_penalty: float = 0.3
    disambiguation_v2_min_realistic_area: float = 1000.0
    disambiguation_v2_max_realistic_area: float = 100000.0
    disambiguation_v2_unrealistic_area_penalty: float = 0.5
    disambiguation_v2_very_small_threshold: float = 5000.0
    disambiguation_v2_large_threshold: float = 25000.0
    disambiguation_v2_gray_zone_confidence_threshold: float = 0.6


# =============================================================================
# Test Suite
# =============================================================================

def test_gray_zone_marked_as_low_confidence():
    """Test that gray zone classifications are marked as low confidence."""
    config = MockConfig()
    
    # Gray zone area (between 9000 and 11000)
    bbox_gray = (100, 100, 200, 200)  # 100x100 = 10000 (gray zone)
    
    result = disambiguate_v2(
        original_label='Brown_Orange_Overlay',
        confidence=0.7,
        bbox=bbox_gray,
        is_open=False,
        config=config,
        context={'track_id': 1}
    )
    
    assert result.confidence_tier == 'low', \
        f"Gray zone should be marked as low confidence, got {result.confidence_tier}"
    assert 'gray_zone' in result.metadata.get('confidence_tier_reason', ''), \
        "Reason should mention gray_zone"
    print("✓ test_gray_zone_marked_as_low_confidence PASSED")


def test_validation_penalty_triggers_low_confidence():
    """Test that validation penalties trigger low confidence."""
    config = MockConfig()
    
    # Suspicious aspect ratio (very narrow)
    bbox = (100, 100, 110, 200)  # 10x100 = AR 0.1 (< 0.3 minimum)
    
    result = disambiguate_v2(
        original_label='Brown_Orange_Small',
        confidence=0.8,
        bbox=bbox,
        is_open=False,
        config=config,
        context={'track_id': 2}
    )
    
    assert result.confidence_tier == 'low', \
        f"Validation penalty should trigger low confidence, got {result.confidence_tier}"
    assert 'validation_penalty' in result.metadata.get('confidence_tier_reason', ''), \
        "Reason should mention validation_penalty"
    print("✓ test_validation_penalty_triggers_low_confidence PASSED")


def test_label_changed_triggers_low_confidence():
    """Test that label changes trigger low confidence."""
    config = MockConfig()
    
    # Very small area - will change label
    bbox = (100, 100, 150, 140)  # 50x40 = 2000 (very small)
    
    result = disambiguate_v2(
        original_label='Brown_Orange_Overlay',  # Will be changed to Small
        confidence=0.7,
        bbox=bbox,
        is_open=False,
        config=config,
        context={'track_id': 3}
    )
    
    assert result.confidence_tier == 'low', \
        f"Label change should trigger low confidence, got {result.confidence_tier}"
    assert result.label == 'Brown_Orange_Small', \
        f"Label should be changed to Small, got {result.label}"
    assert 'label_changed' in result.metadata.get('confidence_tier_reason', ''), \
        "Reason should mention label_changed"
    print("✓ test_label_changed_triggers_low_confidence PASSED")


def test_family_label_resolved_triggers_low_confidence():
    """Test that family label resolution triggers low confidence."""
    config = MockConfig()
    
    # Medium area with family label
    bbox = (100, 100, 200, 200)  # 100x100 = 10000 (gray zone)
    
    result = disambiguate_v2(
        original_label='Brown_Orange_Family',  # Family label needs resolution
        confidence=0.7,
        bbox=bbox,
        is_open=False,
        config=config,
        context={'track_id': 4}
    )
    
    assert result.confidence_tier == 'low', \
        f"Family label resolution should trigger low confidence, got {result.confidence_tier}"
    # Should return a specific class, not family
    assert result.label in config.disambiguation_classes, \
        f"Should resolve to specific class, got {result.label}"
    print("✓ test_family_label_resolved_triggers_low_confidence PASSED")


def test_clear_classification_high_confidence():
    """Test that clear classifications remain high confidence."""
    config = MockConfig()
    
    # Very large area - clearly regular
    bbox = (100, 100, 300, 300)  # 200x200 = 40000 (large)
    
    result = disambiguate_v2(
        original_label='Brown_Orange_Overlay',
        confidence=0.9,
        bbox=bbox,
        is_open=False,
        config=config,
        context={'track_id': 5}
    )
    
    assert result.confidence_tier == 'high', \
        f"Clear classification should remain high confidence, got {result.confidence_tier}"
    assert result.label == 'Brown_Orange_Overlay', \
        f"Label should remain Overlay, got {result.label}"
    print("✓ test_clear_classification_high_confidence PASSED")


def test_resolve_gray_zone_never_returns_uncertain():
    """Test that resolve_gray_zone never returns 'Uncertain', always a specific class."""
    config = MockConfig()
    
    # Set behavior to 'uncertain' (which should now pick best match)
    config.disambiguation_gray_zone_behavior = 'uncertain'
    
    size_metadata = {
        'raw_area': 10000.0,
        'thresholds': {
            'small': 9000.0,
            'regular': 11000.0
        }
    }
    
    target_classes = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    
    resolved_label, reason = resolve_gray_zone(
        original_label='Brown_Orange_Family',
        confidence=0.5,
        size_bin_metadata=size_metadata,
        config=config,
        target_classes=target_classes
    )
    
    assert resolved_label != 'Uncertain', \
        f"Should never return 'Uncertain', got {resolved_label}"
    assert resolved_label in target_classes, \
        f"Should return a specific class, got {resolved_label}"
    assert 'resolved' in reason, \
        f"Reason should mention resolution, got {reason}"
    print("✓ test_resolve_gray_zone_never_returns_uncertain PASSED")


def test_resolve_gray_zone_never_returns_family_label():
    """Test that resolve_gray_zone never returns family label."""
    config = MockConfig()
    
    # Test all gray zone behaviors
    behaviors = ['keep_original', 'uncertain', 'prefer_small', 'prefer_regular', 'use_confidence']
    
    size_metadata = {
        'raw_area': 10000.0,
        'thresholds': {
            'small': 9000.0,
            'regular': 11000.0
        }
    }
    
    target_classes = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    family_name = 'Brown_Orange_Family'
    
    for behavior in behaviors:
        config.disambiguation_gray_zone_behavior = behavior
        
        resolved_label, reason = resolve_gray_zone(
            original_label=family_name,  # Start with family label
            confidence=0.7,
            size_bin_metadata=size_metadata,
            config=config,
            target_classes=target_classes,
            family_name=family_name
        )
        
        assert resolved_label != family_name, \
            f"Behavior '{behavior}' should not return family label, got {resolved_label}"
        assert resolved_label in target_classes, \
            f"Behavior '{behavior}' should return specific class, got {resolved_label}"
    
    print("✓ test_resolve_gray_zone_never_returns_family_label PASSED")


def test_open_state_skips_disambiguation():
    """Test that open state ROIs skip disambiguation and remain high confidence."""
    config = MockConfig()
    
    bbox = (100, 100, 200, 200)  # Any bbox
    
    result = disambiguate_v2(
        original_label='Brown_Orange_Overlay',
        confidence=0.7,
        bbox=bbox,
        is_open=True,  # Open state
        config=config,
        context={'track_id': 6}
    )
    
    assert not result.disambiguated, \
        "Open state should skip disambiguation"
    assert 'open_state' in result.reason, \
        "Reason should mention open_state"
    # Default confidence_tier should be 'high' when skipped
    assert result.confidence_tier == 'high', \
        f"Skipped disambiguation should keep default tier, got {result.confidence_tier}"
    print("✓ test_open_state_skips_disambiguation PASSED")


def test_non_family_class_skips_disambiguation():
    """Test that non-family classes skip disambiguation."""
    config = MockConfig()
    
    bbox = (100, 100, 200, 200)
    
    result = disambiguate_v2(
        original_label='Red_Bag',  # Not in family
        confidence=0.8,
        bbox=bbox,
        is_open=False,
        config=config,
        context={'track_id': 7}
    )
    
    assert not result.disambiguated, \
        "Non-family class should skip disambiguation"
    assert 'not_target_family' in result.reason, \
        "Reason should mention not_target_family"
    assert result.label == 'Red_Bag', \
        "Label should remain unchanged"
    print("✓ test_non_family_class_skips_disambiguation PASSED")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_gray_zone_marked_as_low_confidence,
        test_validation_penalty_triggers_low_confidence,
        test_label_changed_triggers_low_confidence,
        test_family_label_resolved_triggers_low_confidence,
        test_clear_classification_high_confidence,
        test_resolve_gray_zone_never_returns_uncertain,
        test_resolve_gray_zone_never_returns_family_label,
        test_open_state_skips_disambiguation,
        test_non_family_class_skips_disambiguation,
    ]
    
    print("=" * 70)
    print("Running Confidence Tier Tests")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    errors = []
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"✗ {test_func.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((test_func.__name__, f"Error: {e}"))
            print(f"✗ {test_func.__name__} ERROR: {e}")
    
    print()
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 70)
    
    if errors:
        print("\nFailed Tests:")
        for test_name, error_msg in errors:
            print(f"  - {test_name}: {error_msg}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
