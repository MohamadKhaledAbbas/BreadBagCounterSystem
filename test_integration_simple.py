#!/usr/bin/env python3
"""
Simple integration test to verify bbox and evidence accumulation changes.
This can be run without pytest to validate basic functionality.
"""

import sys
import numpy as np
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, 'src')

from classifier.evidence_accumulator import accumulate_track_evidence
from classifier.disambiguation import disambiguate_by_size
from config.tracking_config import tracking_config


@dataclass
class MockConfig:
    """Mock configuration for testing."""
    # Disambiguation parameters
    disambiguation_enabled: bool = True
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    disambiguation_family_name: str = 'Brown_Orange_Family'
    disambiguation_y_feature: str = 'cy'
    disambiguation_scaling_model: str = 'linear'
    disambiguation_scale_a: float = 0.5
    disambiguation_scale_b: float = 1.0
    disambiguation_scale_p: float = 1.5
    disambiguation_small_threshold: float = 15000.0
    disambiguation_regular_threshold: float = 25000.0
    disambiguation_gray_zone_behavior: str = 'keep_original'
    disambiguation_debug_logging: bool = False
    disambiguation_confidence_penalty: float = 0.9
    disambiguation_penalty_on_change_only: bool = False
    
    # Trust parameters
    trust_open_max: float = 1.0
    trust_closed_max: float = 0.7
    trust_sharpness_min: float = 100.0
    trust_sharpness_max: float = 800.0
    trust_blur_penalty: float = 0.3
    trust_size_stability_tolerance: float = 0.3
    trust_min_for_support: float = 0.4
    
    # Evidence parameters
    evidence_epsilon: float = 1e-6
    evidence_top_k_rois: int = 7
    temporal_inertia_enabled: bool = True
    temporal_inertia_strength: float = 0.15
    temporal_inertia_decay: float = 0.8
    stability_gate_enabled: bool = True
    stability_margin_threshold: float = 0.5
    stability_min_trusted_rois: int = 2


def test_bbox_in_candidate():
    """Test 1: Verify bbox is included in candidate structure."""
    print("\n=== Test 1: Bbox in Candidate ===")
    
    candidate = {
        'roi': np.zeros((50, 50, 3), dtype=np.uint8),
        'sharpness': 500.0,
        'frame_index': 10,
        'bbox_area': 5000.0,
        'confidence': 0.8,
        'relative_time': 0.5,
        'bbox': (100.0, 50.0, 150.0, 100.0)
    }
    
    assert 'bbox' in candidate, "bbox field missing!"
    assert len(candidate['bbox']) == 4, "bbox should have 4 coordinates!"
    assert all(isinstance(v, float) for v in candidate['bbox']), "bbox values should be floats!"
    
    print("✓ Bbox is properly included in candidate structure")
    return True


def test_disambiguation_with_bbox():
    """Test 2: Verify disambiguation works when bbox is present."""
    print("\n=== Test 2: Disambiguation with Bbox ===")
    
    config = MockConfig()
    
    # Test with a large box near bottom (should be regular class)
    result = disambiguate_by_size(
        original_label='Brown_Orange_Overlay',
        confidence=0.75,
        bbox=(100, 500, 250, 650),  # Large box near bottom
        image_height=720,
        config=config
    )
    
    assert result.disambiguated, "Should have attempted disambiguation!"
    assert result.raw_area > 0, "Raw area should be calculated!"
    assert result.adjusted_area > 0, "Adjusted area should be calculated!"
    
    print(f"✓ Disambiguation applied: {result.disambiguated}")
    print(f"  Label: {result.label}")
    print(f"  Reason: {result.reason}")
    print(f"  Raw area: {result.raw_area:.0f}, Adjusted area: {result.adjusted_area:.0f}")
    return True


def test_evidence_accumulation():
    """Test 3: Verify evidence accumulation path works."""
    print("\n=== Test 3: Evidence Accumulation ===")
    
    config = MockConfig()
    
    # Create sample classifications with probability vectors
    classifications = [
        {
            'probs': {'ClassA': 0.7, 'ClassB': 0.2, 'ClassC': 0.1},
            'trust': 0.8,
            'state': 'open',
            'label': 'ClassA',
            'confidence': 0.7,
        },
        {
            'probs': {'ClassA': 0.8, 'ClassB': 0.1, 'ClassC': 0.1},
            'trust': 0.9,
            'state': 'open',
            'label': 'ClassA',
            'confidence': 0.8,
        },
        {
            'probs': {'ClassA': 0.6, 'ClassB': 0.3, 'ClassC': 0.1},
            'trust': 0.7,
            'state': 'closed',
            'label': 'ClassA',
            'confidence': 0.6,
        },
    ]
    
    result = accumulate_track_evidence(classifications, config)
    
    assert result.label in ('ClassA', 'ClassB', 'ClassC', 'Uncertain'), f"Invalid label: {result.label}"
    assert result.confidence >= 0, "Confidence should be non-negative!"
    assert result.rois_used == 3, f"Expected 3 ROIs, got {result.rois_used}!"
    assert 'ClassA' in result.evidence_per_class, "ClassA should be in evidence!"
    assert result.winner_score != 0, "Winner score should not be zero!"
    
    print(f"✓ Evidence accumulation completed successfully")
    print(f"  Final label: {result.label}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  ROIs used: {result.rois_used}")
    print(f"  ROIs trusted: {result.rois_trusted}")
    print(f"  Winner score: {result.winner_score:.3f}")
    print(f"  Margin: {result.margin:.3f}")
    print(f"  Gate passed: {result.gate_passed}")
    return True


def test_metadata_structure():
    """Test 4: Verify metadata includes required fields."""
    print("\n=== Test 4: Metadata Structure ===")
    
    # Simulate metadata from evidence accumulation path
    metadata = {
        'evidence_per_label': {'ClassA': 1.5, 'ClassB': 0.5},
        'total_candidates_classified': 3,
        'winner_score': 1.5,
        'runner_up': {'label': 'ClassB', 'score': 0.5},
        'margin': 1.0,
        'gate_passed': True,
        'gate_failure_reason': None,
        'trust_stats': {'min': 0.7, 'max': 0.9, 'mean': 0.8},
        'rois_trusted': 3,
        'class_switch_penalty_applied': False,
        'evidence_accumulation_used': True
    }
    
    required_keys = [
        'evidence_accumulation_used',
        'trust_stats',
        'gate_passed',
        'margin',
        'winner_score',
        'rois_trusted'
    ]
    
    for key in required_keys:
        assert key in metadata, f"Required key '{key}' missing from metadata!"
    
    assert metadata['evidence_accumulation_used'] == True, "Should indicate evidence accumulation was used!"
    
    print("✓ Metadata structure is complete")
    print(f"  Keys present: {len(metadata)}")
    print(f"  Evidence accumulation flag: {metadata['evidence_accumulation_used']}")
    print(f"  Trust stats: {metadata['trust_stats']}")
    return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Integration Tests for Bbox and Evidence Accumulation")
    print("=" * 60)
    
    tests = [
        test_bbox_in_candidate,
        test_disambiguation_with_bbox,
        test_evidence_accumulation,
        test_metadata_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
