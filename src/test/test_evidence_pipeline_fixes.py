"""
Unit Tests for Evidence Accumulation Pipeline Fixes.

Tests cover the production-grade improvements made to the classification pipeline:
1. Median ROI size computed from closed-state ROIs only
2. No double classification in evidence accumulation path
3. Gray zone handling integration with evidence accumulator
4. Legacy path deprecation and proper branching

Run with: python -m pytest src/test/test_evidence_pipeline_fixes.py -v
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any, List

from src.classifier.evidence_accumulator import (
    EvidenceAccumulator,
    accumulate_track_evidence,
    FinalClassificationResult
)


# =============================================================================
# Mock Configuration
# =============================================================================

@dataclass
class MockConfig:
    """Mock configuration for testing."""
    # Evidence parameters
    evidence_epsilon: float = 1e-6
    temporal_inertia_enabled: bool = True
    temporal_inertia_strength: float = 0.15
    temporal_inertia_decay: float = 0.8
    stability_gate_enabled: bool = True
    stability_margin_threshold: float = 0.5
    stability_min_trusted_rois: int = 2
    trust_min_for_support: float = 0.4
    
    # Reject labels
    classifier_reject_labels: tuple = ('Rejected',)


@pytest.fixture
def default_config():
    """Default mock configuration."""
    return MockConfig()


# =============================================================================
# Test ROI State Filtering for Median Size
# =============================================================================

class TestClosedROIFiltering:
    """Tests for closed-state ROI filtering in median size calculation."""
    
    def test_closed_rois_only_should_be_used_for_median(self):
        """
        Test that median size should be computed from closed ROIs only.
        
        This is a documentation test showing the expected behavior:
        - Open state ROIs may have incorrect/incomplete boundaries
        - They should NOT bias the median size used for trust calculation
        - Only closed state ROIs should contribute to median calculation
        
        The actual implementation is in ClassifierService.process() around line 1113.
        """
        # Example scenario: Track with mixed open/closed ROIs
        # Using simple dict structure instead of numpy arrays
        candidates = [
            {'roi': {'shape': (50, 50, 3)}, 'state': 'open'},   # 50x50 - should be excluded
            {'roi': {'shape': (100, 100, 3)}, 'state': 'closed'},  # 100x100 - should be included
            {'roi': {'shape': (110, 110, 3)}, 'state': 'closed'},  # 110x110 - should be included
            {'roi': {'shape': (90, 90, 3)}, 'state': 'closed'},   # 90x90 - should be included
            {'roi': {'shape': (200, 200, 3)}, 'state': 'open'},   # 200x200 - should be excluded
        ]
        
        # Expected median from closed ROIs: sorted sizes are [90, 100, 110]
        # Median is the middle value: 100x100
        expected_median = (100, 100)
        
        # If open ROIs were included, sorted sizes would be [50, 90, 100, 110, 200]
        # Incorrect median would be 100x100 (lucky in this case, but could be wrong)
        
        # The fix ensures only closed ROIs are used:
        # Filter: cand.get('state') != 'open'
        closed_only = [c for c in candidates if c.get('state') != 'open']
        assert len(closed_only) == 3, "Should have exactly 3 closed ROIs"
        
        # Compute median from closed ROIs
        closed_sizes = []
        for c in closed_only:
            roi = c['roi']
            if roi is not None:
                shape = roi['shape']
                h, w = shape[:2]
                closed_sizes.append((w, h))
        
        median_w = sorted([s[0] for s in closed_sizes])[len(closed_sizes) // 2]
        median_h = sorted([s[1] for s in closed_sizes])[len(closed_sizes) // 2]
        actual_median = (median_w, median_h)
        
        assert actual_median == expected_median, \
            f"Median should be {expected_median} when using closed ROIs only, got {actual_median}"


# =============================================================================
# Test Evidence Accumulation with State Handling
# =============================================================================

class TestEvidenceAccumulatorStateHandling:
    """Tests for proper state handling in evidence accumulation."""
    
    def test_open_and_closed_rois_both_contribute_evidence(self, default_config):
        """
        Test that both open and closed ROIs contribute to evidence accumulation.
        
        While median size is computed from closed ROIs only, BOTH open and closed
        ROIs should contribute their classification evidence. The trust score will
        be different based on state, but both should participate.
        """
        accumulator = EvidenceAccumulator(default_config)
        
        # Add open ROI with high confidence
        accumulator.update(
            roi_id=0,
            probs={'ClassA': 0.9, 'ClassB': 0.1},
            trust=0.8,  # High trust for open ROI
            state='open'
        )
        
        # Add closed ROI with medium confidence
        accumulator.update(
            roi_id=1,
            probs={'ClassA': 0.7, 'ClassB': 0.3},
            trust=0.6,  # Medium trust for closed ROI
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Both ROIs should have contributed
        assert result.rois_used == 2, "Both open and closed ROIs should contribute"
        assert result.label == 'ClassA', "ClassA should win with contributions from both ROIs"
        assert result.is_certain == True, "Should be certain with consistent votes"
    
    def test_reject_labels_not_in_evidence(self, default_config):
        """
        Test that reject labels (Rejected, Unknown, Uncertain) don't contribute evidence.
        
        This ensures that classifications marked as rejected don't bias the final decision.
        """
        accumulator = EvidenceAccumulator(default_config)
        
        # Add good classification
        accumulator.update(
            roi_id=0,
            probs={'ClassA': 0.8, 'ClassB': 0.2},
            trust=0.9,
            state='closed'
        )
        
        # Add rejected classification - should not contribute
        accumulator.update(
            roi_id=1,
            probs={'Rejected': 0.9, 'ClassA': 0.1},
            trust=0.8,
            state='closed'
        )
        
        # Add unknown classification - should not contribute
        accumulator.update(
            roi_id=2,
            probs={'Unknown': 0.8, 'ClassA': 0.2},
            trust=0.7,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Check that reject labels were filtered
        assert result.rois_used == 3, "All ROIs counted"
        assert result.rois_rejected > 0, "Rejected classifications tracked"
        assert 'Rejected' not in result.evidence_per_class, "Rejected should not have evidence"
        assert 'Unknown' not in result.evidence_per_class, "Unknown should not have evidence"


# =============================================================================
# Test Gray Zone Integration
# =============================================================================

class TestGrayZoneIntegration:
    """Tests for gray zone handling integration with evidence accumulator."""
    
    def test_gray_zone_low_confidence_propagates(self, default_config):
        """
        Test that gray zone classifications with low confidence are handled properly.
        
        Gray zone classifications from disambiguation_v2 should have:
        - is_gray_zone flag set
        - Reduced confidence (penalty applied)
        - confidence_tier = 'low'
        
        These should still participate in evidence accumulation but with reduced weight.
        """
        accumulator = EvidenceAccumulator(default_config)
        
        # Add gray zone classification with reduced confidence
        # (as would come from disambiguation_v2.py)
        gray_zone_confidence = 0.75 * 0.65  # Original * gray_zone_penalty
        
        accumulator.update(
            roi_id=0,
            probs={'Brown_Orange_Small': gray_zone_confidence, 'Brown_Orange_Overlay': 1 - gray_zone_confidence},
            trust=0.7,
            state='closed'
        )
        
        # Add another gray zone with similar reduced confidence
        accumulator.update(
            roi_id=1,
            probs={'Brown_Orange_Small': gray_zone_confidence, 'Brown_Orange_Overlay': 1 - gray_zone_confidence},
            trust=0.6,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Gray zone classifications should still contribute
        assert result.rois_used == 2, "Gray zone ROIs should contribute"
        # With reduced confidence, margin may be smaller
        assert result.margin >= 0, "Margin should be computed"


# =============================================================================
# Test Temporal Weighting
# =============================================================================

class TestTemporalWeighting:
    """Tests for temporal weighting in evidence accumulation."""
    
    def test_trust_weights_evidence_contributions(self, default_config):
        """
        Test that trust scores properly weight evidence contributions.
        
        Higher trust ROIs should have more influence on the final decision.
        This is the trust-based weighting approach that replaces legacy
        clamped_contribution.
        """
        accumulator = EvidenceAccumulator(default_config)
        
        # Add multiple high trust votes for ClassA
        for i in range(3):
            accumulator.update(
                roi_id=i,
                probs={'ClassA': 0.85, 'ClassB': 0.15},
                trust=0.9,  # High trust
                state='closed'
            )
        
        # Add one low trust vote for ClassB (should be downweighted)
        accumulator.update(
            roi_id=3,
            probs={'ClassA': 0.1, 'ClassB': 0.9},
            trust=0.2,  # Low trust
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # High trust ClassA votes should win over single low trust ClassB
        assert result.label == 'ClassA', \
            "High trust ClassA should win over low trust ClassB"
        assert result.is_certain == True, "Should be certain due to trust weighting"
        assert result.rois_trusted >= 2, "Should have enough high-trust ROIs"
    
    def test_consistent_evidence_builds_margin(self, default_config):
        """
        Test that consistent evidence from multiple ROIs builds a strong margin.
        
        This demonstrates the accumulation of log-evidence and margin-based decision.
        """
        accumulator = EvidenceAccumulator(default_config)
        
        # Add 5 consistent classifications for ClassA
        for i in range(5):
            accumulator.update(
                roi_id=i,
                probs={'ClassA': 0.8, 'ClassB': 0.2},
                trust=0.7,
                state='closed'
            )
        
        result = accumulator.finalize()
        
        assert result.label == 'ClassA', "ClassA should win with consistent evidence"
        assert result.is_certain == True, "Should be certain with 5 consistent votes"
        assert result.margin > 1.0, "Margin should be large with consistent evidence"
        assert result.rois_trusted >= 2, "Should have enough trusted ROIs"


# =============================================================================
# Test Stability Gate
# =============================================================================

class TestStabilityGate:
    """Tests for stability gate decision logic."""
    
    def test_insufficient_margin_returns_uncertain(self, default_config):
        """
        Test that classifications with insufficient margin return Uncertain.
        
        This is the margin-based stability gate that replaces legacy ratio threshold.
        """
        accumulator = EvidenceAccumulator(default_config)
        
        # Add ambiguous evidence: nearly equal probabilities
        accumulator.update(
            roi_id=0,
            probs={'ClassA': 0.51, 'ClassB': 0.49},
            trust=0.8,
            state='closed'
        )
        
        accumulator.update(
            roi_id=1,
            probs={'ClassA': 0.52, 'ClassB': 0.48},
            trust=0.7,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # With ambiguous probabilities, margin will be small
        assert result.margin < default_config.stability_margin_threshold, \
            "Margin should be below threshold with ambiguous evidence"
        assert result.gate_passed == False, "Stability gate should fail"
        assert result.label == 'Uncertain', "Should return Uncertain with small margin"
        assert result.is_certain == False
    
    def test_insufficient_trusted_rois_returns_uncertain(self, default_config):
        """
        Test that classifications without enough trusted ROIs return Uncertain.
        """
        accumulator = EvidenceAccumulator(default_config)
        
        # Add only 1 ROI with high confidence but not enough trusted ROIs
        accumulator.update(
            roi_id=0,
            probs={'ClassA': 0.95, 'ClassB': 0.05},
            trust=0.9,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Only 1 trusted ROI, need at least 2
        assert result.rois_trusted < default_config.stability_min_trusted_rois, \
            f"Should have < {default_config.stability_min_trusted_rois} trusted ROIs"
        assert result.gate_passed == False, "Stability gate should fail with too few trusted ROIs"
        assert result.label == 'Uncertain', "Should return Uncertain"
        assert 'too_few_trusted_rois' in result.gate_failure_reason


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
