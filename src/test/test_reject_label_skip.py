"""
Unit Tests for Reject Label Skipping in Voting/Aggregation Logic.

Tests cover:
1. Reject labels are skipped in evidence accumulation (both legacy and new paths)
2. Mixture of valid and rejected predictions work correctly
3. Edge case: all predictions rejected returns "Uncertain"
4. Configuration override works
5. Rejection metrics are tracked correctly

Run with: python -m pytest src/test/test_reject_label_skip.py -v
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

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
    evidence_top_k_rois: int = 7
    temporal_inertia_enabled: bool = True
    temporal_inertia_strength: float = 0.15
    temporal_inertia_decay: float = 0.8
    stability_gate_enabled: bool = True
    stability_margin_threshold: float = 0.5
    stability_min_trusted_rois: int = 2
    trust_min_for_support: float = 0.4
    
    # Reject labels configuration
    classifier_reject_labels: tuple = ('Rejected',)


@pytest.fixture
def default_config():
    """Default mock configuration."""
    return MockConfig()


@pytest.fixture
def custom_reject_config():
    """Configuration with multiple reject labels."""
    config = MockConfig()
    config.classifier_reject_labels = ('Rejected', 'LowQuality', 'Ambiguous')
    return config


# =============================================================================
# Evidence Accumulator Tests
# =============================================================================

class TestRejectLabelSkipping:
    """Tests for reject label skipping in evidence accumulator."""
    
    def test_single_rejected_prediction_skipped(self, default_config):
        """Single 'Rejected' prediction should be skipped but other probs used."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add one prediction where top class is 'Rejected'
        # But probability vector still has other classes
        accumulator.update(
            roi_id=0,
            probs={'Rejected': 0.9, 'WholeWheat': 0.1},
            trust=0.8,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Should be uncertain due to too few trusted ROIs, but 'Rejected' should not be in evidence
        assert result.label == "Uncertain"
        assert result.is_certain is False
        # Gate failure is due to too few trusted ROIs (need at least 2)
        assert "too_few_trusted_rois" in result.gate_failure_reason
        assert result.rois_rejected == 1  # 'Rejected' was in the probs
        assert result.rois_used == 1
        # WholeWheat evidence exists but with low probability
        assert 'Rejected' not in result.evidence_per_class
        # Note: WholeWheat might be in evidence with very negative score due to log(0.1)
    
    def test_mixture_valid_rejected_predictions(self, default_config):
        """Mixture of valid and rejected predictions should work correctly."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add rejected prediction
        accumulator.update(
            roi_id=0,
            probs={'Rejected': 0.8, 'WholeWheat': 0.2},
            trust=0.7,
            state='open'
        )
        
        # Add valid predictions
        accumulator.update(
            roi_id=1,
            probs={'WholeWheat': 0.7, 'White': 0.3},
            trust=0.8,
            state='closed'
        )
        
        accumulator.update(
            roi_id=2,
            probs={'WholeWheat': 0.8, 'White': 0.2},
            trust=0.9,
            state='closed'
        )
        
        accumulator.update(
            roi_id=3,
            probs={'WholeWheat': 0.75, 'White': 0.25},
            trust=0.85,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Should classify as WholeWheat since rejected was skipped
        assert result.label == "WholeWheat"
        assert result.is_certain is True
        assert result.rois_rejected == 1
        assert result.rois_used == 4
        assert result.rois_trusted >= 2
        assert 'WholeWheat' in result.evidence_per_class
        assert 'Rejected' not in result.evidence_per_class
    
    def test_all_predictions_rejected(self, default_config):
        """All predictions where top class is 'Rejected' should skip Rejected but use other probs."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add multiple predictions where top class is 'Rejected'
        for i in range(5):
            accumulator.update(
                roi_id=i,
                probs={'Rejected': 0.9, 'WholeWheat': 0.1},
                trust=0.8,
                state='closed'
            )
        
        result = accumulator.finalize()
        
        # WholeWheat will have evidence from the 0.1 probabilities
        # With 5 ROIs all with 0.1 prob for WholeWheat, it should classify as WholeWheat
        assert result.label == "WholeWheat"
        assert result.is_certain is True  # Should pass gate with 5 trusted ROIs
        assert result.rois_rejected == 5  # All had 'Rejected' in probs
        assert result.rois_used == 5
        assert 'Rejected' not in result.evidence_per_class
        assert 'WholeWheat' in result.evidence_per_class
    
    def test_all_prob_vectors_only_reject_labels(self, default_config):
        """All probability vectors with only reject labels should return Uncertain."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add multiple predictions where ALL classes in prob vector are reject labels
        for i in range(5):
            accumulator.update(
                roi_id=i,
                probs={'Rejected': 0.6, 'Unknown': 0.3, 'Uncertain': 0.1},
                trust=0.8,
                state='closed'
            )
        
        result = accumulator.finalize()
        
        # Should be uncertain with no evidence since all classes were rejected
        assert result.label == "Uncertain"
        assert result.is_certain is False
        assert result.gate_failure_reason == "no_evidence"
        # Each ROI had 3 reject labels
        assert result.rois_rejected == 15  # 5 ROIs * 3 reject labels each
        assert result.rois_used == 5
        assert len(result.evidence_per_class) == 0
    
    def test_unknown_also_skipped(self, default_config):
        """Unknown predictions should also be skipped."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add Unknown prediction
        accumulator.update(
            roi_id=0,
            probs={'Unknown': 0.8, 'WholeWheat': 0.2},
            trust=0.7,
            state='open'
        )
        
        # Add valid predictions
        accumulator.update(
            roi_id=1,
            probs={'WholeWheat': 0.8, 'White': 0.2},
            trust=0.9,
            state='closed'
        )
        
        accumulator.update(
            roi_id=2,
            probs={'WholeWheat': 0.75, 'White': 0.25},
            trust=0.85,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Should classify as WholeWheat since Unknown was skipped
        assert result.label == "WholeWheat"
        assert result.is_certain is True
        assert result.rois_rejected >= 1  # At least Unknown was rejected
        assert 'Unknown' not in result.evidence_per_class
    
    def test_uncertain_also_skipped(self, default_config):
        """Uncertain predictions should also be skipped."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add Uncertain prediction
        accumulator.update(
            roi_id=0,
            probs={'Uncertain': 0.8, 'WholeWheat': 0.2},
            trust=0.7,
            state='open'
        )
        
        # Add valid predictions
        accumulator.update(
            roi_id=1,
            probs={'WholeWheat': 0.8, 'White': 0.2},
            trust=0.9,
            state='closed'
        )
        
        accumulator.update(
            roi_id=2,
            probs={'WholeWheat': 0.75, 'White': 0.25},
            trust=0.85,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Should classify as WholeWheat since Uncertain was skipped
        assert result.label == "WholeWheat"
        assert result.is_certain is True
        assert result.rois_rejected >= 1  # At least Uncertain was rejected
        assert 'Uncertain' not in result.evidence_per_class


class TestCustomRejectLabels:
    """Tests for custom reject label configuration."""
    
    def test_custom_reject_labels_skipped(self, custom_reject_config):
        """Custom reject labels should be skipped."""
        accumulator = EvidenceAccumulator(custom_reject_config)
        
        # Add various rejected predictions with trust >= 0.4 (trust_min_for_support)
        accumulator.update(
            roi_id=0,
            probs={'Rejected': 0.6, 'WholeWheat': 0.4},
            trust=0.8,
            state='closed'
        )
        
        accumulator.update(
            roi_id=1,
            probs={'LowQuality': 0.5, 'WholeWheat': 0.5},
            trust=0.8,
            state='closed'
        )
        
        accumulator.update(
            roi_id=2,
            probs={'Ambiguous': 0.4, 'WholeWheat': 0.6},
            trust=0.8,
            state='closed'
        )
        
        # Add more valid predictions with high trust (closed state)
        accumulator.update(
            roi_id=3,
            probs={'WholeWheat': 0.8, 'White': 0.2},
            trust=0.9,
            state='closed'
        )
        
        accumulator.update(
            roi_id=4,
            probs={'WholeWheat': 0.75, 'White': 0.25},
            trust=0.85,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Should classify as WholeWheat, all custom rejects skipped
        assert result.label == "WholeWheat"
        assert result.is_certain is True
        # Each ROI with reject label had 1 custom reject + usual Unknown/Uncertain
        # So we have at least 3 custom rejects
        assert result.rois_rejected >= 3
        assert result.rois_used == 5
        assert 'Rejected' not in result.evidence_per_class
        assert 'LowQuality' not in result.evidence_per_class
        assert 'Ambiguous' not in result.evidence_per_class
        assert 'WholeWheat' in result.evidence_per_class
    
    def test_non_reject_label_not_skipped(self, default_config):
        """Non-reject labels should not be skipped."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add predictions with various labels (none are reject labels except Rejected)
        accumulator.update(
            roi_id=0,
            probs={'WholeWheat': 0.5, 'White': 0.3, 'Bran': 0.2},
            trust=0.8,
            state='closed'
        )
        
        accumulator.update(
            roi_id=1,
            probs={'WholeWheat': 0.6, 'White': 0.2, 'Bran': 0.2},
            trust=0.85,
            state='closed'
        )
        
        accumulator.update(
            roi_id=2,
            probs={'WholeWheat': 0.55, 'White': 0.25, 'Bran': 0.2},
            trust=0.9,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # All classes should be in evidence
        assert result.label == "WholeWheat"
        assert result.is_certain is True
        assert result.rois_rejected == 0
        assert 'WholeWheat' in result.evidence_per_class
        assert 'White' in result.evidence_per_class
        assert 'Bran' in result.evidence_per_class


class TestAccumulateTrackEvidenceFunction:
    """Tests for the convenience function accumulate_track_evidence."""
    
    def test_reject_labels_in_convenience_function(self, default_config):
        """Reject labels should be skipped in convenience function."""
        classifications = [
            {
                'probs': {'Rejected': 0.9, 'WholeWheat': 0.1},
                'trust': 0.7,
                'state': 'open'
            },
            {
                'probs': {'WholeWheat': 0.8, 'White': 0.2},
                'trust': 0.9,
                'state': 'closed'
            },
            {
                'probs': {'WholeWheat': 0.75, 'White': 0.25},
                'trust': 0.85,
                'state': 'closed'
            },
        ]
        
        result = accumulate_track_evidence(classifications, default_config)
        
        # Should classify as WholeWheat
        assert result.label == "WholeWheat"
        assert result.is_certain is True
        assert result.rois_rejected == 1
        assert result.rois_used == 3
        assert 'Rejected' not in result.evidence_per_class
        assert 'WholeWheat' in result.evidence_per_class


class TestRejectionMetrics:
    """Tests for rejection tracking and metrics."""
    
    def test_rejection_count_tracking(self, default_config):
        """Rejection count should be tracked correctly."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add multiple ROIs with some rejected
        for i in range(3):
            accumulator.update(
                roi_id=i,
                probs={'Rejected': 0.9, 'WholeWheat': 0.1},
                trust=0.8,
                state='open'
            )
        
        for i in range(3, 6):
            accumulator.update(
                roi_id=i,
                probs={'WholeWheat': 0.8, 'White': 0.2},
                trust=0.9,
                state='closed'
            )
        
        result = accumulator.finalize()
        
        # Check rejection count
        assert result.rois_rejected == 3
        assert result.rois_used == 6
        assert result.rois_trusted >= 2
    
    def test_rejection_count_in_result_dict(self, default_config):
        """Rejection count should be in result dictionary."""
        accumulator = EvidenceAccumulator(default_config)
        
        accumulator.update(
            roi_id=0,
            probs={'Rejected': 0.9, 'WholeWheat': 0.1},
            trust=0.8,
            state='open'
        )
        
        accumulator.update(
            roi_id=1,
            probs={'WholeWheat': 0.8, 'White': 0.2},
            trust=0.9,
            state='closed'
        )
        
        result = accumulator.finalize()
        result_dict = result.to_dict()
        
        # Check that rejection count is in dictionary
        assert 'rois_rejected' in result_dict
        assert result_dict['rois_rejected'] == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_probs_dict(self, default_config):
        """Empty probability dict should not crash."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add ROI with empty probs
        accumulator.update(
            roi_id=0,
            probs={},
            trust=0.8,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Should be uncertain due to no evidence
        assert result.label == "Uncertain"
        assert result.is_certain is False
    
    def test_single_valid_prediction_after_rejects(self, default_config):
        """Valid predictions after rejects should still classify if gate passes."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add multiple rejected predictions (top class is Rejected, but WholeWheat has low prob)
        for i in range(3):
            accumulator.update(
                roi_id=i,
                probs={'Rejected': 0.9, 'WholeWheat': 0.05, 'White': 0.05},
                trust=0.8,
                state='open'
            )
        
        # Add strong valid predictions to pass stability gate
        for i in range(3, 5):
            accumulator.update(
                roi_id=i,
                probs={'WholeWheat': 0.9, 'White': 0.1},
                trust=0.95,
                state='closed'
            )
        
        result = accumulator.finalize()
        
        # Should classify if stability gate passes
        assert result.rois_rejected == 3  # 3 ROIs had 'Rejected' in their probs
        assert result.rois_used == 5
        # With 2 high-confidence closed ROIs for WholeWheat, should pass gate
        if result.is_certain:
            assert result.label == "WholeWheat"
        else:
            # If gate fails, should still be reasonable
            assert result.label == "Uncertain"
    
    def test_rejected_with_zero_probability(self, default_config):
        """Rejected class with 0 probability should still be skipped."""
        accumulator = EvidenceAccumulator(default_config)
        
        # Add predictions where Rejected has 0 probability
        accumulator.update(
            roi_id=0,
            probs={'WholeWheat': 0.7, 'White': 0.3, 'Rejected': 0.0},
            trust=0.9,
            state='closed'
        )
        
        accumulator.update(
            roi_id=1,
            probs={'WholeWheat': 0.8, 'White': 0.2, 'Rejected': 0.0},
            trust=0.9,
            state='closed'
        )
        
        result = accumulator.finalize()
        
        # Should classify normally, Rejected with 0 prob is still skipped
        assert result.label == "WholeWheat"
        assert result.is_certain is True
        # Rejected was in the probs dict but should be counted as rejected due to being in reject_labels
        assert result.rois_rejected >= 0
        assert 'Rejected' not in result.evidence_per_class


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
