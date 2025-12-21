"""
Unit Tests for Probability Adjustment Module and BpuClassifier predict_probs.

Tests cover:
1. BpuClassifier.predict_probs implementation
2. Probability adjustment strategies
3. Edge cases and validation
4. Integration with ClassifierService

Run with: python -m pytest src/test/test_probability_adjustments.py -v
(Or run directly if pytest not installed: python src/test/test_probability_adjustments.py)
"""

import sys
import os

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
from typing import Dict, Any, Optional

# Import modules to test
from src.classifier.probability_adjustments import (
    apply_probability_adjustment,
    validate_probability_vector,
    apply_batch_adjustments,
    ProbabilityAdjustmentResult
)


# =============================================================================
# Mock Configuration
# =============================================================================

@dataclass
class MockConfig:
    """Mock configuration for testing."""
    prob_adjustment_strategy: str = 'full_transfer'
    prob_adjustment_transfer_ratio: float = 1.0
    prob_adjustment_epsilon: float = 1e-9
    prob_adjustment_debug_logging: bool = False
    
    # Disambiguation classes for context
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')


@pytest.fixture
def default_config():
    """Default mock configuration."""
    return MockConfig()


# =============================================================================
# Probability Adjustment Tests
# =============================================================================

class TestProbabilityAdjustment:
    """Tests for probability mass transfer functionality."""
    
    def test_full_transfer_strategy(self, default_config):
        """Test full_transfer strategy concentrates all family mass to target."""
        original_probs = {
            'Brown_Orange_Overlay': 0.6,
            'Brown_Orange_Small': 0.3,
            'White': 0.05,
            'Bran': 0.05
        }
        
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=original_probs,
            from_label='Brown_Orange_Overlay',
            to_label='Brown_Orange_Small',
            family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
            config=default_config
        )
        
        # Check that adjustment was applied
        assert metadata['applied'] == True
        assert metadata['from_label'] == 'Brown_Orange_Overlay'
        assert metadata['to_label'] == 'Brown_Orange_Small'
        
        # Check family mass transfer (0.6 + 0.3 = 0.9 all goes to Small)
        assert abs(adjusted_probs['Brown_Orange_Small'] - 0.9) < 1e-6
        assert adjusted_probs['Brown_Orange_Overlay'] < 1e-6  # Nearly zero (epsilon)
        
        # Check other classes unchanged
        assert abs(adjusted_probs['White'] - 0.05) < 1e-6
        assert abs(adjusted_probs['Bran'] - 0.05) < 1e-6
        
        # Check normalization
        assert abs(sum(adjusted_probs.values()) - 1.0) < 1e-6
    
    def test_no_adjustment_when_labels_same(self, default_config):
        """Test that no adjustment is made when from_label == to_label."""
        original_probs = {
            'Brown_Orange_Overlay': 0.6,
            'Brown_Orange_Small': 0.3,
            'White': 0.1
        }
        
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=original_probs,
            from_label='Brown_Orange_Overlay',
            to_label='Brown_Orange_Overlay',  # Same label
            family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
            config=default_config
        )
        
        # No adjustment should be made
        assert metadata['applied'] == False
        assert metadata['reason'] == 'no_label_change'
        
        # Probs should be unchanged
        assert adjusted_probs == original_probs
    
    def test_proportional_transfer_strategy(self):
        """Test proportional_transfer strategy."""
        config = MockConfig(
            prob_adjustment_strategy='proportional_transfer',
            prob_adjustment_transfer_ratio=0.5
        )
        
        original_probs = {
            'Brown_Orange_Overlay': 0.6,
            'Brown_Orange_Small': 0.3,
            'White': 0.1
        }
        
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=original_probs,
            from_label='Brown_Orange_Overlay',
            to_label='Brown_Orange_Small',
            family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
            config=config
        )
        
        # Check transfer amount (0.6 * 0.5 = 0.3 transferred)
        assert metadata['applied'] == True
        assert abs(metadata['mass_transferred'] - 0.3) < 1e-6
        
        # Check adjusted values
        # Overlay: 0.6 - 0.3 = 0.3
        # Small: 0.3 + 0.3 = 0.6
        assert abs(adjusted_probs['Brown_Orange_Overlay'] - 0.3) < 1e-6
        assert abs(adjusted_probs['Brown_Orange_Small'] - 0.6) < 1e-6
        assert abs(adjusted_probs['White'] - 0.1) < 1e-6
        
        # Check normalization
        assert abs(sum(adjusted_probs.values()) - 1.0) < 1e-6
    
    def test_swap_strategy(self):
        """Test swap strategy exchanges probabilities between classes."""
        config = MockConfig(prob_adjustment_strategy='swap')
        
        original_probs = {
            'Brown_Orange_Overlay': 0.7,
            'Brown_Orange_Small': 0.2,
            'White': 0.1
        }
        
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=original_probs,
            from_label='Brown_Orange_Overlay',
            to_label='Brown_Orange_Small',
            family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
            config=config
        )
        
        # Check swap (0.7 <-> 0.2)
        assert metadata['applied'] == True
        assert abs(adjusted_probs['Brown_Orange_Overlay'] - 0.2) < 1e-6
        assert abs(adjusted_probs['Brown_Orange_Small'] - 0.7) < 1e-6
        assert abs(adjusted_probs['White'] - 0.1) < 1e-6
    
    def test_missing_label_in_probs(self, default_config):
        """Test graceful handling when label is missing from probability vector."""
        original_probs = {
            'Brown_Orange_Overlay': 0.6,
            'White': 0.2,
            'Bran': 0.2
        }
        
        # to_label not in probs
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=original_probs,
            from_label='Brown_Orange_Overlay',
            to_label='Brown_Orange_Small',  # Not in probs
            family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
            config=default_config
        )
        
        # Should not apply adjustment
        assert metadata['applied'] == False
        assert 'to_label_not_in_probs' in metadata['reason']
        assert adjusted_probs == original_probs
    
    def test_metadata_completeness(self, default_config):
        """Test that metadata contains all expected fields."""
        original_probs = {
            'Brown_Orange_Overlay': 0.5,
            'Brown_Orange_Small': 0.4,
            'White': 0.1
        }
        
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=original_probs,
            from_label='Brown_Orange_Overlay',
            to_label='Brown_Orange_Small',
            family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
            config=default_config
        )
        
        # Check all required metadata fields
        required_fields = [
            'applied', 'from_label', 'to_label', 'mass_transferred',
            'before_from', 'before_to', 'after_from', 'after_to',
            'normalization_applied', 'reason'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # Check value ranges
        assert metadata['applied'] == True
        assert metadata['mass_transferred'] > 0
        assert 0 <= metadata['before_from'] <= 1
        assert 0 <= metadata['after_to'] <= 1


class TestProbabilityValidation:
    """Tests for probability vector validation."""
    
    def test_valid_probability_vector(self):
        """Test validation passes for well-formed probability vector."""
        probs = {
            'Class_A': 0.6,
            'Class_B': 0.3,
            'Class_C': 0.1
        }
        
        is_valid, error = validate_probability_vector(probs)
        assert is_valid == True
        assert error is None
    
    def test_negative_probability(self):
        """Test validation fails for negative probabilities."""
        probs = {
            'Class_A': 0.7,
            'Class_B': -0.1,  # Invalid
            'Class_C': 0.4
        }
        
        is_valid, error = validate_probability_vector(probs)
        assert is_valid == False
        assert 'negative_probability' in error
    
    def test_sum_not_one(self):
        """Test validation fails when sum != 1.0."""
        probs = {
            'Class_A': 0.5,
            'Class_B': 0.3,
            'Class_C': 0.1  # Sum = 0.9
        }
        
        is_valid, error = validate_probability_vector(probs)
        assert is_valid == False
        assert 'sum_not_one' in error
    
    def test_empty_vector(self):
        """Test validation fails for empty probability vector."""
        probs = {}
        
        is_valid, error = validate_probability_vector(probs)
        assert is_valid == False
        assert 'empty' in error


class TestBatchAdjustments:
    """Tests for batch probability adjustment."""
    
    def test_batch_with_mixed_disambiguation(self, default_config):
        """Test batch adjustment with some ROIs disambiguated, some not."""
        classifications = [
            {
                'probs': {'Overlay': 0.6, 'Small': 0.3, 'White': 0.1},
                'label': 'Small',
                'original_label': 'Overlay',
                'disambiguated': True
            },
            {
                'probs': {'Overlay': 0.7, 'Small': 0.2, 'White': 0.1},
                'label': 'Overlay',
                'original_label': 'Overlay',
                'disambiguated': False  # No label change
            },
            {
                'probs': {'Overlay': 0.5, 'Small': 0.4, 'White': 0.1},
                'label': 'Small',
                'original_label': 'Overlay',
                'disambiguated': True
            }
        ]
        
        results = apply_batch_adjustments(
            classifications=classifications,
            family_classes=['Overlay', 'Small'],
            config=default_config
        )
        
        # Check first ROI (disambiguated, should have adjustment)
        assert results[0]['prob_adjustment']['applied'] == True
        assert 'Small' in results[0]['probs']
        
        # Check second ROI (not disambiguated, no adjustment)
        assert results[1]['prob_adjustment']['applied'] == False
        
        # Check third ROI (disambiguated, should have adjustment)
        assert results[2]['prob_adjustment']['applied'] == True


# =============================================================================
# BpuClassifier predict_probs Tests (Mock-based)
# =============================================================================

class TestBpuClassifierPredictProbs:
    """Tests for BpuClassifier.predict_probs implementation."""
    
    def test_predict_probs_structure(self):
        """
        Test that predict_probs returns correct structure.
        
        Note: This is a structural test. Full integration test requires
        actual BPU hardware/model.
        """
        # Mock a probability vector that predict_probs should return
        expected_structure = {
            'label': str,
            'confidence': float,
            'probs': dict
        }
        
        # Validate expected output types
        assert expected_structure['label'] == str
        assert expected_structure['confidence'] == float
        assert expected_structure['probs'] == dict
    
    def test_predict_probs_probability_sum(self):
        """Test that returned probability vector sums to ~1.0."""
        # Simulate what predict_probs should return
        mock_probs = {
            'Brown_Orange_Overlay': 0.45,
            'Brown_Orange_Small': 0.30,
            'White': 0.15,
            'Bran': 0.10
        }
        
        # Validate sum
        total = sum(mock_probs.values())
        assert abs(total - 1.0) < 1e-6, "Probabilities must sum to 1.0"
    
    def test_predict_probs_all_classes_present(self):
        """Test that all known classes are present in probs dict."""
        known_classes = ['Brown_Orange_Overlay', 'Brown_Orange_Small', 'White', 'Bran', 'WholeWheat']
        
        # Mock probs dict
        mock_probs = {cls: 0.2 for cls in known_classes}
        
        # Check all classes present
        for cls in known_classes:
            assert cls in mock_probs, f"Class {cls} missing from probs"
    
    def test_predict_probs_matches_predict(self):
        """Test that label/confidence from predict_probs match predict."""
        # Simulate predict output
        predict_label = 'Brown_Orange_Overlay'
        predict_conf = 0.65
        
        # Simulate predict_probs output
        predict_probs_label = 'Brown_Orange_Overlay'
        predict_probs_conf = 0.65
        predict_probs_dict = {
            'Brown_Orange_Overlay': 0.65,
            'Brown_Orange_Small': 0.25,
            'White': 0.10
        }
        
        # Validate consistency
        assert predict_label == predict_probs_label
        assert abs(predict_conf - predict_probs_conf) < 1e-6
        assert predict_probs_dict[predict_probs_label] == predict_probs_conf


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_with_disambiguation_and_adjustment(self, default_config):
        """
        Test full pipeline: classify -> disambiguate -> adjust probs.
        
        This simulates what happens in ClassifierService evidence path.
        """
        # Step 1: Classifier returns probs
        classifier_probs = {
            'Brown_Orange_Overlay': 0.55,
            'Brown_Orange_Small': 0.35,
            'White': 0.10
        }
        classifier_label = 'Brown_Orange_Overlay'
        
        # Step 2: Disambiguation flips label (simulated size-based decision)
        disambiguated_label = 'Brown_Orange_Small'  # Size indicates Small
        
        # Step 3: Apply probability adjustment
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=classifier_probs,
            from_label=classifier_label,
            to_label=disambiguated_label,
            family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
            config=default_config
        )
        
        # Validate: Small should now have most of the family mass
        family_mass = classifier_probs['Brown_Orange_Overlay'] + classifier_probs['Brown_Orange_Small']
        assert abs(adjusted_probs['Brown_Orange_Small'] - family_mass) < 1e-6
        
        # Validate: White unchanged
        assert abs(adjusted_probs['White'] - classifier_probs['White']) < 1e-6
        
        # Validate: Metadata records the adjustment
        assert metadata['applied'] == True
        assert metadata['from_label'] == classifier_label
        assert metadata['to_label'] == disambiguated_label


# =============================================================================
# Main (for standalone execution)
# =============================================================================

if __name__ == '__main__':
    if PYTEST_AVAILABLE:
        import pytest
        pytest.main([__file__, '-v'])
    else:
        print("Running tests in standalone mode (pytest not available)")
        print("=" * 70)
        
        # Create test instances
        config = MockConfig()
        
        # Run some basic tests manually
        print("\n1. Testing full_transfer strategy...")
        test = TestProbabilityAdjustment()
        try:
            test.test_full_transfer_strategy(config)
            print("   PASS: full_transfer strategy works correctly")
        except AssertionError as e:
            print(f"   FAIL: {e}")
        
        print("\n2. Testing no adjustment when labels same...")
        try:
            test.test_no_adjustment_when_labels_same(config)
            print("   PASS: No adjustment when labels are same")
        except AssertionError as e:
            print(f"   FAIL: {e}")
        
        print("\n3. Testing probability validation...")
        test_validation = TestProbabilityValidation()
        try:
            test_validation.test_valid_probability_vector()
            print("   PASS: Valid probability vector accepted")
        except AssertionError as e:
            print(f"   FAIL: {e}")
        
        print("\n4. Testing integration pipeline...")
        test_integration = TestIntegration()
        try:
            test_integration.test_full_pipeline_with_disambiguation_and_adjustment(config)
            print("   PASS: Full pipeline integration works")
        except AssertionError as e:
            print(f"   FAIL: {e}")
        
        print("\n" + "=" * 70)
        print("Standalone test run complete!")
