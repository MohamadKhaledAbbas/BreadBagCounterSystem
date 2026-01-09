"""
Unit Tests for Enhanced Bidirectional Smoother - Uncertain/Unknown Override.

Tests cover:
1. Uncertain labels with unanimous context → Override
2. Uncertain labels with majority context → Override
3. Uncertain labels with split context → Keep as Uncertain
4. Uncertain labels at batch transitions → Override (not protected)
5. Unknown label override scenarios
6. Context filtering (exclude Uncertain/Unknown from agreement)
7. Confidence tier and uncertain_override flag marking
8. Relaxed threshold validation
9. Statistics tracking
10. Configuration parameters

Run with: python -m unittest src/test/test_bidirectional_smoother_uncertain.py
"""

import unittest
from typing import Dict, Any

from src.classifier.bidirectional_smoother import BidirectionalSmoother


class TestBidirectionalSmootherUncertain(unittest.TestCase):
    """
    Comprehensive test suite for uncertain/unknown label handling in bidirectional smoother.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create smoother with test configuration
        self.smoother = BidirectionalSmoother(
            buffer_size=7,
            confidence_threshold=0.90,
            context_agreement_ratio=0.8,
            uncertain_override_ratio=0.5,
            batch_transition_protection=True,
            enabled=True,
            inactivity_timeout_ms=5000.0
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Flush any remaining events
        self.smoother.flush()
    
    def _create_event(self, event_id: int, bag_type: str, confidence: float) -> Dict[str, Any]:
        """Helper to create test event data."""
        return {
            'event_id': event_id,
            'bag_type': bag_type,
            'confidence': confidence,
        }
    
    def test_uncertain_with_unanimous_context_overrides(self):
        """Test that Uncertain with 100% unanimous context is overridden."""
        # Sequence: Brown, Brown, Brown, Uncertain(0.95), Brown, Brown, Brown
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'Uncertain', 0.95),  # High confidence, but Uncertain
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'Brown_Orange_Overlay', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        # Flush remaining
        results.extend(self.smoother.flush())
        
        # Find the event that was originally Uncertain
        uncertain_result = None
        for result in results:
            if result['event_id'] == 4:
                uncertain_result = result
                break
        
        self.assertIsNotNone(uncertain_result, "Uncertain event should be in results")
        
        # Should be overridden to Brown_Orange_Overlay
        self.assertEqual(uncertain_result['bag_type'], 'Brown_Orange_Overlay')
        self.assertTrue(uncertain_result.get('smoothed', False))
        self.assertEqual(uncertain_result.get('original_bag_type'), 'Uncertain')
        
        # Should be marked as low tier with uncertain_override flag
        self.assertEqual(uncertain_result.get('confidence_tier'), 'low')
        self.assertTrue(uncertain_result.get('uncertain_override', False))
        
        # Check smoothing reason
        reason = uncertain_result.get('smoothing_reason', '')
        self.assertIn('uncertain_override', reason)
        self.assertIn('agreement=1.00', reason)
        self.assertIn('Brown_Orange_Overlay', reason)
        
        # Check statistics
        stats = self.smoother.get_stats()
        self.assertEqual(stats['uncertain_overrides'], 1)
        self.assertEqual(stats['uncertain_kept'], 0)
        self.assertGreater(stats['uncertain_override_rate'], 0.99)  # Should be 1.0
    
    def test_uncertain_with_majority_context_overrides(self):
        """Test that Uncertain with 67% majority context is overridden (above 50% threshold)."""
        # Sequence: Brown, Brown, White, Uncertain(0.60), Brown, White, Brown
        # Context: 4 Brown, 2 White = 67% Brown (above 50% threshold)
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'White', 0.87),
            self._create_event(4, 'Uncertain', 0.60),
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'White', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        uncertain_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(uncertain_result)
        
        # Should be overridden to Brown_Orange_Overlay (67% > 50%)
        self.assertEqual(uncertain_result['bag_type'], 'Brown_Orange_Overlay')
        self.assertTrue(uncertain_result.get('smoothed', False))
        self.assertEqual(uncertain_result.get('confidence_tier'), 'low')
        self.assertTrue(uncertain_result.get('uncertain_override', False))
        
        # Check reason includes agreement ratio
        reason = uncertain_result.get('smoothing_reason', '')
        self.assertIn('uncertain_override', reason)
        self.assertIn('agreement=0.67', reason)
    
    def test_uncertain_with_split_context_kept(self):
        """Test that Uncertain with 50% tie (no majority) is kept as Uncertain."""
        # Sequence: Brown, White, Brown, Uncertain(0.60), White, Brown, White
        # Context: 3 Brown, 3 White = 50% each (no majority)
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'White', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'Uncertain', 0.60),
            self._create_event(5, 'White', 0.86),
            self._create_event(6, 'Brown_Orange_Overlay', 0.89),
            self._create_event(7, 'White', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        uncertain_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(uncertain_result)
        
        # Should remain Uncertain (50% tie, no majority)
        self.assertEqual(uncertain_result['bag_type'], 'Uncertain')
        self.assertFalse(uncertain_result.get('smoothed', False))
        
        # Check statistics - should be kept, not overridden
        stats = self.smoother.get_stats()
        self.assertGreater(stats['uncertain_kept'], 0)
    
    def test_uncertain_at_batch_transition_overrides(self):
        """Test that Uncertain at batch transition is NOT protected (checked for override)."""
        # Sequence: Brown, Brown, Brown, Uncertain(0.60), White, White, White
        # Normally this would be protected as batch transition, but NOT for Uncertain
        # Context: 3 Brown (prev), 3 White (next) = 50-50 split
        # Result: No consensus (50% each), kept as Uncertain
        # The key test is that batch transition protection is SKIPPED (not that it overrides)
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'Uncertain', 0.60),
            self._create_event(5, 'White', 0.86),
            self._create_event(6, 'White', 0.89),
            self._create_event(7, 'White', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        uncertain_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(uncertain_result)
        
        # Should remain Uncertain (50-50 split, no majority)
        # But importantly, batch transition protection was NOT applied
        self.assertEqual(uncertain_result['bag_type'], 'Uncertain')
        self.assertFalse(uncertain_result.get('smoothed', False))
        
        # Verify batch transition protection was NOT applied
        reason = uncertain_result.get('smoothing_reason', '')
        self.assertNotIn('batch_transition_protected', reason)
        self.assertIn('uncertain_no_consensus', reason)
    
    def test_uncertain_at_batch_transition_with_majority_overrides(self):
        """Test that Uncertain at batch transition with >50% majority IS overridden."""
        # Sequence: Brown, Brown, Brown, Uncertain(0.60), Brown, White, White
        # Context: 4 Brown (prev+next), 2 White = 67% Brown
        # Normally protected as batch transition, but Uncertain skips protection
        # Result: Override to Brown (67% > 50%)
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'Uncertain', 0.60),
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'White', 0.89),
            self._create_event(7, 'White', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        uncertain_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(uncertain_result)
        
        # Should be overridden to Brown (67% > 50%)
        self.assertEqual(uncertain_result['bag_type'], 'Brown_Orange_Overlay')
        self.assertTrue(uncertain_result.get('smoothed', False))
        self.assertTrue(uncertain_result.get('uncertain_override', False))
        
        # Verify batch transition protection was NOT applied
        reason = uncertain_result.get('smoothing_reason', '')
        self.assertNotIn('batch_transition_protected', reason)
        self.assertIn('uncertain_override', reason)
    
    def test_unknown_label_override(self):
        """Test that Unknown labels are handled same as Uncertain."""
        # Sequence: Brown, Brown, Brown, Unknown(0.80), Brown, Brown, Brown
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'Unknown', 0.80),
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'Brown_Orange_Overlay', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        unknown_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(unknown_result)
        
        # Should be overridden to Brown_Orange_Overlay
        self.assertEqual(unknown_result['bag_type'], 'Brown_Orange_Overlay')
        self.assertTrue(unknown_result.get('smoothed', False))
        self.assertEqual(unknown_result.get('original_bag_type'), 'Unknown')
        self.assertEqual(unknown_result.get('confidence_tier'), 'low')
        self.assertTrue(unknown_result.get('uncertain_override', False))
    
    def test_context_filtering_excludes_uncertain(self):
        """Test that Uncertain/Unknown labels are filtered from context agreement calculation."""
        # Sequence: Brown, Brown, Uncertain, Uncertain(0.60), Brown, Unknown, Brown
        # Without filtering: 4 Brown, 2 Uncertain, 1 Unknown = 4/7 = 57%
        # With filtering: 4 Brown (100% after filtering out Uncertain/Unknown)
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Uncertain', 0.87),
            self._create_event(4, 'Uncertain', 0.60),
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'Unknown', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        uncertain_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(uncertain_result)
        
        # Should be overridden to Brown (100% after filtering)
        self.assertEqual(uncertain_result['bag_type'], 'Brown_Orange_Overlay')
        
        # Agreement should be 1.00 (all 4 valid context items are Brown)
        reason = uncertain_result.get('smoothing_reason', '')
        self.assertIn('agreement=1.00', reason)
    
    def test_regular_label_with_high_confidence_bypasses(self):
        """Test that regular (non-Uncertain) labels still bypass with high confidence."""
        # High confidence regular label should bypass, Uncertain should not
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'White', 0.95),  # High confidence, regular label
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'Brown_Orange_Overlay', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        white_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(white_result)
        
        # Should remain White (high confidence bypass)
        self.assertEqual(white_result['bag_type'], 'White')
        self.assertFalse(white_result.get('smoothed', False))
        
        # Check statistics - should be high confidence bypassed
        stats = self.smoother.get_stats()
        self.assertGreater(stats['high_confidence_bypassed'], 0)
    
    def test_uncertain_high_confidence_does_not_bypass(self):
        """Test that Uncertain labels do NOT bypass even with high confidence."""
        # Uncertain with 0.95 confidence should still check context
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'Uncertain', 0.95),  # High confidence Uncertain
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'Brown_Orange_Overlay', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        uncertain_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(uncertain_result)
        
        # Should be overridden to Brown (NOT bypassed despite high confidence)
        self.assertEqual(uncertain_result['bag_type'], 'Brown_Orange_Overlay')
        self.assertTrue(uncertain_result.get('smoothed', False))
        self.assertTrue(uncertain_result.get('uncertain_override', False))
    
    def test_statistics_tracking(self):
        """Test that statistics correctly track uncertain overrides and keeps."""
        # Reset stats
        self.smoother.reset_stats()
        
        # Create sequence with multiple uncertain scenarios
        events = [
            # Unanimous override scenario
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'Uncertain', 0.60),  # Will be overridden
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'Brown_Orange_Overlay', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
            # Split context scenario
            self._create_event(8, 'Brown_Orange_Overlay', 0.85),
            self._create_event(9, 'White', 0.88),
            self._create_event(10, 'Brown_Orange_Overlay', 0.87),
            self._create_event(11, 'Uncertain', 0.60),  # Will be kept
            self._create_event(12, 'White', 0.86),
            self._create_event(13, 'Brown_Orange_Overlay', 0.89),
            self._create_event(14, 'White', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        stats = self.smoother.get_stats()
        
        # Should have at least 1 override and 1 kept
        self.assertGreater(stats['uncertain_overrides'], 0)
        self.assertGreater(stats['uncertain_kept'], 0)
        
        # Calculate override rate
        total = stats['uncertain_overrides'] + stats['uncertain_kept']
        expected_rate = stats['uncertain_overrides'] / total
        self.assertAlmostEqual(stats['uncertain_override_rate'], expected_rate, places=2)
    
    def test_configuration_parameters(self):
        """Test that configuration parameters are loaded correctly."""
        # Test with custom parameters
        custom_smoother = BidirectionalSmoother(
            buffer_size=9,
            confidence_threshold=0.85,
            context_agreement_ratio=0.75,
            uncertain_override_ratio=0.6,
            batch_transition_protection=False,
            enabled=True
        )
        
        self.assertEqual(custom_smoother.buffer_size, 9)
        self.assertEqual(custom_smoother.confidence_threshold, 0.85)
        self.assertEqual(custom_smoother.context_agreement_ratio, 0.75)
        self.assertEqual(custom_smoother.uncertain_override_ratio, 0.6)
        self.assertFalse(custom_smoother.batch_transition_protection)
        self.assertTrue(custom_smoother.enabled)
    
    def test_backward_compatibility(self):
        """Test that non-Uncertain labels still work as before."""
        # Regular low-confidence label with unanimous context
        events = [
            self._create_event(1, 'Brown_Orange_Overlay', 0.85),
            self._create_event(2, 'Brown_Orange_Overlay', 0.88),
            self._create_event(3, 'Brown_Orange_Overlay', 0.87),
            self._create_event(4, 'White', 0.60),  # Low conf, wrong label
            self._create_event(5, 'Brown_Orange_Overlay', 0.86),
            self._create_event(6, 'Brown_Orange_Overlay', 0.89),
            self._create_event(7, 'Brown_Orange_Overlay', 0.87),
        ]
        
        results = []
        for event in events:
            result = self.smoother.add_event(event)
            if result:
                results.append(result)
        
        results.extend(self.smoother.flush())
        
        white_result = next((r for r in results if r['event_id'] == 4), None)
        self.assertIsNotNone(white_result)
        
        # Should be overridden to Brown (100% > 80% threshold)
        self.assertEqual(white_result['bag_type'], 'Brown_Orange_Overlay')
        self.assertTrue(white_result.get('smoothed', False))
        
        # Should NOT have uncertain_override flag
        self.assertNotIn('uncertain_override', white_result)
        
        # Reason should be regular context_override
        reason = white_result.get('smoothing_reason', '')
        self.assertIn('context_override', reason)
        self.assertNotIn('uncertain_override', reason)


if __name__ == '__main__':
    unittest.main()
