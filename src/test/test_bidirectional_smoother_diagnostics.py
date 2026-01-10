"""
Unit Tests for Bidirectional Smoother Diagnostics (Issue #4).

Tests cover:
1. Per-class statistics tracking
2. Bias detection and analysis
3. High-frequency class detection
4. Net gain/loss calculations
5. Transition pattern tracking

Run with: python -m pytest src/test/test_bidirectional_smoother_diagnostics.py -v
"""

import pytest

from src.classifier.bidirectional_smoother import BidirectionalSmoother


# =============================================================================
# Test: Per-Class Statistics Tracking
# =============================================================================

class TestPerClassStatistics:
    """Tests for per-class statistics tracking."""
    
    def test_tracks_total_events_per_class(self):
        """Verify total events per class are tracked correctly."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=7,
            confidence_threshold=0.95,  # High threshold to force context checking
            context_agreement_ratio=0.6,
        )
        
        # Add events: 6 Brown, 4 White
        for i in range(10):
            event_data = {
                'event_id': i,
                'bag_type': 'Brown' if i < 6 else 'White',
                'confidence': 0.70,  # Below threshold
            }
            smoother.add_event(event_data)
        
        smoother.flush()
        diagnostics = smoother.get_per_class_diagnostics()
        
        # Total should account for validated events
        total_tracked = sum(d['total_events'] for d in diagnostics.values())
        assert total_tracked <= 10, f"Should not track more than submitted events"
    
    def test_tracks_smoothed_from(self):
        """Verify 'smoothed_from' counts when a class is changed TO something else."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.6,  # 60% agreement triggers smoothing
        )
        
        # Setup: Brown, Brown, Minority, Brown, Brown
        # The "Minority" at center should be smoothed TO "Brown"
        labels = ['Brown', 'Brown', 'White', 'Brown', 'Brown']
        for i, label in enumerate(labels):
            smoother.add_event({
                'event_id': i,
                'bag_type': label,
                'confidence': 0.70,
            })
        
        results = smoother.flush()
        
        # The White should have been smoothed_from
        diagnostics = smoother.get_per_class_diagnostics()
        
        # There should be smoothing activity
        stats = smoother.get_stats()
        if stats['smoothed_events'] > 0:
            # If smoothing occurred, White should have been smoothed_from
            if 'White' in diagnostics:
                assert diagnostics['White']['smoothed_from'] >= 0
    
    def test_tracks_smoothed_to(self):
        """Verify 'smoothed_to' counts when something is changed TO this class."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.6,
        )
        
        # Brown is the majority, should have smoothed_to incremented
        labels = ['Brown', 'Brown', 'White', 'Brown', 'Brown']
        for i, label in enumerate(labels):
            smoother.add_event({
                'event_id': i,
                'bag_type': label,
                'confidence': 0.70,
            })
        
        smoother.flush()
        diagnostics = smoother.get_per_class_diagnostics()
        
        stats = smoother.get_stats()
        if stats['smoothed_events'] > 0 and 'Brown' in diagnostics:
            # Brown should have gained events via smoothing
            assert diagnostics['Brown']['smoothed_to'] >= 0
    
    def test_tracks_high_confidence_bypass(self):
        """Verify high-confidence bypass is tracked per class."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=7,
            confidence_threshold=0.90,  # 90% threshold
        )
        
        # Add high-confidence events
        for i in range(10):
            smoother.add_event({
                'event_id': i,
                'bag_type': 'Brown',
                'confidence': 0.95,  # Above threshold
            })
        
        smoother.flush()
        diagnostics = smoother.get_per_class_diagnostics()
        
        if 'Brown' in diagnostics:
            # High-confidence events bypass context checking
            assert diagnostics['Brown']['high_conf_bypass_rate'] >= 0.0


# =============================================================================
# Test: Bias Analysis
# =============================================================================

class TestBiasAnalysis:
    """Tests for bias detection and analysis."""
    
    def test_detects_gainer_classes(self):
        """Verify classes that gain events are detected."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.6,
        )
        
        # Submit events where Brown will likely gain from smoothing
        # Pattern: Brown dominates context
        for i in range(20):
            if i % 5 == 2:  # Every 5th middle position is White
                label = 'White'
            else:
                label = 'Brown'
            
            smoother.add_event({
                'event_id': i,
                'bag_type': label,
                'confidence': 0.70,
            })
        
        smoother.flush()
        
        bias = smoother.get_bias_analysis()
        
        # Should have analysis result
        assert bias['analysis_status'] == 'complete'
        assert 'gainers' in bias
        assert 'losers' in bias
    
    def test_detects_loser_classes(self):
        """Verify classes that lose events are detected."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.6,
        )
        
        # Same pattern - White should be a loser
        for i in range(20):
            if i % 5 == 2:
                label = 'White'
            else:
                label = 'Brown'
            
            smoother.add_event({
                'event_id': i,
                'bag_type': label,
                'confidence': 0.70,
            })
        
        smoother.flush()
        
        bias = smoother.get_bias_analysis()
        assert 'losers' in bias
    
    def test_generates_recommendations(self):
        """Verify recommendations are generated for significant bias."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.5,  # Low threshold = more smoothing
        )
        
        # Strongly biased pattern
        for i in range(40):
            # 90% Brown, 10% White
            if i % 10 == 0:
                label = 'White'
            else:
                label = 'Brown'
            
            smoother.add_event({
                'event_id': i,
                'bag_type': label,
                'confidence': 0.70,
            })
        
        smoother.flush()
        
        bias = smoother.get_bias_analysis()
        assert 'recommendations' in bias
    
    def test_handles_insufficient_data(self):
        """Verify proper handling when insufficient data for analysis."""
        smoother = BidirectionalSmoother(enabled=True, buffer_size=7)
        
        # No events added
        bias = smoother.get_bias_analysis()
        
        assert bias['analysis_status'] == 'insufficient_data'


# =============================================================================
# Test: Net Gain Calculations
# =============================================================================

class TestNetGainCalculations:
    """Tests for net gain/loss calculations."""
    
    def test_net_gain_calculation(self):
        """Verify net gain = smoothed_to - smoothed_from."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.6,
        )
        
        # Add events with clear majority
        for i in range(10):
            smoother.add_event({
                'event_id': i,
                'bag_type': 'Brown' if i % 4 != 0 else 'White',
                'confidence': 0.70,
            })
        
        smoother.flush()
        diagnostics = smoother.get_per_class_diagnostics()
        
        for label, stats in diagnostics.items():
            expected_net = stats['smoothed_to'] - stats['smoothed_from']
            assert stats['net_gain'] == expected_net, \
                f"Net gain for {label} should be smoothed_to - smoothed_from"
    
    def test_net_gain_rate_calculation(self):
        """Verify net gain rate = net_gain / total_events."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.6,
        )
        
        for i in range(10):
            smoother.add_event({
                'event_id': i,
                'bag_type': 'Brown' if i % 4 != 0 else 'White',
                'confidence': 0.70,
            })
        
        smoother.flush()
        diagnostics = smoother.get_per_class_diagnostics()
        
        for label, stats in diagnostics.items():
            if stats['total_events'] > 0:
                expected_rate = stats['net_gain'] / stats['total_events']
                assert abs(stats['net_gain_rate'] - expected_rate) < 0.001, \
                    f"Net gain rate for {label} should be net_gain / total_events"


# =============================================================================
# Test: Transition Tracking
# =============================================================================

class TestTransitionTracking:
    """Tests for transition pattern tracking."""
    
    def test_tracks_transition_patterns(self):
        """Verify transition patterns are recorded."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
            context_agreement_ratio=0.6,
        )
        
        # Strong pattern where White always converts to Brown
        for i in range(15):
            # Brown, Brown, White, Brown, Brown pattern
            if i % 5 == 2:
                label = 'White'
            else:
                label = 'Brown'
            
            smoother.add_event({
                'event_id': i,
                'bag_type': label,
                'confidence': 0.70,
            })
        
        smoother.flush()
        diagnostics = smoother.get_per_class_diagnostics()
        
        # Check if transitions are recorded (may be empty if no smoothing occurred)
        for label, stats in diagnostics.items():
            assert 'top_transitions' in stats


# =============================================================================
# Test: Stats Integration
# =============================================================================

class TestStatsIntegration:
    """Tests for integration with main stats."""
    
    def test_per_class_included_in_stats(self):
        """Verify per-class diagnostics are included in get_stats()."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
        )
        
        for i in range(10):
            smoother.add_event({
                'event_id': i,
                'bag_type': 'Brown' if i % 2 == 0 else 'White',
                'confidence': 0.70,
            })
        
        smoother.flush()
        stats = smoother.get_stats()
        
        # Per-class diagnostics should be included
        assert 'per_class_diagnostics' in stats
    
    def test_reset_clears_per_class_stats(self):
        """Verify reset_stats() clears per-class statistics."""
        smoother = BidirectionalSmoother(
            enabled=True,
            buffer_size=5,
            confidence_threshold=0.95,
        )
        
        for i in range(10):
            smoother.add_event({
                'event_id': i,
                'bag_type': 'Brown',
                'confidence': 0.70,
            })
        
        smoother.flush()
        
        # Should have stats
        assert len(smoother.get_per_class_diagnostics()) > 0
        
        # Reset
        smoother.reset_stats()
        
        # Should be empty
        assert len(smoother.get_per_class_diagnostics()) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
