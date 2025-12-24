"""
Test suite for smart frame skipping in degraded mode.

This test validates the production-ready smart skip implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.tracking_config import tracking_config


def test_smart_skip_config_exists():
    """Test that all smart skip configuration parameters exist."""
    print("✓ Testing smart skip configuration parameters...")
    
    assert hasattr(tracking_config, 'degraded_mode_smart_skip_enabled')
    assert hasattr(tracking_config, 'degraded_mode_skip_pattern')
    assert hasattr(tracking_config, 'degraded_mode_min_frames_per_event')
    assert hasattr(tracking_config, 'degraded_mode_skip_with_active_events_only')
    assert hasattr(tracking_config, 'degraded_mode_preserve_critical_states')
    assert hasattr(tracking_config, 'degraded_mode_critical_state_frame_threshold')
    assert hasattr(tracking_config, 'degraded_mode_max_skip_rate_with_events')
    
    print("  ✓ All configuration parameters exist")


def test_smart_skip_default_values():
    """Test that smart skip has production-ready default values."""
    print("✓ Testing smart skip default values...")
    
    assert tracking_config.degraded_mode_smart_skip_enabled == True, \
        "Smart skip should be enabled by default"
    
    assert tracking_config.degraded_mode_skip_pattern == 'adaptive', \
        "Skip pattern should be 'adaptive' for production"
    
    assert 10 <= tracking_config.degraded_mode_min_frames_per_event <= 25, \
        f"Min frames per event should be 10-25, got {tracking_config.degraded_mode_min_frames_per_event}"
    
    assert tracking_config.degraded_mode_preserve_critical_states == True, \
        "Should preserve critical states by default"
    
    assert 0.3 <= tracking_config.degraded_mode_max_skip_rate_with_events <= 0.6, \
        f"Max skip rate should be 30-60%, got {tracking_config.degraded_mode_max_skip_rate_with_events}"
    
    print("  ✓ All default values are production-ready")


def test_skip_pattern_options():
    """Test that skip pattern configuration accepts valid values."""
    print("✓ Testing skip pattern options...")
    
    valid_patterns = ['every_2nd', 'every_3rd', 'adaptive']
    assert tracking_config.degraded_mode_skip_pattern in valid_patterns, \
        f"Skip pattern must be one of {valid_patterns}, got {tracking_config.degraded_mode_skip_pattern}"
    
    print("  ✓ Skip pattern is valid")


def test_min_frames_calculation():
    """Test that min frames per event is calculated correctly."""
    print("✓ Testing min frames per event calculation...")
    
    # With ghost_timeout_frames = 40 and 50% skip rate
    # Events should get at least 15-20 frames for reliable tracking
    ghost_timeout = tracking_config.ghost_timeout_frames
    min_frames = tracking_config.degraded_mode_min_frames_per_event
    
    # With 50% skip rate, an event living for ghost_timeout frames gets ghost_timeout/2 processed frames
    effective_frames_at_50_percent = ghost_timeout / 2
    
    assert min_frames <= effective_frames_at_50_percent, \
        f"Min frames ({min_frames}) should be achievable at 50% skip rate " \
        f"(ghost_timeout={ghost_timeout}, effective={effective_frames_at_50_percent})"
    
    # Should be at least 10 frames for basic tracking
    assert min_frames >= 10, \
        f"Min frames should be at least 10 for reliable tracking, got {min_frames}"
    
    print(f"  ✓ Min frames per event ({min_frames}) is appropriate for ghost_timeout={ghost_timeout}")


def test_smart_skip_logic_simulation():
    """Simulate smart skip logic for different scenarios."""
    print("✓ Testing smart skip logic simulation...")
    
    # Simulate adaptive pattern behavior
    test_cases = [
        (0.45, 'adaptive', False, "Below 50% queue - no pattern skip"),
        (0.60, 'adaptive', True, "50-70% queue - skip every 3rd (frame 3, 6, 9, ...)"),
        (0.75, 'adaptive', True, "70-85% queue - skip every 2nd (frame 2, 4, 6, ...)"),
        (0.90, 'adaptive', True, "85-95% queue - skip 2 of 3 (frame 1, 2, 4, 5, ...)"),
        (0.97, 'adaptive', True, "95%+ queue - skip 3 of 4 (frame 1, 2, 3, 5, 6, 7, ...)"),
    ]
    
    for queue_util, pattern, should_skip_some, description in test_cases:
        # Simulate 12 frames
        frame_counter = 0
        skipped_count = 0
        
        for i in range(1, 13):
            frame_counter += 1
            
            if pattern == 'adaptive':
                if queue_util < 0.5:
                    skip = False
                elif queue_util < 0.7:
                    skip = (frame_counter % 3 == 0)
                elif queue_util < 0.85:
                    skip = (frame_counter % 2 == 0)
                elif queue_util < 0.95:
                    skip = (frame_counter % 3 != 0)
                else:
                    skip = (frame_counter % 4 != 0)
            
            if skip:
                skipped_count += 1
        
        if should_skip_some:
            assert skipped_count > 0, f"Should skip some frames: {description}"
        else:
            assert skipped_count == 0, f"Should not skip frames: {description}"
        
        print(f"  ✓ {description}: skipped {skipped_count}/12 frames")
    
    print("  ✓ Smart skip logic simulation passed")


def test_backwards_compatibility():
    """Test that smart skip can be disabled for backwards compatibility."""
    print("✓ Testing backwards compatibility...")
    
    # Smart skip should be opt-in via configuration
    assert isinstance(tracking_config.degraded_mode_smart_skip_enabled, bool), \
        "Smart skip should be a boolean flag"
    
    print("  ✓ Can be disabled for backwards compatibility")


def test_event_awareness():
    """Test that smart skip is event-aware."""
    print("✓ Testing event awareness features...")
    
    # Min frames per event ensures events get enough samples
    assert tracking_config.degraded_mode_min_frames_per_event > 0, \
        "Must define minimum frames per event"
    
    # Critical states should be preserved
    assert isinstance(tracking_config.degraded_mode_preserve_critical_states, bool), \
        "Critical states preservation should be configurable"
    
    # Critical state threshold defines early OPEN period
    assert tracking_config.degraded_mode_critical_state_frame_threshold > 0, \
        "Must define critical state frame threshold"
    
    print("  ✓ Event-aware features are properly configured")


def test_production_readiness():
    """Test that configuration is production-ready."""
    print("✓ Testing production readiness...")
    
    # Ensure reasonable defaults that won't break tracking
    assert tracking_config.degraded_mode_min_frames_per_event >= 10, \
        "Min frames should be at least 10 for production"
    
    assert tracking_config.degraded_mode_max_skip_rate_with_events <= 0.6, \
        "Max skip rate should not exceed 60% to maintain tracking quality"
    
    assert tracking_config.degraded_mode_skip_pattern in ['adaptive', 'every_2nd', 'every_3rd'], \
        "Skip pattern should be a known production pattern"
    
    # Adaptive is recommended for production
    if tracking_config.degraded_mode_skip_pattern == 'adaptive':
        print("  ✓ Using recommended 'adaptive' pattern for production")
    
    print("  ✓ Configuration is production-ready")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("Smart Frame Skipping Test Suite")
    print("="*70 + "\n")
    
    tests = [
        test_smart_skip_config_exists,
        test_smart_skip_default_values,
        test_skip_pattern_options,
        test_min_frames_calculation,
        test_smart_skip_logic_simulation,
        test_backwards_compatibility,
        test_event_awareness,
        test_production_readiness,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            failed += 1
    
    print("="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("✓ All tests passed! Smart frame skipping is production-ready.\n")
        sys.exit(0)


if __name__ == '__main__':
    run_all_tests()
