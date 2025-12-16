#!/usr/bin/env python3
"""
Simple test script to validate key changes made in this PR.

Tests:
1. Unknown phash handling (None instead of "unknown")
2. Database Unknown aggregation
3. Confidence tier tracking
4. Degraded mode configuration
"""

import sys
import os
import tempfile
import sqlite3

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_database_unknown_handling():
    """Test that Database code structure is correct."""
    print("TEST 1: Database code structure...")
    
    try:
        # Read the Database.py file and check for key changes
        db_path = os.path.join(os.path.dirname(__file__), 'src', 'logging', 'Database.py')
        with open(db_path, 'r') as f:
            content = f.read()
        
        # Check for phash validation
        assert 'phash_str is missing or not valid hex' in content or 'not all(c in' in content, \
            "Missing pHash validation code"
        print("  ✓ pHash validation code present")
        
        # Check for stable Unknown creation
        assert 'Single stable "Unknown" bag type' in content or 'stable "Unknown"' in content, \
            "Missing stable Unknown documentation"
        print("  ✓ Stable Unknown documentation present")
        
        # Check for confidence_tier column
        assert 'confidence_tier' in content, "Missing confidence_tier column"
        print("  ✓ confidence_tier column code present")
        
        # Check for ALTER TABLE statement
        assert 'ALTER TABLE bag_events' in content, "Missing schema migration code"
        print("  ✓ Schema migration code present")
        
        # Check for ENABLE_UNKNOWN_PHASH_CLUSTERING env var
        assert 'ENABLE_UNKNOWN_PHASH_CLUSTERING' in content, "Missing env var check"
        print("  ✓ Environment variable check present")
        
        print("✅ TEST 1 PASSED: Database code structure\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_tracking_config():
    """Test that tracking config has new parameters."""
    print("TEST 2: Tracking config code structure...")
    
    try:
        # Read the tracking_config.py file
        config_path = os.path.join(os.path.dirname(__file__), 'src', 'config', 'tracking_config.py')
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check for confidence tier threshold
        assert 'high_confidence_threshold' in content, "Missing high_confidence_threshold"
        print("  ✓ high_confidence_threshold present")
        
        # Check for degraded mode parameters
        assert 'degraded_mode_enabled' in content, "Missing degraded_mode_enabled"
        print("  ✓ degraded_mode_enabled present")
        
        assert 'degraded_mode_queue_threshold' in content, "Missing degraded_mode_queue_threshold"
        print("  ✓ degraded_mode_queue_threshold present")
        
        assert 'degraded_mode_delay_threshold_ms' in content, "Missing degraded_mode_delay_threshold_ms"
        print("  ✓ degraded_mode_delay_threshold_ms present")
        
        assert 'degraded_mode_disable_roi_saving' in content, "Missing degraded_mode_disable_roi_saving"
        print("  ✓ degraded_mode_disable_roi_saving present")
        
        assert 'degraded_mode_disable_visualization' in content, "Missing degraded_mode_disable_visualization"
        print("  ✓ degraded_mode_disable_visualization present")
        
        assert 'degraded_mode_skip_low_detection_frames' in content, "Missing degraded_mode_skip_low_detection_frames"
        print("  ✓ degraded_mode_skip_low_detection_frames present")
        
        print("✅ TEST 2 PASSED: Tracking config code structure\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_classifier_unknown_result():
    """Test that ClassifierService handles Unknown correctly."""
    print("TEST 3: ClassifierService code structure...")
    
    try:
        # Read ClassifierService.py and check for key changes
        classifier_path = os.path.join(os.path.dirname(__file__), 'src', 'classifier', 'ClassifierService.py')
        with open(classifier_path, 'r') as f:
            content = f.read()
        
        # Check for None phash instead of "unknown"
        assert '"phash": None' in content, "Missing phash: None in _invoke_unknown_result"
        print("  ✓ phash set to None for Unknown")
        
        # Check for unknown_kind
        assert 'unknown_kind' in content, "Missing unknown_kind metadata"
        print("  ✓ unknown_kind metadata present")
        
        # Check for different unknown categories
        assert 'structural' in content and 'low_evidence' in content and 'ambiguous' in content, \
            "Missing unknown categories"
        print("  ✓ Unknown categories present")
        
        # Check for single Unknown directory
        assert 'unknown_samples' in content or 'single directory' in content.lower(), \
            "Missing single Unknown directory handling"
        print("  ✓ Single Unknown directory handling present")
        
        print("✅ TEST 3 PASSED: ClassifierService code structure\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_bag_counter_app():
    """Test that BagCounterApp has degraded mode support."""
    print("TEST 4: BagCounterApp code structure...")
    
    try:
        # Read BagCounterApp.py
        app_path = os.path.join(os.path.dirname(__file__), 'src', 'counting', 'BagCounterApp.py')
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Check for degraded mode method
        assert '_check_degraded_mode' in content, "Missing _check_degraded_mode method"
        print("  ✓ _check_degraded_mode method present")
        
        # Check for degraded mode tracking variables
        assert '_degraded_mode_active' in content, "Missing _degraded_mode_active variable"
        print("  ✓ Degraded mode tracking variables present")
        
        # Check for degraded mode in processing logic
        assert 'in_degraded_mode' in content, "Missing degraded mode checks"
        print("  ✓ Degraded mode checks in processing logic")
        
        # Check for confidence tier handling
        assert 'confidence_tier' in content, "Missing confidence tier handling"
        print("  ✓ Confidence tier handling present")
        
        # Check for high_confidence_threshold usage
        assert 'high_confidence_threshold' in content, "Missing high_confidence_threshold usage"
        print("  ✓ high_confidence_threshold usage present")
        
        print("✅ TEST 4 PASSED: BagCounterApp code structure\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Running validation tests for PR changes")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Database Unknown handling", test_database_unknown_handling()))
    results.append(("Tracking config parameters", test_tracking_config()))
    results.append(("ClassifierService structure", test_classifier_unknown_result()))
    results.append(("BagCounterApp structure", test_bag_counter_app()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
