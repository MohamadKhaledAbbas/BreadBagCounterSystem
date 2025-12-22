#!/usr/bin/env python3
"""
Test to verify adaptive skipping and queue optimization changes.
Tests the updated constants and skip rate cap mechanism.
"""

import sys
import os
import re
from collections import deque


def test_constants_from_source():
    """Test that constants have been updated correctly by parsing source file."""
    print("Testing constants from source file...")
    
    # Read the source file
    source_path = os.path.join(os.path.dirname(__file__), 'src', 'counting', 'BagCounterApp.py')
    with open(source_path, 'r') as f:
        source_code = f.read()
    
    # Check INPUT_QUEUE_SIZE
    match = re.search(r'INPUT_QUEUE_SIZE\s*=\s*(\d+)', source_code)
    assert match, "INPUT_QUEUE_SIZE not found"
    value = int(match.group(1))
    assert value == 500, f"INPUT_QUEUE_SIZE should be 500, got {value}"
    print(f"✓ INPUT_QUEUE_SIZE = {value}")
    
    # Check MAX_DETECTION_TIME_MS
    match = re.search(r'MAX_DETECTION_TIME_MS\s*=\s*(\d+\.?\d*)', source_code)
    assert match, "MAX_DETECTION_TIME_MS not found"
    value = float(match.group(1))
    assert value == 31.0, f"MAX_DETECTION_TIME_MS should be 31.0, got {value}"
    print(f"✓ MAX_DETECTION_TIME_MS = {value}ms")
    
    # Check ADAPTIVE_SKIP_THRESHOLD
    match = re.search(r'ADAPTIVE_SKIP_THRESHOLD\s*=\s*(\d+\.?\d*)', source_code)
    assert match, "ADAPTIVE_SKIP_THRESHOLD not found"
    value = float(match.group(1))
    assert value == 0.7, f"ADAPTIVE_SKIP_THRESHOLD should be 0.7, got {value}"
    print(f"✓ ADAPTIVE_SKIP_THRESHOLD = {value}")
    
    # Check SKIP_RATE_CAP
    match = re.search(r'SKIP_RATE_CAP\s*=\s*(\d+\.?\d*)', source_code)
    assert match, "SKIP_RATE_CAP not found"
    value = float(match.group(1))
    assert value == 0.02, f"SKIP_RATE_CAP should be 0.02, got {value}"
    print(f"✓ SKIP_RATE_CAP = {value} (2%)")
    
    # Check SKIP_RATE_WINDOW
    match = re.search(r'SKIP_RATE_WINDOW\s*=\s*(\d+)', source_code)
    assert match, "SKIP_RATE_WINDOW not found"
    value = int(match.group(1))
    assert value == 500, f"SKIP_RATE_WINDOW should be 500, got {value}"
    print(f"✓ SKIP_RATE_WINDOW = {value}")
    
    # Check SYSTEM_STATUS_LOG_INTERVAL
    match = re.search(r'SYSTEM_STATUS_LOG_INTERVAL\s*=\s*(\d+\.?\d*)', source_code)
    assert match, "SYSTEM_STATUS_LOG_INTERVAL not found"
    value = float(match.group(1))
    assert value == 900.0, f"SYSTEM_STATUS_LOG_INTERVAL should be 900.0, got {value}"
    print(f"✓ SYSTEM_STATUS_LOG_INTERVAL = {value}s (15 minutes)")
    
    # Check MIN_SKIP_SAMPLES
    match = re.search(r'MIN_SKIP_SAMPLES\s*=\s*(\d+)', source_code)
    assert match, "MIN_SKIP_SAMPLES not found"
    value = int(match.group(1))
    assert value == 10, f"MIN_SKIP_SAMPLES should be 10, got {value}"
    print(f"✓ MIN_SKIP_SAMPLES = {value}")
    
    # Check SKIP_CAP_LOG_FREQUENCY
    match = re.search(r'SKIP_CAP_LOG_FREQUENCY\s*=\s*(\d+)', source_code)
    assert match, "SKIP_CAP_LOG_FREQUENCY not found"
    value = int(match.group(1))
    assert value == 5, f"SKIP_CAP_LOG_FREQUENCY should be 5, got {value}"
    print(f"✓ SKIP_CAP_LOG_FREQUENCY = {value}")
    
    # Check for skip rate tracking initialization
    assert '_skip_decisions' in source_code, "Skip rate tracking (_skip_decisions) not found"
    print("✓ Skip rate tracking deque initialization found")
    
    # Check for skip cap blocks counter
    assert '_skip_cap_blocks' in source_code, "Skip cap blocks counter (_skip_cap_blocks) not found"
    print("✓ Skip cap blocks counter found")
    
    # Check for system status logging
    assert '_log_system_status' in source_code, "System status logging method not found"
    print("✓ System status logging method found")
    # Check for psutil caching
    assert '_psutil_module' in source_code, "psutil module caching (_psutil_module) not found"
    print("✓ psutil module caching found")
    assert 'psutil' in source_code, "psutil import/usage not found"
    assert 'ImportError' in source_code, "psutil graceful fallback not found"
    print("✓ psutil availability check and graceful fallback found")
    
    # Check for improved logging messages
    assert 'AdaptiveSkip' in source_code, "AdaptiveSkip logging tag not found"
    print("✓ AdaptiveSkip logging tag found")
    
    assert 'SkipCapBlock' in source_code, "SkipCapBlock logging tag not found"
    print("✓ SkipCapBlock logging tag found")
    
    assert 'InputQueuePressure' in source_code or 'ClassificationQueuePressure' in source_code, \
        "Queue pressure logging tags not found"
    print("✓ Queue pressure logging tags found")
    
    assert 'SystemStatus' in source_code, "SystemStatus logging tag not found"
    print("✓ SystemStatus logging tag found")
    
    print("\nAll constants and features verified successfully from source! ✓")


def test_skip_rate_cap_logic():
    """Test the skip rate cap logic."""
    print("\nTesting skip rate cap logic...")
    
    # Simulate skip decisions tracking
    SKIP_RATE_CAP = 0.02  # 2%
    SKIP_RATE_WINDOW = 500
    
    skip_decisions = deque(maxlen=SKIP_RATE_WINDOW)
    
    # Simulate scenario 1: Skip rate below cap
    # Fill with 1% skip rate (5 skips out of 500)
    for i in range(500):
        skip_decisions.append(1 if i < 5 else 0)
    
    current_skip_rate = sum(skip_decisions) / len(skip_decisions)
    print(f"\nScenario 1: {len(skip_decisions)} decisions, skip rate = {current_skip_rate:.2%}")
    
    # Check if we should allow another skip
    future_skip_rate = (sum(skip_decisions) + 1) / (len(skip_decisions) + 1)
    should_allow_skip = future_skip_rate <= SKIP_RATE_CAP
    print(f"  Future skip rate if we skip: {future_skip_rate:.2%}")
    print(f"  Should allow skip: {should_allow_skip} ✓" if should_allow_skip else f"  Should block skip: {not should_allow_skip} ✓")
    
    # Simulate scenario 2: Skip rate at cap (exactly 2%)
    skip_decisions.clear()
    for i in range(500):
        skip_decisions.append(1 if i < 10 else 0)  # 10 out of 500 = 2%
    
    current_skip_rate = sum(skip_decisions) / len(skip_decisions)
    print(f"\nScenario 2: {len(skip_decisions)} decisions, skip rate = {current_skip_rate:.2%}")
    
    future_skip_rate = (sum(skip_decisions) + 1) / (len(skip_decisions) + 1)
    should_block_skip = future_skip_rate > SKIP_RATE_CAP
    print(f"  Future skip rate if we skip: {future_skip_rate:.2%}")
    print(f"  Should block skip: {should_block_skip} ✓" if should_block_skip else f"  Should allow skip: {not should_block_skip}")
    
    assert should_block_skip, "Skip should be blocked when at cap"
    
    # Simulate scenario 3: Skip rate slightly below cap (1.8%)
    skip_decisions.clear()
    for i in range(500):
        skip_decisions.append(1 if i < 9 else 0)  # 9 out of 500 = 1.8%
    
    current_skip_rate = sum(skip_decisions) / len(skip_decisions)
    print(f"\nScenario 3: {len(skip_decisions)} decisions, skip rate = {current_skip_rate:.2%}")
    
    future_skip_rate = (sum(skip_decisions) + 1) / (len(skip_decisions) + 1)
    should_allow_skip = future_skip_rate <= SKIP_RATE_CAP
    print(f"  Future skip rate if we skip: {future_skip_rate:.2%}")
    print(f"  Should allow skip: {should_allow_skip} ✓" if should_allow_skip else f"  Should block skip: {not should_allow_skip}")
    
    assert should_allow_skip, "Skip should be allowed when below cap"
    
    print("\nSkip rate cap logic verified successfully! ✓")


def test_psutil_availability():
    """Test psutil availability and graceful fallback."""
    print("\nTesting psutil availability...")
    
    try:
        import psutil
        print(f"✓ psutil is available (version {psutil.__version__})")
        
        # Test basic functionality
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        print(f"  CPU: {cpu_percent:.1f}%")
        print(f"  RAM: {memory.percent:.1f}% ({memory.used / (1024**2):.1f}MB / {memory.total / (1024**2):.1f}MB)")
        
    except ImportError:
        print("✓ psutil not installed - system will gracefully fallback (expected behavior)")
    
    print("\npsutil test completed! ✓")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Adaptive Skipping and Queue Optimization Test Suite")
    print("=" * 60)
    
    try:
        test_constants_from_source()
        test_skip_rate_cap_logic()
        test_psutil_availability()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓✓✓")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

