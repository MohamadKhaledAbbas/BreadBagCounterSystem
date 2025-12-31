#!/usr/bin/env python3
"""
Tests for Spool Utility Functions

Tests the utility functions in spool_utils.py including:
- CRC32 checksum calculation
- ProcessorState persistence (save/load)
- Structured logging formatters
- Throttled logging
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.spool_utils import (
    calculate_crc32,
    save_processor_state,
    load_processor_state,
    ProcessorState,
    format_structured_log,
    throttled_log
)
from src.utils.AppLogging import logger


def test_crc32_calculation():
    """Test CRC32 checksum calculation."""
    # Test with known data
    data1 = b"Hello, World!"
    crc1 = calculate_crc32(data1)
    
    # CRC32 should be deterministic
    crc2 = calculate_crc32(data1)
    assert crc1 == crc2, "CRC32 should be deterministic"
    
    # Different data should have different CRC32
    data2 = b"Hello, World!!"
    crc3 = calculate_crc32(data2)
    assert crc1 != crc3, "Different data should have different CRC32"
    
    # Empty data has a known CRC32 value
    crc4 = calculate_crc32(b"")
    assert isinstance(crc4, int), "CRC32 should return an integer"
    assert crc4 >= 0, "CRC32 should be non-negative"
    
    print("✓ test_crc32_calculation passed")


def test_processor_state_persistence():
    """Test ProcessorState save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "test_state.json")
        
        # Create and save state
        state = ProcessorState(
            last_published_index=12345,
            last_published_segment=67,
            session_id="test-session-123",
            timestamp=time.time()
        )
        
        success = save_processor_state(state_path, state)
        assert success, "Should save state successfully"
        assert os.path.exists(state_path), "State file should exist"
        
        # Load state
        loaded = load_processor_state(state_path)
        assert loaded is not None, "Should load state successfully"
        assert loaded.last_published_index == 12345, "Index should match"
        assert loaded.last_published_segment == 67, "Segment should match"
        assert loaded.session_id == "test-session-123", "Session ID should match"
        
        # Test loading non-existent file
        nonexistent_path = os.path.join(tempfile.gettempdir(), "nonexistent_state_test.json")
        loaded2 = load_processor_state(nonexistent_path)
        assert loaded2 is None, "Should return None for non-existent file"
    
    print("✓ test_processor_state_persistence passed")


def test_format_structured_log():
    """Test structured log formatting."""
    # Basic formatting
    msg = format_structured_log("Test message", key1="value1", key2=42)
    assert "Test message:" in msg, "Should include base message"
    assert "key1=value1" in msg, "Should include key1"
    assert "key2=42" in msg, "Should include key2"
    
    # With hex formatting
    msg2 = format_structured_log("Frame info", index=100, crc32=0xABCD1234)
    assert "index=100" in msg2, "Should include index"
    assert "crc32=0xabcd1234" in msg2, "Should format CRC32 as hex"
    
    # No fields
    msg3 = format_structured_log("Simple message")
    assert msg3 == "Simple message", "Should return message as-is with no fields"
    
    print("✓ test_format_structured_log passed")


def test_throttled_log():
    """Test throttled logging."""
    log_calls = []
    
    def mock_logger(msg):
        log_calls.append(msg)
    
    throttle_dict = {}
    
    # First call should log
    result1 = throttled_log(
        mock_logger,
        "Test message 1",
        key="test_key",
        throttle_dict=throttle_dict,
        min_interval=0.5
    )
    assert result1, "First call should log"
    assert len(log_calls) == 1, "Should have logged once"
    
    # Second call immediately should be throttled
    result2 = throttled_log(
        mock_logger,
        "Test message 2",
        key="test_key",
        throttle_dict=throttle_dict,
        min_interval=0.5
    )
    assert not result2, "Second call should be throttled"
    assert len(log_calls) == 1, "Should still have only one log"
    
    # Wait and try again
    time.sleep(0.6)
    result3 = throttled_log(
        mock_logger,
        "Test message 3",
        key="test_key",
        throttle_dict=throttle_dict,
        min_interval=0.5
    )
    assert result3, "Third call after interval should log"
    assert len(log_calls) == 2, "Should have logged twice"
    
    # Different key should not be throttled
    result4 = throttled_log(
        mock_logger,
        "Test message 4",
        key="different_key",
        throttle_dict=throttle_dict,
        min_interval=0.5
    )
    assert result4, "Different key should log immediately"
    assert len(log_calls) == 3, "Should have logged three times"
    
    print("✓ test_throttled_log passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Spool Utility Functions")
    print("=" * 60)
    print()
    
    test_crc32_calculation()
    test_processor_state_persistence()
    test_format_structured_log()
    test_throttled_log()
    
    print()
    print("=" * 60)
    print("✓ All spool utility tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
