"""
Tests for H.264 NAL unit parsing.

These tests verify the NAL parsing functionality used for segment
boundary alignment and SPS/PPS extraction.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spool.h264_nal import (
    NALUnitType,
    find_start_codes,
    parse_nal_units,
    detect_frame_type,
    extract_sps_pps,
    is_idr_frame,
)


def test_find_start_codes_3byte():
    """Test finding 3-byte start codes (0x000001)."""
    # Simple test with 3-byte start codes
    data = bytes([
        0x00, 0x00, 0x01, 0x67,  # SPS NAL (start code + type 7)
        0x12, 0x34,              # SPS data
        0x00, 0x00, 0x01, 0x68,  # PPS NAL (start code + type 8)
        0x56, 0x78,              # PPS data
    ])
    
    start_codes = find_start_codes(data)
    
    # Should find 2 start codes
    assert len(start_codes) == 2, f"Expected 2 start codes, got {len(start_codes)}"
    
    # First start code at offset 3 (after 0x00 0x00 0x01), length 3
    assert start_codes[0] == (3, 3), f"First start code wrong: {start_codes[0]}"
    
    # Second start code at offset 9, length 3
    assert start_codes[1] == (9, 3), f"Second start code wrong: {start_codes[1]}"
    
    print("✓ test_find_start_codes_3byte passed")


def test_find_start_codes_4byte():
    """Test finding 4-byte start codes (0x00000001)."""
    data = bytes([
        0x00, 0x00, 0x00, 0x01, 0x67,  # SPS with 4-byte start code
        0x12, 0x34,
        0x00, 0x00, 0x00, 0x01, 0x68,  # PPS with 4-byte start code
        0x56, 0x78,
    ])
    
    start_codes = find_start_codes(data)
    
    assert len(start_codes) == 2, f"Expected 2 start codes, got {len(start_codes)}"
    assert start_codes[0][1] == 4, f"First start code should be 4-byte: {start_codes[0]}"
    assert start_codes[1][1] == 4, f"Second start code should be 4-byte: {start_codes[1]}"
    
    print("✓ test_find_start_codes_4byte passed")


def test_parse_nal_units_sps_pps():
    """Test parsing SPS and PPS NAL units."""
    data = bytes([
        0x00, 0x00, 0x01, 0x67,  # SPS (type 7)
        0xAA, 0xBB, 0xCC,        # SPS data
        0x00, 0x00, 0x01, 0x68,  # PPS (type 8)
        0xDD, 0xEE,              # PPS data
    ])
    
    nal_units = parse_nal_units(data)
    
    assert len(nal_units) == 2, f"Expected 2 NAL units, got {len(nal_units)}"
    
    # First NAL should be SPS
    assert nal_units[0].nal_type == NALUnitType.SPS, f"First NAL should be SPS: {nal_units[0].nal_type}"
    
    # Second NAL should be PPS
    assert nal_units[1].nal_type == NALUnitType.PPS, f"Second NAL should be PPS: {nal_units[1].nal_type}"
    
    print("✓ test_parse_nal_units_sps_pps passed")


def test_parse_nal_units_idr():
    """Test parsing IDR NAL unit."""
    data = bytes([
        0x00, 0x00, 0x01, 0x65,  # IDR slice (type 5)
        0x11, 0x22, 0x33, 0x44,  # IDR data
    ])
    
    nal_units = parse_nal_units(data)
    
    assert len(nal_units) == 1, f"Expected 1 NAL unit, got {len(nal_units)}"
    assert nal_units[0].nal_type == NALUnitType.IDR, f"NAL should be IDR: {nal_units[0].nal_type}"
    
    print("✓ test_parse_nal_units_idr passed")


def test_detect_frame_type():
    """Test frame type detection."""
    # Frame with SPS, PPS, and IDR
    idr_frame = bytes([
        0x00, 0x00, 0x01, 0x67, 0xAA,  # SPS
        0x00, 0x00, 0x01, 0x68, 0xBB,  # PPS
        0x00, 0x00, 0x01, 0x65, 0xCC,  # IDR
    ])
    
    has_idr, has_sps, has_pps = detect_frame_type(idr_frame)
    
    assert has_idr, "Should detect IDR"
    assert has_sps, "Should detect SPS"
    assert has_pps, "Should detect PPS"
    
    # Non-IDR frame (P-frame)
    p_frame = bytes([
        0x00, 0x00, 0x01, 0x41, 0xDD,  # Non-IDR slice (type 1)
    ])
    
    has_idr, has_sps, has_pps = detect_frame_type(p_frame)
    
    assert not has_idr, "Should not detect IDR in P-frame"
    assert not has_sps, "Should not detect SPS in P-frame"
    assert not has_pps, "Should not detect PPS in P-frame"
    
    print("✓ test_detect_frame_type passed")


def test_is_idr_frame():
    """Test IDR frame detection."""
    # IDR frame
    idr_data = bytes([
        0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB,
    ])
    
    assert is_idr_frame(idr_data), "Should detect IDR"
    
    # Non-IDR frame
    non_idr_data = bytes([
        0x00, 0x00, 0x01, 0x41, 0xCC, 0xDD,
    ])
    
    assert not is_idr_frame(non_idr_data), "Should not detect IDR"
    
    print("✓ test_is_idr_frame passed")


def test_extract_sps_pps():
    """Test SPS/PPS extraction."""
    data = bytes([
        0x00, 0x00, 0x01, 0x67,  # SPS start
        0x42, 0x00, 0x1E, 0x9A, # SPS data
        0x00, 0x00, 0x01, 0x68,  # PPS start
        0xCE, 0x3C, 0x80,       # PPS data
        0x00, 0x00, 0x01, 0x65,  # IDR start
        0x88, 0x84,             # IDR data
    ])
    
    sps_data, pps_data = extract_sps_pps(data)
    
    assert sps_data is not None, "Should extract SPS"
    assert pps_data is not None, "Should extract PPS"
    
    # SPS should include start code
    assert sps_data[:3] == b'\x00\x00\x01', "SPS should include start code"
    
    # SPS NAL type should be 7
    assert sps_data[3] & 0x1F == 7, "SPS NAL type should be 7"
    
    # PPS should include start code
    assert pps_data[:3] == b'\x00\x00\x01', "PPS should include start code"
    
    # PPS NAL type should be 8
    assert pps_data[3] & 0x1F == 8, "PPS NAL type should be 8"
    
    print("✓ test_extract_sps_pps passed")


def test_extract_sps_pps_missing():
    """Test SPS/PPS extraction when missing."""
    # Frame without SPS/PPS
    data = bytes([
        0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB,  # IDR only
    ])
    
    sps_data, pps_data = extract_sps_pps(data)
    
    assert sps_data is None, "Should not find SPS"
    assert pps_data is None, "Should not find PPS"
    
    print("✓ test_extract_sps_pps_missing passed")


def test_empty_data():
    """Test handling of empty data."""
    empty_data = bytes()
    
    start_codes = find_start_codes(empty_data)
    assert len(start_codes) == 0, "Empty data should have no start codes"
    
    nal_units = parse_nal_units(empty_data)
    assert len(nal_units) == 0, "Empty data should have no NAL units"
    
    has_idr, has_sps, has_pps = detect_frame_type(empty_data)
    assert not has_idr and not has_sps and not has_pps, "Empty data should have no frame types"
    
    assert not is_idr_frame(empty_data), "Empty data is not IDR"
    
    print("✓ test_empty_data passed")


def test_nal_unit_lengths():
    """Test that NAL unit lengths are calculated correctly."""
    data = bytes([
        0x00, 0x00, 0x01, 0x67,  # SPS (offset 3)
        0x11, 0x22, 0x33,        # 3 bytes of data
        0x00, 0x00, 0x01, 0x68,  # PPS (offset 10)
        0x44, 0x55,              # 2 bytes of data
    ])
    
    nal_units = parse_nal_units(data)
    
    assert len(nal_units) == 2
    
    # SPS: from offset 3, length should be 7-3 = 4 bytes (header + 3 data)
    # Actually 10-3 = 7 (next NAL start) - 3 (start code) = 4
    sps = nal_units[0]
    assert sps.offset == 3, f"SPS offset should be 3: {sps.offset}"
    assert sps.length == 4, f"SPS length should be 4: {sps.length}"
    
    # PPS: from offset 10 to end
    pps = nal_units[1]
    assert pps.offset == 10, f"PPS offset should be 10: {pps.offset}"
    assert pps.length == 3, f"PPS length should be 3: {pps.length}"
    
    print("✓ test_nal_unit_lengths passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing H.264 NAL Parsing")
    print("=" * 60)
    print()
    
    try:
        test_find_start_codes_3byte()
        test_find_start_codes_4byte()
        test_parse_nal_units_sps_pps()
        test_parse_nal_units_idr()
        test_detect_frame_type()
        test_is_idr_frame()
        test_extract_sps_pps()
        test_extract_sps_pps_missing()
        test_empty_data()
        test_nal_unit_lengths()
        
        print()
        print("=" * 60)
        print("✓ All H.264 NAL parsing tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
