#!/usr/bin/env python3
"""
Simple validation script for disambiguation threshold updates.
Tests the updated production thresholds without requiring pytest.
"""

import sys
from dataclasses import dataclass

# Mock configuration matching production values
@dataclass
class MockConfig:
    disambiguation_enabled: bool = True
    disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
    disambiguation_small_threshold: float = 9000.0
    disambiguation_regular_threshold: float = 11000.0
    disambiguation_gray_zone_behavior: str = 'keep_original'
    disambiguation_debug_logging: bool = True
    disambiguation_family_name: str = 'Brown_Orange_Family'
    disambiguation_confidence_penalty: float = 0.9
    disambiguation_penalty_on_change_only: bool = False


def test_small_threshold():
    """Test that area below 9000 forces Small class."""
    from src.classifier.disambiguation import disambiguate_by_size
    
    config = MockConfig()
    
    # Small box: 80x100 = 8000 px² (below 9000 threshold)
    result = disambiguate_by_size(
        original_label="Brown_Orange_Overlay",  # Classifier said regular
        confidence=0.75,
        bbox=(100, 50, 180, 150),  # 80x100 = 8000
        is_open=False,
        config=config
    )
    
    assert result.raw_area < 9000, f"Expected area < 9000, got {result.raw_area}"
    assert result.label == "Brown_Orange_Small", f"Expected Small, got {result.label}"
    assert result.disambiguated is True
    print(f"✓ Small threshold test passed: area={result.raw_area:.0f} → {result.label}")


def test_regular_threshold():
    """Test that area above 11000 forces Overlay class."""
    from src.classifier.disambiguation import disambiguate_by_size
    
    config = MockConfig()
    
    # Large box: 120x100 = 12000 px² (above 11000 threshold)
    result = disambiguate_by_size(
        original_label="Brown_Orange_Small",  # Classifier said small
        confidence=0.75,
        bbox=(100, 50, 220, 150),  # 120x100 = 12000
        is_open=False,
        config=config
    )
    
    assert result.raw_area > 11000, f"Expected area > 11000, got {result.raw_area}"
    assert result.label == "Brown_Orange_Overlay", f"Expected Overlay, got {result.label}"
    assert result.disambiguated is True
    print(f"✓ Regular threshold test passed: area={result.raw_area:.0f} → {result.label}")


def test_gray_zone_lower():
    """Test lower boundary of gray zone (9100 px²)."""
    from src.classifier.disambiguation import disambiguate_by_size
    
    config = MockConfig()
    
    # Box: 91x100 = 9100 px² (just above small threshold)
    result = disambiguate_by_size(
        original_label="Brown_Orange_Small",
        confidence=0.7,
        bbox=(100, 50, 191, 150),  # 91x100 = 9100
        is_open=False,
        config=config
    )
    
    assert 9000 < result.raw_area < 11000, f"Expected gray zone, got {result.raw_area}"
    assert result.label == "Brown_Orange_Small", f"Expected to keep original Small, got {result.label}"
    assert "gray_zone" in result.reason
    print(f"✓ Gray zone lower test passed: area={result.raw_area:.0f} → {result.label} (kept original)")


def test_gray_zone_upper():
    """Test upper boundary of gray zone (10900 px²)."""
    from src.classifier.disambiguation import disambiguate_by_size
    
    config = MockConfig()
    
    # Box: 109x100 = 10900 px² (just below regular threshold)
    result = disambiguate_by_size(
        original_label="Brown_Orange_Overlay",
        confidence=0.7,
        bbox=(100, 50, 209, 150),  # 109x100 = 10900
        is_open=False,
        config=config
    )
    
    assert 9000 < result.raw_area < 11000, f"Expected gray zone, got {result.raw_area}"
    assert result.label == "Brown_Orange_Overlay", f"Expected to keep original Overlay, got {result.label}"
    assert "gray_zone" in result.reason
    print(f"✓ Gray zone upper test passed: area={result.raw_area:.0f} → {result.label} (kept original)")


def test_open_state_skipped():
    """Test that open state ROIs skip disambiguation."""
    from src.classifier.disambiguation import disambiguate_by_size
    
    config = MockConfig()
    
    # Box in gray zone but OPEN state
    result = disambiguate_by_size(
        original_label="Brown_Orange_Overlay",
        confidence=0.75,
        bbox=(100, 50, 200, 150),  # 100x100 = 10000 (gray zone)
        is_open=True,  # OPEN state
        config=config
    )
    
    assert result.label == "Brown_Orange_Overlay", f"Expected unchanged, got {result.label}"
    assert result.disambiguated is False
    assert result.reason == "skipped_open_state"
    print(f"✓ Open state skip test passed: {result.reason}")


def test_non_family_unchanged():
    """Test that non-family classes are unchanged."""
    from src.classifier.disambiguation import disambiguate_by_size
    
    config = MockConfig()
    
    result = disambiguate_by_size(
        original_label="Blue_Yellow",
        confidence=0.8,
        bbox=(100, 100, 200, 200),
        is_open=False,
        config=config
    )
    
    assert result.label == "Blue_Yellow"
    assert result.disambiguated is False
    assert result.reason == "not_target_family"
    print(f"✓ Non-family unchanged test passed: {result.reason}")


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("Disambiguation Threshold Validation")
    print("Production Thresholds:")
    print("  - Small threshold: 9000 px²")
    print("  - Regular threshold: 11000 px²")
    print("  - Gray zone: [9000, 11000]")
    print("="*60 + "\n")
    
    tests = [
        ("Small threshold boundary", test_small_threshold),
        ("Regular threshold boundary", test_regular_threshold),
        ("Gray zone lower boundary", test_gray_zone_lower),
        ("Gray zone upper boundary", test_gray_zone_upper),
        ("Open state skip", test_open_state_skipped),
        ("Non-family unchanged", test_non_family_unchanged),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
