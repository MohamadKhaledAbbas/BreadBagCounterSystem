# Disambiguation V2 Production Guide

## Overview

This document describes the production-grade Disambiguation V2 module for the Brown_Orange_Family classification system. V2 enhances the original disambiguation logic with multi-threshold size bins, robust validation, and comprehensive monitoring capabilities.

## Table of Contents

1. [What's New in V2](#whats-new-in-v2)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Deployment Guide](#deployment-guide)
5. [Monitoring & Metrics](#monitoring--metrics)
6. [Troubleshooting](#troubleshooting)
7. [Rollback Strategy](#rollback-strategy)

## What's New in V2

### Key Improvements

1. **Multi-Threshold Size Bins**
   - V1: Binary decision (Small < 9000 < Gray Zone < 11000 < Overlay)
   - V2: Five bins (very_small, small, gray_zone, regular, large) for more granular classification
   - Configurable thresholds for each bin

2. **Robust Validation**
   - Aspect ratio validation (detects elongated/squished bboxes)
   - Area range validation (detects unrealistic sizes)
   - Confidence penalties for suspicious bboxes
   - Graceful handling of degenerate bboxes

3. **Enhanced Gray Zone Strategies**
   - `keep_original`: Trust classifier (V1 behavior, **recommended**)
   - `prefer_small`: Bias toward small class
   - `prefer_regular`: Bias toward regular class
   - `use_confidence`: Decision based on classifier confidence threshold

4. **Detailed Diagnostics**
   - Before/after labels and confidence
   - Size bin classification
   - Validation results and penalties
   - Context tracking (track_id, frame_index)
   - Structured metadata for analysis

5. **Statistics Tracking**
   - Total attempts vs. applied disambiguations
   - Skip reasons (open_state, not_family, validation_failed)
   - Label change rate
   - Confidence penalty application rate
   - Distribution by size bin and resolution reason

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ClassifierService                         │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  _apply_disambiguation()                             │   │
│  │    • Checks if V2 enabled                            │   │
│  │    • Creates context (track_id, frame_index)         │   │
│  │    • Updates statistics                              │   │
│  └────────────────┬─────────────────────────────────────┘   │
│                   │                                           │
└───────────────────┼───────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              disambiguation_v2.py                            │
│                                                               │
│  disambiguate_v2()                                           │
│    ├── Check if V2 enabled                                   │
│    ├── Skip if open state                                    │
│    ├── Check if target family                                │
│    │                                                          │
│    ├── validate_bbox()                                       │
│    │     ├── Check dimensions                                │
│    │     ├── Check aspect ratio                              │
│    │     └── Check area range                                │
│    │                                                          │
│    ├── compute_size_bin()                                    │
│    │     └── Map area to bin (very_small → large)           │
│    │                                                          │
│    ├── Make decision based on bin                            │
│    │     ├── very_small/small → Small class                  │
│    │     ├── regular/large → Overlay class                   │
│    │     └── gray_zone → resolve_gray_zone()                 │
│    │                                                          │
│    └── Apply confidence penalties                            │
│          ├── Validation penalty (if suspicious)              │
│          └── Disambiguation penalty (if label changed)       │
│                                                               │
│  Returns: DisambiguationV2Result with label, confidence,     │
│           disambiguated flag, reason, and detailed metadata  │
└─────────────────────────────────────────────────────────────┘
```

### Decision Flow

```
Input: (label, confidence, bbox, is_open)
  │
  ├─→ V2 disabled? ────────────────────────→ Return original
  │
  ├─→ Open state? ─────────────────────────→ Skip (return original)
  │
  ├─→ Not target family? ──────────────────→ Skip (return original)
  │
  ├─→ validate_bbox()
  │     │
  │     ├─→ Degenerate? ───────────────────→ Skip (validation_failed)
  │     │
  │     └─→ Suspicious? ────────────────────→ Apply validation penalty
  │
  ├─→ compute_size_bin()
  │     │
  │     ├─→ very_small/small ──────────────→ Small class
  │     │
  │     ├─→ regular/large ─────────────────→ Overlay class
  │     │
  │     └─→ gray_zone ─────────────────────→ resolve_gray_zone()
  │                                              │
  │                                              ├─→ keep_original
  │                                              ├─→ uncertain
  │                                              ├─→ prefer_small
  │                                              ├─→ prefer_regular
  │                                              └─→ use_confidence
  │
  └─→ Apply confidence penalty (if label changed)
       │
       └─→ Return DisambiguationV2Result
```

## Configuration

All V2 parameters are centralized in `src/config/tracking_config.py`.

### Core Parameters

```python
# Enable/disable V2
disambiguation_v2_enabled: bool = True  # Set to False to use V1

# Multi-threshold bins
disambiguation_v2_very_small_threshold: float = 5000.0   # px²
disambiguation_small_threshold: float = 9000.0            # px² (shared with V1)
disambiguation_regular_threshold: float = 11000.0         # px² (shared with V1)
disambiguation_v2_large_threshold: float = 25000.0        # px²

# Gray zone behavior
disambiguation_gray_zone_behavior: str = 'keep_original'  # Recommended for production
# Options: 'keep_original', 'uncertain', 'prefer_small', 'prefer_regular', 'use_confidence'
```

### Validation Parameters

```python
# Aspect ratio validation
disambiguation_v2_min_aspect_ratio: float = 0.3          # width/height >= 0.3
disambiguation_v2_max_aspect_ratio: float = 3.0          # width/height <= 3.0
disambiguation_v2_aspect_ratio_penalty: float = 0.3      # 30% confidence reduction

# Area validation
disambiguation_v2_min_realistic_area: float = 1000.0     # px²
disambiguation_v2_max_realistic_area: float = 100000.0   # px²
disambiguation_v2_unrealistic_area_penalty: float = 0.5  # 50% confidence reduction
```

### Confidence Penalty

```python
# Penalty when disambiguation changes the label
disambiguation_confidence_penalty: float = 0.9            # 10% reduction
disambiguation_penalty_on_change_only: bool = False      # Apply to all or only changes?
```

### Debug Logging

```python
# Enable detailed logging for initial deployment
disambiguation_v2_debug_logging: bool = True             # Set to False after validation
```

### Environment Variable Overrides

All parameters can be overridden via environment variables:

```bash
# Enable/disable V2
export DISAMBIGUATION_V2_ENABLED=true

# Adjust thresholds
export DISAMBIGUATION_V2_VERY_SMALL_THRESHOLD=4500.0
export DISAMBIGUATION_SMALL_THRESHOLD=8500.0
export DISAMBIGUATION_REGULAR_THRESHOLD=11500.0
export DISAMBIGUATION_V2_LARGE_THRESHOLD=26000.0

# Change gray zone behavior
export DISAMBIGUATION_GRAY_ZONE_BEHAVIOR=use_confidence

# Enable debug logging
export DISAMBIGUATION_V2_DEBUG=true
```

## Deployment Guide

### Phase 1: Preparation (Day 0)

1. **Review Configuration**
   ```bash
   # Check current settings
   cd /path/to/BreadBagCounterSystem
   grep -A 5 "disambiguation_v2" src/config/tracking_config.py
   ```

2. **Run Tests**
   ```bash
   # Run V2 tests
   python src/test/test_disambiguation_v2.py
   
   # Expected: 20+ tests passing
   # Test Results: 20 passed, X failed (assertion formatting issues are OK)
   ```

3. **Enable Debug Logging**
   ```python
   # In tracking_config.py
   disambiguation_v2_debug_logging = True
   ```

### Phase 2: Shadow Mode (Days 1-3)

Deploy with V2 enabled but monitor closely.

1. **Deploy V2**
   ```bash
   # Ensure V2 is enabled
   export DISAMBIGUATION_V2_ENABLED=true
   export DISAMBIGUATION_V2_DEBUG=true
   
   # Start application
   python main.py
   ```

2. **Monitor Logs**
   ```bash
   # Check disambiguation decisions
   tail -f data/logs/*.jsonl | grep "Disambiguation V2"
   
   # Expected output:
   # [Disambiguation V2] Track 123 Frame 45: family=Brown_Orange_Family, 
   # original=Brown_Orange_Overlay(conf=0.750), final=Brown_Orange_Small(conf=0.675), 
   # bbox=(100, 100, 170, 150), area=3500, size_bin=very_small, 
   # validation=True, reason=family_size_very_small (area=3500 < 9000)
   ```

3. **Check Statistics**
   ```python
   # In your monitoring code
   stats = classifier_service.get_statistics()
   print(stats['disambiguation_v2'])
   
   # Expected fields:
   # - enabled: True
   # - total_attempts: N
   # - applied: M
   # - skipped_open_state: X
   # - label_changed: Y
   # - by_size_bin: {...}
   # - by_reason: {...}
   ```

### Phase 3: Validation (Days 4-7)

1. **Analyze Statistics**
   ```bash
   # Use log analyzer (enhanced version)
   python tools/log_analyzer.py --log-dir data/logs --day 2025-12-25
   
   # Check disambiguation section in report
   ```

2. **Compare V1 vs V2** (Optional)
   ```bash
   # Run with V1 for comparison
   export DISAMBIGUATION_V2_ENABLED=false
   python main.py
   
   # Compare classification accuracy
   ```

3. **Tune Thresholds** (If needed)
   ```python
   # Adjust based on observed data
   disambiguation_v2_very_small_threshold = 4500.0  # Lower if too many false Overlays
   disambiguation_v2_large_threshold = 26000.0       # Higher if too many false Smalls
   ```

### Phase 4: Production (Day 8+)

1. **Disable Debug Logging**
   ```python
   disambiguation_v2_debug_logging = False
   ```

2. **Monitor Long-Term**
   - Check statistics daily for first week
   - Weekly review after that
   - Alert on validation_failed rate > 5%

## Monitoring & Metrics

### Key Metrics

#### 1. Disambiguation Rate
```python
applied_rate = stats['disambiguation_v2']['applied'] / stats['disambiguation_v2']['total_attempts']
# Target: 80-90% (family members in closed state)
```

#### 2. Label Change Rate
```python
change_rate = stats['disambiguation_v2']['label_changed'] / stats['disambiguation_v2']['applied']
# Expected: 10-30% (depends on classifier accuracy)
```

#### 3. Validation Failure Rate
```python
validation_failure_rate = stats['disambiguation_v2']['validation_failed'] / stats['disambiguation_v2']['total_attempts']
# Target: < 5% (indicates detection quality)
```

#### 4. Size Bin Distribution
```python
size_bin_dist = stats['disambiguation_v2']['by_size_bin']
# Expected distribution:
# - very_small: 5-15%
# - small: 20-35%
# - gray_zone: 15-25%
# - regular: 25-40%
# - large: 10-20%
```

### Log Analysis Queries

#### Find All Disambiguation Decisions
```bash
cat data/logs/*.jsonl | jq 'select(.message | contains("Disambiguation V2"))'
```

#### Count by Size Bin
```bash
cat data/logs/*.jsonl | jq -r 'select(.message | contains("size_bin")) | .size_bin' | sort | uniq -c
```

#### Find Validation Failures
```bash
cat data/logs/*.jsonl | jq 'select(.message | contains("validation failed"))'
```

#### Analyze Gray Zone Decisions
```bash
cat data/logs/*.jsonl | jq 'select(.message | contains("gray_zone"))'
```

### Alerts

Set up alerts for the following conditions:

1. **High Validation Failure Rate**
   - Threshold: > 5%
   - Action: Check detection quality, review bbox dimensions

2. **Unexpected Size Bin Distribution**
   - Threshold: gray_zone > 30% or < 10%
   - Action: Review thresholds, may need tuning

3. **High Label Change Rate**
   - Threshold: > 40%
   - Action: Classifier may need retraining

## Troubleshooting

### Issue: Too Many False Overlays (Smalls Classified as Overlay)

**Symptoms:**
- Label change rate > 40%
- Most changes are Small → Overlay
- Gray zone decisions favor Overlay

**Diagnosis:**
```python
# Check threshold settings
print(f"Small threshold: {config.disambiguation_small_threshold}")
print(f"Regular threshold: {config.disambiguation_regular_threshold}")

# Analyze actual areas for true Smalls
cat logs/*.jsonl | jq 'select(.label == "Brown_Orange_Small") | .area'
```

**Solution:**
1. Increase `disambiguation_small_threshold` (e.g., 9000 → 9500)
2. Decrease `disambiguation_regular_threshold` (e.g., 11000 → 10500)
3. Change gray zone behavior to `prefer_small`

### Issue: Too Many False Smalls (Overlays Classified as Small)

**Symptoms:**
- Label change rate > 40%
- Most changes are Overlay → Small
- Gray zone decisions favor Small

**Solution:**
1. Decrease `disambiguation_small_threshold` (e.g., 9000 → 8500)
2. Increase `disambiguation_regular_threshold` (e.g., 11000 → 11500)
3. Change gray zone behavior to `prefer_regular`

### Issue: High Validation Failure Rate

**Symptoms:**
- `validation_failed` > 5%
- Logs show "degenerate_bbox" or "suspicious_aspect_ratio"

**Diagnosis:**
```bash
# Find common validation failures
cat logs/*.jsonl | jq 'select(.validation_reason != null) | .validation_reason' | sort | uniq -c
```

**Solution:**
1. Check detection model quality
2. Review camera setup (angle, distance)
3. Adjust validation thresholds if necessary:
   ```python
   disambiguation_v2_min_aspect_ratio = 0.2  # More lenient
   disambiguation_v2_max_aspect_ratio = 4.0   # More lenient
   ```

### Issue: Inconsistent Classifications

**Symptoms:**
- Same bag classified differently in consecutive frames
- High volatility in track label history

**Diagnosis:**
- Check if areas vary significantly between frames
- Review if bags are being manipulated during tracking

**Solution:**
1. Ensure using CLOSED state ROIs only (V2 skips open states)
2. Increase `disambiguation_confidence_penalty` to reduce confidence in changed labels
3. Enable temporal inertia in evidence accumulator

## Rollback Strategy

### Immediate Rollback (< 5 minutes)

If V2 causes critical issues:

```bash
# Disable V2, use V1
export DISAMBIGUATION_V2_ENABLED=false

# Restart application
./stop_app.sh
./run_app.sh
```

Or edit `tracking_config.py`:
```python
disambiguation_v2_enabled: bool = False
```

### Gradual Rollback (Testing Phase)

If V2 shows reduced accuracy:

1. **Switch to Conservative Settings**
   ```python
   disambiguation_gray_zone_behavior = 'uncertain'  # More conservative
   disambiguation_v2_debug_logging = True            # Re-enable debug
   ```

2. **Widen Gray Zone**
   ```python
   disambiguation_small_threshold = 8000.0   # Wider gray zone
   disambiguation_regular_threshold = 12000.0
   ```

3. **Monitor for Improvement**
   - If still problematic, disable V2
   - If improved, gradually narrow gray zone back

### Complete Rollback (If V2 is Incompatible)

1. Disable V2: `disambiguation_v2_enabled = False`
2. Remove V2-specific logging from monitoring
3. Document issues for future V2 improvements
4. Continue with V1 until V2 is fixed

## Revert Checklist

- [ ] Set `disambiguation_v2_enabled = False` in config or environment
- [ ] Restart application
- [ ] Verify V1 is being used (check logs for "disambiguation=ON" without "disambiguation_v2=ON")
- [ ] Confirm classifications return to expected behavior
- [ ] Update monitoring dashboards to reflect V1 usage
- [ ] Document reason for rollback
- [ ] Plan fixes for V2 issues

## Best Practices

1. **Always Enable Debug Logging Initially**
   - Set `disambiguation_v2_debug_logging = True` for first week
   - Disable after validating behavior

2. **Monitor Statistics Daily**
   - Review `get_statistics()['disambiguation_v2']`
   - Track trends over time

3. **Tune Conservatively**
   - Make small threshold adjustments (±500 px²)
   - Test for at least 24 hours before further changes

4. **Keep Gray Zone Reasonable**
   - Target width: 2000-3000 px²
   - Too narrow: ambiguous cases get forced decisions
   - Too wide: classifier is ignored too often

5. **Prefer `keep_original` for Gray Zone**
   - Trusts classifier within ambiguous range
   - Only use other strategies if classifier consistently fails in gray zone

6. **Document All Changes**
   - Log threshold adjustments with rationale
   - Track accuracy improvements/regressions

## Future Enhancements

Potential improvements for V3:

1. **Machine Learning-Based Thresholds**
   - Auto-tune thresholds based on labeled data
   - Adaptive thresholds per camera/setup

2. **Multi-Feature Disambiguation**
   - Combine size with color histograms
   - Use texture features in gray zone

3. **Confidence Boosting**
   - Increase confidence for clear size bins
   - Reduce penalty when size strongly agrees with classifier

4. **Temporal Smoothing**
   - Consider previous frames' size measurements
   - Reduce frame-to-frame jitter

## References

- [Original Disambiguation (V1)](/src/classifier/disambiguation.py)
- [V2 Implementation](/src/classifier/disambiguation_v2.py)
- [V2 Tests](/src/test/test_disambiguation_v2.py)
- [Configuration](/src/config/tracking_config.py)
- [ROI Filtering Guide](/docs/ROI_FILTERING_AND_THRESHOLDS.md)
- [Classification Reliability](/docs/CLASSIFICATION_STABILITY_HEURISTICS.md)

## Support

For issues or questions:

1. Check logs: `data/logs/*.jsonl`
2. Review statistics: `classifier_service.get_statistics()['disambiguation_v2']`
3. Consult this guide's troubleshooting section
4. Contact: [MohamadKhaledAbbas](https://github.com/MohamadKhaledAbbas)
