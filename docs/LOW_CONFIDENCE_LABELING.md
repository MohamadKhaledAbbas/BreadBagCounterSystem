# Low-Confidence Labeling Enhancement - Implementation Guide

## Overview

This document describes the enhanced low-confidence labeling system for the bread bag classification system. The enhancement ensures that all ambiguous, gray zone, or uncertain classifications are consistently flagged as 'low confidence' across all bag classes, with proper visibility in the UI, analytics, logs, and database.

## Implementation Date

December 25, 2025

## Key Features

### 1. No Generic Family Labels in Output
- All classifications now return **specific class labels** (e.g., `Brown_Orange_Small`, `Brown_Orange_Overlay`)
- Generic family labels (e.g., `Brown_Orange_Family`) are **never** returned to display/results
- When ambiguous, the system picks the most reasonable specific class based on size, shape, or best available features
- The ambiguity is flagged via the `confidence_tier` field

### 2. Confidence Tier Tracking
- **High Confidence**: Clear, unambiguous classifications
  - Large or small size bins (clearly outside gray zone)
  - No validation penalties
  - Classifier and size-based disambiguation agree
  
- **Low Confidence**: Ambiguous or uncertain classifications
  - Gray zone size (area between small and regular thresholds)
  - Validation penalties applied (suspicious aspect ratio, unrealistic area)
  - Label changed by disambiguation (classifier disagreed with size)
  - Original detection was a generic family label

### 3. Track-Level Confidence Aggregation
The system determines an overall `confidence_tier` for each tracked bag based on:
- Any ROI marked as low confidence → Track is low confidence
- Stability gate failure → Track is low confidence
- Final label is Unknown/Uncertain → Track is low confidence
- All checks pass → Track is high confidence

## Architecture

### Data Flow

```
Detection → Classification → Disambiguation → Confidence Tier → Database
                                    ↓
                            [Gray Zone Check]
                            [Validation Check]
                            [Label Change Check]
                            [Family Label Check]
                                    ↓
                            Set confidence_tier
                            ('high' or 'low')
```

### Module Changes

#### 1. `src/classifier/disambiguation_v2.py`

**Changes:**
- Added `confidence_tier` field to `DisambiguationV2Result`
- Updated `resolve_gray_zone()` to never return "Uncertain" or family labels
  - Picks best match based on area proximity to thresholds
  - All ambiguous resolutions are flagged as low confidence
- Added confidence tier determination logic in `disambiguate_v2()`
  - Gray zone → `'low'`
  - Validation penalty → `'low'`
  - Label changed → `'low'`
  - Family label resolved → `'low'`
  - Otherwise → `'high'`

**Example:**
```python
result = disambiguate_v2(
    original_label='Brown_Orange_Family',
    confidence=0.7,
    bbox=(100, 100, 200, 200),  # Gray zone area
    is_open=False,
    config=tracking_config
)

# Result:
# result.label = 'Brown_Orange_Small'  # Specific class
# result.confidence_tier = 'low'        # Flagged as ambiguous
# result.reason = 'gray_zone_keep_original_resolved'
```

#### 2. `src/classifier/ClassifierService.py`

**Changes:**
- Updated `_apply_disambiguation()` return signature to include `confidence_tier`
- Propagated `confidence_tier` through both evidence accumulation and legacy paths
- Added track-level confidence tier determination:
  ```python
  # Evidence accumulation path
  if low_confidence_rois > 0:
      metadata['track_confidence_tier'] = 'low'
  elif not accumulator_result.gate_passed:
      metadata['track_confidence_tier'] = 'low'
  elif final_label in ('Unknown', 'Uncertain'):
      metadata['track_confidence_tier'] = 'low'
  else:
      metadata['track_confidence_tier'] = 'high'
  ```

#### 3. `src/counting/BagCounterApp.py`

**Changes:**
- Updated `on_classification_result()` to use `track_confidence_tier` from metadata
- Falls back to threshold-based determination if metadata unavailable
- Passes `confidence_tier` to database logging

**Example:**
```python
# Priority order:
# 1. Use track_confidence_tier from metadata (from disambiguation)
# 2. Fall back to threshold-based (conf >= 0.5 = high)
confidence_tier = metadata.get('track_confidence_tier') or \
                  ('high' if conf >= 0.5 else 'low')
```

#### 4. `src/logging/Database.py`

**No changes required** - Already has `confidence_tier` column and logging support:
```python
def log_event(self, bag_type_id, track_id, confidence, confidence_tier='high'):
    # Logs event with confidence tier
```

## Configuration

All parameters are in `src/config/tracking_config.py`:

```python
# Core thresholds
disambiguation_small_threshold: float = 9000.0          # Below = small
disambiguation_regular_threshold: float = 11000.0      # Above = regular
# Gray zone is between these two values

# Validation thresholds
disambiguation_v2_min_aspect_ratio: float = 0.3        # Min width/height
disambiguation_v2_max_aspect_ratio: float = 3.0         # Max width/height
disambiguation_v2_min_realistic_area: float = 1000.0   # Min area
disambiguation_v2_max_realistic_area: float = 100000.0 # Max area

# Confidence threshold (for UI fallback)
high_confidence_threshold: float = 0.5
```

## UI & Analytics

### Analytics Dashboard

The analytics HTML already supports confidence tiers:

```html
<div class="hero-breakdown">
    <span class="tier-badge high-tier">دقة عالية: {{ total.high_count }}</span>
    <span class="tier-badge low-tier">دقة منخفضة: {{ total.low_count }}</span>
</div>

<div class="confidence-breakdown">
    <div class="conf-item high">
        <span class="conf-label">دقة عالية</span>
        <span class="conf-value">{{ c.high_count }}</span>
    </div>
    <div class="conf-item low">
        <span class="conf-label">دقة منخفضة</span>
        <span class="conf-value">{{ c.low_count }}</span>
    </div>
</div>
```

### Database Query

The `get_aggregated_stats()` method automatically provides confidence tier breakdown:

```python
SELECT
    bt.id,
    bt.name AS bag_type,
    COUNT(be.id) AS count,
    SUM(CASE WHEN be.confidence_tier = 'high' THEN 1 ELSE 0 END) AS high_count,
    SUM(CASE WHEN be.confidence_tier = 'low' THEN 1 ELSE 0 END) AS low_count
FROM bag_types bt
LEFT JOIN bag_events be ON bt.id = be.bag_type_id
GROUP BY bt.id
```

## Testing

### Test Coverage

The test suite (`src/test/test_confidence_tier.py`) covers:

1. ✅ Gray zone classifications marked as low confidence
2. ✅ Validation penalties trigger low confidence
3. ✅ Label changes trigger low confidence
4. ✅ Family label resolution triggers low confidence
5. ✅ Clear classifications remain high confidence
6. ✅ `resolve_gray_zone()` never returns "Uncertain"
7. ✅ `resolve_gray_zone()` never returns family labels
8. ✅ Open state ROIs skip disambiguation
9. ✅ Non-family classes skip disambiguation

### Running Tests

```bash
cd /home/runner/work/BreadBagCounterSystem/BreadBagCounterSystem
python src/test/test_confidence_tier.py
```

**Expected Output:**
```
======================================================================
Running Confidence Tier Tests
======================================================================

✓ test_gray_zone_marked_as_low_confidence PASSED
✓ test_validation_penalty_triggers_low_confidence PASSED
✓ test_label_changed_triggers_low_confidence PASSED
✓ test_family_label_resolved_triggers_low_confidence PASSED
✓ test_clear_classification_high_confidence PASSED
✓ test_resolve_gray_zone_never_returns_uncertain PASSED
✓ test_resolve_gray_zone_never_returns_family_label PASSED
✓ test_open_state_skips_disambiguation PASSED
✓ test_non_family_class_skips_disambiguation PASSED

======================================================================
Test Results: 9 passed, 0 failed out of 9 tests
======================================================================
```

## Production Behavior Examples

### Example 1: Gray Zone Detection

**Input:**
- ROI area: 10,000 px² (between 9,000 and 11,000)
- Classifier: `Brown_Orange_Overlay` (0.65 confidence)

**Output:**
- Label: `Brown_Orange_Overlay` (keeps original in gray zone)
- Confidence: 0.585 (0.65 × 0.9 penalty)
- **Confidence Tier: `'low'`** ← Gray zone flagged
- Reason: `gray_zone_keep_original`

**Database:**
```sql
INSERT INTO bag_events (bag_type_id, track_id, confidence, confidence_tier)
VALUES (5, 123, 0.585, 'low');
```

### Example 2: Clear Small Detection

**Input:**
- ROI area: 3,000 px² (< 5,000, very small)
- Classifier: `Brown_Orange_Overlay` (0.80 confidence)

**Output:**
- Label: `Brown_Orange_Small` (changed by size)
- Confidence: 0.72 (0.80 × 0.9 penalty)
- **Confidence Tier: `'low'`** ← Label changed
- Reason: `family_size_very_small`

### Example 3: Clear Large Detection

**Input:**
- ROI area: 40,000 px² (> 25,000, large)
- Classifier: `Brown_Orange_Overlay` (0.85 confidence)

**Output:**
- Label: `Brown_Orange_Overlay` (confirmed by size)
- Confidence: 0.765 (0.85 × 0.9 penalty)
- **Confidence Tier: `'high'`** ← Clear classification
- Reason: `family_size_large`

### Example 4: Family Label Resolution

**Input:**
- ROI area: 10,000 px² (gray zone)
- Classifier: `Brown_Orange_Family` (0.70 confidence)

**Output:**
- Label: `Brown_Orange_Small` (picked best match)
- Confidence: 0.63 (0.70 × 0.9 penalty)
- **Confidence Tier: `'low'`** ← Family label resolved + gray zone
- Reason: `gray_zone_keep_original_resolved`

## Monitoring & Logging

### Structured Logs

Disambiguation events include confidence tier information:

```json
{
  "timestamp": "2025-12-25T05:26:13.980Z",
  "level": "INFO",
  "component": "Disambiguation V2",
  "track_id": 123,
  "original_label": "Brown_Orange_Family",
  "final_label": "Brown_Orange_Small",
  "confidence_tier": "low",
  "reason": "gray_zone_keep_original_resolved",
  "area": 10000,
  "size_bin": "gray_zone"
}
```

### Statistics

ClassifierService tracks disambiguation statistics including confidence tiers:

```python
stats = classifier_service.get_statistics()
print(stats['disambiguation_v2'])

# Output:
{
  'enabled': True,
  'total_attempts': 150,
  'applied': 120,
  'label_changed': 45,
  'by_size_bin': {
    'very_small': 10,
    'small': 30,
    'gray_zone': 25,  # ← All flagged as low confidence
    'regular': 40,
    'large': 15
  }
}
```

## Migration & Backward Compatibility

### Existing Data
- No migration required - `confidence_tier` column already exists with default `'high'`
- New classifications will use enhanced logic
- Old records remain valid

### Fallback Behavior
- If `track_confidence_tier` not in metadata: falls back to threshold-based (conf >= 0.5)
- If disambiguation disabled: standard confidence threshold applies
- If V1 disambiguation: returns `'high'` (no low-confidence flagging)

### Feature Flags
```python
# Disable V2 disambiguation entirely
disambiguation_v2_enabled = False  # Falls back to V1

# Disable all disambiguation
disambiguation_enabled = False  # No disambiguation at all
```

## Acceptance Criteria Status

✅ **No output of generic family labels** - `resolve_gray_zone()` always returns specific class  
✅ **All ambiguous/gray zone results flagged as 'low confidence'** - Implemented in `disambiguate_v2()`  
✅ **Confidence tier visible in logs** - Structured logging includes `confidence_tier`  
✅ **Confidence tier visible in analytics** - UI displays `high_count` and `low_count`  
✅ **Confidence tier visible in UI** - Analytics dashboard shows breakdown  
✅ **Confidence tier visible in database** - `confidence_tier` column populated  
✅ **All code, config, and documentation centralized** - Changes in 3 files  
✅ **Tests cover the logic** - 9 comprehensive tests, all passing  
✅ **Production reliability maintained** - Backward compatible, feature-flagged  

## Summary

The low-confidence labeling enhancement provides:

1. **Clear Semantics**: No more generic labels in production output
2. **Transparency**: All ambiguous decisions are flagged
3. **Consistency**: Same logic applies to all classes (extensible beyond Brown_Orange_Family)
4. **Visibility**: Confidence tiers shown in UI, logs, analytics, and database
5. **Testing**: Comprehensive test coverage with 9 passing tests
6. **Backward Compatibility**: Graceful fallback, no breaking changes

**Status: ✅ PRODUCTION READY**

## References

- Implementation: `src/classifier/disambiguation_v2.py`
- Service Integration: `src/classifier/ClassifierService.py`
- App Integration: `src/counting/BagCounterApp.py`
- Database: `src/logging/Database.py`
- Tests: `src/test/test_confidence_tier.py`
- UI: `src/endpoint/templates/analytics.html`
- Configuration: `src/config/tracking_config.py`
