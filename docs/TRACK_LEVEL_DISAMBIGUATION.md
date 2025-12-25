# Track-Level Disambiguation Architecture

## Overview

The bag classification system has been refactored to apply size-based disambiguation **after** track aggregation/voting, not during per-ROI classification. This ensures:

1. All ROI classifications preserve their raw labels from the classifier
2. Aggregation/voting runs on raw labels to determine the overall winner
3. If the winner is a family/generic label, disambiguation runs ONCE using closed ROIs
4. Final output is always a specific subclass, never a generic family type

## Key Concepts

### Family Labels

A **family label** represents a group of visually similar classes that can be distinguished by size:

- **Explicit family name**: `Brown_Orange_Family` (generic label from classifier)
- **Family members**: `Brown_Orange_Overlay`, `Brown_Orange_Small` (specific subclasses)

The system treats both explicit family names and family member labels as "family labels" that may need disambiguation.

### Decision Order

```
1. Classify all ROIs → preserve raw labels (no per-ROI disambiguation)
2. Run aggregation/voting on raw labels → determine winner label
3. Check if winner is a family label:
   - NO → output winner as-is, assign confidence tier
   - YES → proceed to track-level disambiguation
4. Track-level disambiguation:
   - Select all closed ROIs (is_open==False)
   - If no closed ROIs → fallback to default subclass (Overlay) with 'low' tier
   - If closed ROIs exist → select best ROI by (trust, confidence, sharpness)
   - Run disambiguate_v2() ONCE on best closed ROI
   - Output resolved specific class with confidence tier from disambiguation
```

## Implementation Details

### Family Label Detection

```python
def _is_family_label(self, label: str) -> bool:
    """Check if label needs disambiguation."""
    # Check explicit family name
    if label == config.disambiguation_family_name:  # 'Brown_Orange_Family'
        return True
    
    # Check if label is a family member
    if label in config.disambiguation_classes:  # ('Brown_Orange_Overlay', 'Brown_Orange_Small')
        return True
    
    return False
```

### Track-Level Disambiguation

```python
def _disambiguate_track_family_label(
    self,
    final_label: str,
    final_confidence: float,
    candidates: List[Dict[str, Any]],
    track_id: int
) -> Tuple[str, float, str, Dict[str, Any]]:
    """Disambiguate family label at track level using closed ROIs."""
    
    # 1. Filter to closed ROIs only
    closed_candidates = [c for c in candidates if c['state'] == 'closed']
    
    # 2. If no closed ROIs, fallback to default
    if not closed_candidates:
        return default_subclass, final_confidence, 'low', {...}
    
    # 3. Select best closed ROI
    best_roi = max(closed_candidates, key=lambda c: (c['trust'], c['confidence'], c['sharpness']))
    
    # 4. Run disambiguation ONCE
    result = disambiguate_v2(
        original_label=final_label,
        confidence=final_confidence,
        bbox=best_roi['bbox'],
        is_open=False,
        config=config,
        context={'track_id': track_id, 'track_level': True}
    )
    
    return result.label, result.confidence, result.confidence_tier, metadata
```

### Fallback Logic

When no closed ROIs are available (all ROIs are from open state):

1. **Default subclass**: Use the first target class (typically 'Overlay')
2. **Confidence tier**: Always 'low' since we can't measure size reliably
3. **Metadata**: Mark as `fallback_used: true` for transparency

Fallback scenarios:
- No closed ROIs in track
- Closed ROIs have no bbox data
- Bbox validation fails (degenerate box)

## Configuration

All configuration is centralized in `tracking_config.py`:

```python
# Family definition
disambiguation_family_name: str = 'Brown_Orange_Family'
disambiguation_classes: tuple = ('Brown_Orange_Overlay', 'Brown_Orange_Small')

# Size thresholds
disambiguation_small_threshold: float = 9000.0  # pixels²
disambiguation_regular_threshold: float = 11000.0  # pixels²

# Gray zone behavior (ambiguous sizes)
disambiguation_gray_zone_behavior: str = 'keep_original'  # or 'prefer_small', 'prefer_regular'

# Confidence penalty when disambiguation changes label
disambiguation_confidence_penalty: float = 0.9
```

## Confidence Tiers

The confidence tier reflects the reliability of the final classification:

### High Confidence

- Clear size bin (very_small, small, regular, large)
- Classifier and size agree
- No validation issues
- Non-ambiguous cases

### Low Confidence

- Gray zone size (between thresholds)
- Validation penalty applied (suspicious aspect ratio, unrealistic area)
- Label changed by disambiguation (classifier disagreed with size)
- Fallback used (no closed ROIs or bbox data)
- Original label was a family label (needed resolution)

## Database and Logging

### Never Output Family Labels

The system ensures that database writes and logs never contain generic family types:

- Classification results always show specific subclass
- Metadata includes `original_family_label` if disambiguation was applied
- Confidence tier reflects the disambiguation quality

### Metadata Tracking

Track-level disambiguation adds comprehensive metadata:

```python
{
    'track_disambiguation': {
        'disambiguation_applied': True,
        'disambiguation_reason': 'family_size_regular (area=15000 > 11000)',
        'original_family_label': 'Brown_Orange_Family',
        'resolved_label': 'Brown_Orange_Overlay',
        'resolved_confidence': 0.72,
        'confidence_tier': 'high',
        'best_closed_roi_trust': 0.85,
        'best_closed_roi_confidence': 0.80,
        'total_closed_rois': 3,
        'fallback_used': False
    }
}
```

## Benefits of Track-Level Approach

### 1. Better Aggregation

Raw labels from all ROIs participate in voting, providing:
- More accurate representation of classifier confidence
- Better handling of mixed family member detections
- Cleaner separation of classification and disambiguation logic

### 2. Single Disambiguation Point

Running disambiguation once at track level:
- Reduces computational overhead (one call vs many per-ROI calls)
- Provides consistent decision per track
- Easier to debug and monitor

### 3. Explicit Fallback Handling

Clear fallback strategy when closed ROIs are unavailable:
- Transparent default selection
- Low confidence tier signals uncertainty
- Metadata tracks fallback usage for analysis

### 4. Generic to Future Families

The logic is family-agnostic:
- Works with any family name in config
- Easily extended to multiple families
- No hardcoded class names in core logic

## Testing

Comprehensive test coverage in `test_track_disambiguation.py`:

- Family label detection (explicit, members, non-family)
- Disambiguation with closed ROIs (Small, Overlay)
- Fallback scenarios (no closed ROIs, no bbox)
- Best ROI selection (trust, confidence, sharpness)
- Confidence tier assignment
- Metadata accuracy

Run tests:
```bash
python src/test/test_track_disambiguation.py
```

## Future Enhancements

Potential improvements to the architecture:

1. **Multiple Family Support**: Handle multiple family groups simultaneously
2. **Adaptive Thresholds**: Learn optimal thresholds from production data
3. **Ensemble Disambiguation**: Use multiple closed ROIs for voting
4. **Confidence Calibration**: Better confidence estimation for fallback cases
5. **Dynamic Default Selection**: Choose fallback based on historical distribution

## Migration Notes

### Changes from Previous Version

**Before (Per-ROI Disambiguation)**:
- Disambiguation ran during ROI classification
- Each ROI label could be changed individually
- Probability adjustments needed to maintain consistency
- Mixed specific/family labels in aggregation

**After (Track-Level Disambiguation)**:
- Disambiguation runs after aggregation
- Raw labels preserved during classification
- Single disambiguation decision per track
- Always outputs specific subclass

### Backward Compatibility

The refactoring maintains compatibility with:
- Existing `disambiguate_v2()` function (used at track level now)
- Configuration parameters (same names and meanings)
- Database schema (same fields, improved metadata)
- Logging format (enhanced with track-level context)

### Code Locations

Key files modified:
- `src/classifier/ClassifierService.py`: Main classification flow
- `src/test/test_track_disambiguation.py`: New test suite
- `docs/TRACK_LEVEL_DISAMBIGUATION.md`: This documentation

Original files preserved:
- `src/classifier/disambiguation_v2.py`: Unchanged, used at track level
- `src/config/tracking_config.py`: Configuration unchanged
