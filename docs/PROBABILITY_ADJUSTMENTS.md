# Probability Adjustments Documentation

## Overview

This document describes the probability adjustment mechanism (Variant B: probability mass transfer) implemented in the BreadBagCounterSystem to ensure that size-based disambiguation affects the final track label in the evidence-accumulation path.

## Problem Statement

When using evidence accumulation for classification, the system:
1. Runs classifier to get probability vectors for each ROI
2. Applies size-based disambiguation on CLOSED state ROIs that may flip the label (e.g., Overlay → Small)
3. Accumulates evidence using log-probabilities

**Issue**: If disambiguation flips the label but the probability vector remains unchanged, the evidence accumulator may still favor the original class, causing the final decision to contradict the disambiguation.

**Solution**: After disambiguation, adjust the probability vector to reflect the disambiguated label by transferring probability mass between sibling classes in the family.

## Architecture

### Components

```
ClassifierService (process method)
  ↓
1. classify_single_with_probs() → Get full probability vector
  ↓
2. apply_disambiguation() → Size-based label decision
  ↓
3. apply_probability_adjustment() → Adjust probs if label changed
  ↓
4. accumulate_track_evidence() → Evidence accumulation with adjusted probs
  ↓
5. finalize() → Final label from accumulator
```

### Module: `probability_adjustments.py`

Located at: `src/classifier/probability_adjustments.py`

#### Key Functions

**`apply_probability_adjustment()`**
```python
def apply_probability_adjustment(
    original_probs: Dict[str, float],
    from_label: Optional[str],
    to_label: Optional[str],
    family_classes: Optional[List[str]] = None,
    config: Optional[Any] = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Apply probability mass transfer based on disambiguation decision.
    
    Returns:
        - adjusted_probs: New probability vector
        - metadata: Details about the adjustment
    """
```

**Key Parameters:**
- `from_label`: Original classifier label
- `to_label`: Label after disambiguation
- `family_classes`: List of sibling classes (e.g., ['Overlay', 'Small'])
- `config`: TrackingConfig with adjustment parameters

**Return Value:**
- `adjusted_probs`: Normalized probability vector with mass transferred
- `metadata`: Dict containing:
  - `applied`: bool
  - `from_label`, `to_label`: str
  - `mass_transferred`: float
  - `before_from`, `before_to`: float (probabilities before adjustment)
  - `after_from`, `after_to`: float (probabilities after adjustment)
  - `normalization_applied`: bool
  - `reason`: str (explanation)

## Probability Mass Transfer Strategies

### 1. Full Transfer (Default)

**Strategy**: `prob_adjustment_strategy = 'full_transfer'`

Transfers ALL family mass to the target class.

**Algorithm:**
```python
family_mass = sum(probs[cls] for cls in family_classes)
adjusted_probs[to_label] = family_mass
for cls in family_classes:
    if cls != to_label:
        adjusted_probs[cls] = epsilon  # Near-zero
```

**Example:**
```python
Original: {Overlay: 0.6, Small: 0.3, White: 0.05, Bran: 0.05}
Disambiguation: Overlay → Small
Adjusted:  {Overlay: 0.0, Small: 0.9, White: 0.05, Bran: 0.05}
```

**Rationale**: Most conservative approach. Ensures evidence accumulator definitively respects the disambiguation decision.

### 2. Proportional Transfer

**Strategy**: `prob_adjustment_strategy = 'proportional_transfer'`

Transfers a configurable portion from source to target.

**Algorithm:**
```python
transfer_amount = probs[from_label] * transfer_ratio
adjusted_probs[from_label] -= transfer_amount
adjusted_probs[to_label] += transfer_amount
```

**Configuration:**
- `prob_adjustment_transfer_ratio`: 0.0 to 1.0 (default: 1.0)

**Example (ratio=0.5):**
```python
Original: {Overlay: 0.6, Small: 0.3, White: 0.1}
Disambiguation: Overlay → Small
Transfer: 0.6 * 0.5 = 0.3
Adjusted:  {Overlay: 0.3, Small: 0.6, White: 0.1}
```

### 3. Swap

**Strategy**: `prob_adjustment_strategy = 'swap'`

Swaps probabilities between from_label and to_label.

**Algorithm:**
```python
adjusted_probs[from_label] = probs[to_label]
adjusted_probs[to_label] = probs[from_label]
```

**Example:**
```python
Original: {Overlay: 0.7, Small: 0.2, White: 0.1}
Disambiguation: Overlay → Small
Adjusted:  {Overlay: 0.2, Small: 0.7, White: 0.1}
```

## Configuration

Located in: `src/config/tracking_config.py`

### Parameters

```python
# Strategy selection
prob_adjustment_strategy: str = 'full_transfer'
# Options: 'full_transfer', 'proportional_transfer', 'swap'

# Transfer ratio (for proportional_transfer)
prob_adjustment_transfer_ratio: float = 1.0
# Range: 0.0 - 1.0

# Epsilon for numerical stability
prob_adjustment_epsilon: float = 1e-9
# Prevents exact zeros in log-evidence

# Debug logging
prob_adjustment_debug_logging: bool = False
# Enable detailed logging of each adjustment
```

### Environment Variables

```bash
# Override via environment
PROB_ADJUSTMENT_STRATEGY=full_transfer
PROB_ADJUSTMENT_TRANSFER_RATIO=0.8
PROB_ADJUSTMENT_DEBUG=true
```

## Integration with ClassifierService

### Evidence Accumulation Path

When `evidence_accumulation_enabled = True`:

1. **Classify with probabilities:**
   ```python
   label, conf, probs = self._classify_single_with_probs(roi, idx)
   ```

2. **Apply disambiguation:**
   ```python
   label, conf, disambiguated, reason = self._apply_disambiguation(
       label, conf, bbox, image_height
   )
   ```

3. **Adjust probabilities if label changed:**
   ```python
   if disambiguated and original_label != label:
       adjusted_probs, metadata = apply_probability_adjustment(
           original_probs=probs,
           from_label=original_label,
           to_label=label,
           family_classes=family_classes,
           config=tracking_config
       )
       probs = adjusted_probs
   ```

4. **Accumulate evidence with adjusted probs:**
   ```python
   accumulator_result = accumulate_track_evidence(
       classifications_with_probs,
       tracking_config
   )
   ```

### Metadata

Classification results include:

```python
metadata = {
    "probability_adjustment_applied": True,
    "probability_adjustment_count": 3,
    "probability_adjustment_samples": [
        {
            "from_label": "Overlay",
            "to_label": "Small",
            "mass_transferred": 0.55,
            "before_from": 0.60,
            "before_to": 0.30,
            "after_from": 0.0,
            "after_to": 0.90,
            "reason": "full_transfer_to_Small"
        }
    ]
}
```

## BpuClassifier predict_probs Implementation

### Method Signature

```python
def predict_probs(self, image) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict class label, confidence, and full probability vector.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Tuple of (label, confidence, probs_dict) where:
        - label: predicted class name
        - confidence: probability of predicted class
        - probs_dict: {class_name: probability} for all classes
                     (normalized, non-negative, sums to ~1.0)
    """
```

### Contract

**Guarantees:**
1. Returns full probability vector for all known classes
2. Probabilities are normalized (sum = 1.0 ± epsilon)
3. All probabilities are non-negative
4. `label` corresponds to argmax(probs)
5. `confidence` equals probs[label]

**Example Output:**
```python
label = "Brown_Orange_Overlay"
confidence = 0.65
probs = {
    "Brown_Orange_Overlay": 0.65,
    "Brown_Orange_Small": 0.20,
    "White": 0.10,
    "Bran": 0.05
}
```

### Backward Compatibility

The existing `predict()` method remains unchanged:
```python
def predict(self, image) -> Tuple[str, float]:
    # Returns (label, confidence) as before
```

Classes can override `predict_probs()` to provide full probability vectors, or use the default implementation in `BaseClassifier` which wraps `predict()`.

## Log Analyzer Support

### New Metrics

The log analyzer (`tools/log_analyzer.py`) now tracks:

#### Evidence Accumulation
```json
{
  "evidence_accumulation": {
    "used_count": 245,
    "usage_rate": 0.98,
    "gate_passed_count": 220,
    "gate_failed_count": 25,
    "gate_pass_rate": 0.90,
    "gate_failure_reasons": {
      "margin_too_small": 15,
      "too_few_trusted_rois": 10
    },
    "inertia_applied_count": 89,
    "inertia_rate": 0.36,
    "trust_stats_samples": [...]
  }
}
```

#### Disambiguation
```json
{
  "disambiguation": {
    "applied_count": 78,
    "application_rate": 0.31,
    "samples": [
      {
        "track_id": 123,
        "label": "Brown_Orange_Small",
        "count": 3
      }
    ]
  }
}
```

#### Probability Adjustment
```json
{
  "probability_adjustment": {
    "applied_tracks": 78,
    "total_adjustments": 234,
    "application_rate": 0.31,
    "samples": [
      {
        "track_id": 123,
        "label": "Brown_Orange_Small",
        "from_label": "Brown_Orange_Overlay",
        "to_label": "Brown_Orange_Small",
        "mass_transferred": 0.55,
        "before_from": 0.60,
        "before_to": 0.30,
        "after_from": 0.0,
        "after_to": 0.90,
        "reason": "full_transfer_to_Brown_Orange_Small"
      }
    ]
  }
}
```

### Usage

```bash
# Analyze logs with new metrics
python tools/log_analyzer.py --log-dir ./data/logs --day 2025-12-21

# Output includes probability adjustment analysis
# in HTML and JSON reports
```

## Extending for New Disambiguation Families

To add a new disambiguation family (e.g., "LargeBag" vs "SmallBag"):

### 1. Update Configuration

Edit `src/config/tracking_config.py`:

```python
# Add new family configuration (optional - for future extensions)
disambiguation_classes_new_family: tuple = ('LargeBag', 'SmallBag')
```

### 2. Update Disambiguation Module

Edit `src/classifier/disambiguation.py`:

```python
def disambiguate_by_size(
    original_label: str,
    confidence: float,
    bbox: Tuple[float, float, float, float],
    image_height: int,
    config: Any
) -> DisambiguationResult:
    # Check for new family
    if original_label in ['LargeBag', 'SmallBag']:
        # Apply same size-based logic
        # ...
```

### 3. No Changes Needed

The probability adjustment module is already extensible:
- `family_classes` parameter accepts any list of sibling classes
- Strategies work for any family size
- Configuration is reusable

**Example Usage:**
```python
adjusted_probs, metadata = apply_probability_adjustment(
    original_probs=probs,
    from_label='LargeBag',
    to_label='SmallBag',
    family_classes=['LargeBag', 'SmallBag'],  # New family
    config=tracking_config
)
```

## Testing

### Test Suite

Located at: `src/test/test_probability_adjustments.py`

**Run tests:**
```bash
# With pytest (if available)
python -m pytest src/test/test_probability_adjustments.py -v

# Standalone mode
PYTHONPATH=/path/to/repo python src/test/test_probability_adjustments.py
```

### Test Coverage

- ✅ Full transfer strategy
- ✅ Proportional transfer strategy
- ✅ Swap strategy
- ✅ No adjustment when labels unchanged
- ✅ Missing label handling
- ✅ Probability validation
- ✅ Batch adjustments
- ✅ Integration pipeline

### Example Test

```python
def test_full_transfer_strategy():
    original_probs = {
        'Brown_Orange_Overlay': 0.6,
        'Brown_Orange_Small': 0.3,
        'White': 0.05,
        'Bran': 0.05
    }
    
    adjusted_probs, metadata = apply_probability_adjustment(
        original_probs=original_probs,
        from_label='Brown_Orange_Overlay',
        to_label='Brown_Orange_Small',
        family_classes=['Brown_Orange_Overlay', 'Brown_Orange_Small'],
        config=config
    )
    
    # Check family mass transfer (0.6 + 0.3 = 0.9 → Small)
    assert abs(adjusted_probs['Brown_Orange_Small'] - 0.9) < 1e-6
    assert adjusted_probs['Brown_Orange_Overlay'] < 1e-6
    
    # Check other classes unchanged
    assert abs(adjusted_probs['White'] - 0.05) < 1e-6
```

## Edge Cases Handled

### 1. Missing Labels

If `from_label` or `to_label` is not in the probability vector:
- No adjustment is applied
- Original probs are returned unchanged
- Metadata indicates reason: `"from_label_not_in_probs"` or `"to_label_not_in_probs"`

### 2. Same Label

If disambiguation doesn't change the label (`from_label == to_label`):
- No adjustment is applied
- Metadata indicates reason: `"no_label_change"`

### 3. Normalization

After adjustment, probabilities may sum to ≠ 1.0 due to floating-point errors:
- Automatic renormalization: `probs[k] = probs[k] / sum(probs.values())`
- Metadata includes `normalization_applied: true`

### 4. Near-Zero Probabilities

To prevent log(0) in evidence accumulation:
- Family members not matching target are set to `epsilon` (not exact 0)
- Configurable: `prob_adjustment_epsilon = 1e-9`

### 5. Empty Probability Vectors

If classifier returns empty or invalid probability vector:
- Validation catches this: `validate_probability_vector(probs)`
- Returns error indicating issue (e.g., `"empty_probability_vector"`)

## Performance Considerations

### Computational Cost

- Probability adjustment: **O(n)** where n = number of classes
- Typically 5-10 classes → negligible overhead (~microseconds)
- Applied only when disambiguation changes label (~30% of cases)

### Memory Overhead

- One additional dict per ROI in classifications_with_probs
- Metadata includes adjustment samples (limited to first 3 per track)
- Typical overhead: < 1KB per track

### Production Impact

- No impact on non-disambiguated classifications
- Minimal impact on disambiguated classifications (~0.1ms per adjustment)
- Evidence accumulation remains the bottleneck (log operations)

## Troubleshooting

### Issue: Adjusted probs don't sum to 1.0

**Cause**: Floating-point arithmetic errors

**Solution**: Enable normalization (automatically applied):
```python
probs_sum = sum(adjusted_probs.values())
if abs(probs_sum - 1.0) > 1e-6:
    adjusted_probs = {k: v / probs_sum for k, v in adjusted_probs.items()}
```

### Issue: Evidence accumulator still favors original class

**Checks:**
1. Verify `probability_adjustment_applied: true` in metadata
2. Check `mass_transferred` value (should be > 0)
3. Review `after_to` value (should have received mass)
4. Enable debug logging: `PROB_ADJUSTMENT_DEBUG=true`

### Issue: Classifier doesn't implement predict_probs

**Solution**: BpuClassifier now implements `predict_probs()`. If using a different classifier:
```python
class MyClassifier(BaseClassifier):
    def predict_probs(self, image):
        # Implement full probability vector return
        probs = self.model.predict_proba(image)
        label = argmax(probs)
        confidence = probs[label]
        probs_dict = {self.class_names[i]: p for i, p in enumerate(probs)}
        return label, confidence, probs_dict
```

## References

- **Implementation**: `src/classifier/probability_adjustments.py`
- **Integration**: `src/classifier/ClassifierService.py` (lines 885-973)
- **Configuration**: `src/config/tracking_config.py` (Part 1.5)
- **Tests**: `src/test/test_probability_adjustments.py`
- **Log Analysis**: `tools/log_analyzer.py` (classification section)

## Version History

- **V8 (2025-12-21)**: Initial implementation of probability adjustment mechanism
  - Full transfer, proportional transfer, and swap strategies
  - Integration with evidence accumulation path
  - Extended log analyzer support
  - Comprehensive test suite
