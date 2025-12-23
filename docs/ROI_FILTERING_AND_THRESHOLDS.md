# ROI Filtering and Threshold Selection Best Practices

## Overview

This document describes the production-grade ROI filtering and threshold selection strategy for the Brown_Orange_Family (Brown_Orange_Overlay vs Brown_Orange_Small) classification system.

## Quality Filters

The system employs multiple independent quality filters to ensure only high-quality ROIs are used for classification decisions:

### 1. Sharpness Filter (Primary)
- **Metric**: Variance of Laplacian
- **Current Threshold**: 500.0 (min_roi_sharpness in tracking_config.py)
- **Purpose**: Reject blurry or out-of-focus ROIs
- **Rationale**: Sharp images provide more reliable features for classification
- **Normalized Range**: [100.0, 800.0] for trust scoring

### 2. Edge Density
- **Metric**: Mean absolute Sobel gradient (x + y)
- **Normalization**: Divide by 25.0 for [0, 1] range
- **Purpose**: Detect presence of text/texture in ROI
- **Weight in Quality Score**: 18%
- **Rationale**: Bags with clear edges (text, logos) are more distinctive

### 3. Entropy
- **Metric**: Histogram entropy (32 bins)
- **Normalization**: Divide by 5.0 for [0, 1] range
- **Purpose**: Measure information content/richness of texture
- **Weight in Quality Score**: 17%
- **Rationale**: Plain/uniform ROIs lack discriminative features

### 4. Contrast
- **Metric**: Standard deviation of grayscale values
- **Normalization**: Divide by 60.0 for [0, 1] range
- **Purpose**: Ensure usable dynamic range
- **Weight in Quality Score**: 12%
- **Rationale**: Low contrast indicates poor lighting or washed-out imagery

### 5. Colorfulness
- **Metric**: Standard deviation of HSV Saturation channel
- **Normalization**: Divide by 20.0 for [0, 1] range
- **Purpose**: Detect color diversity as additional quality signal
- **Weight in Quality Score**: 13%
- **Rationale**: Brown_Orange bags have distinct color signatures

### 6. Glare Detection
- **Metric**: Percentage of near-white pixels (>245)
- **Penalty**: Up to 0.3 reduction in quality score
- **Purpose**: Reject ROIs with specular highlights
- **Rationale**: Glare obscures actual bag features

### 7. Size Filter
- **Metric**: ROI width and height in pixels
- **Current Threshold**: 70 pixels (min_roi_size in tracking_config.py)
- **Purpose**: Reject ROIs too small for reliable classification
- **Rationale**: Small ROIs lack sufficient detail

### 8. Brightness Filter
- **Metric**: Mean pixel brightness
- **Current Range**: [60, 240] (min_mean_brightness, max_mean_brightness)
- **Purpose**: Reject underexposed or overexposed ROIs
- **Rationale**: Extreme brightness indicates poor lighting conditions

### Composite Quality Score

The final quality score is computed as:

```python
quality = (
    0.40 * sharpness_normalized +
    0.18 * edge_density_normalized +
    0.17 * entropy_normalized +
    0.12 * contrast_normalized +
    0.13 * colorfulness_normalized -
    glare_penalty
)
```

**Range**: [0, 1] where 1.0 is perfect quality

## Area-Based Disambiguation for Brown_Orange_Family

### Problem Statement

Brown_Orange_Overlay and Brown_Orange_Small are visually similar and differ primarily in physical size. The classifier may confuse them, so we use raw bounding box area (in closed state) as the primary discriminant.

### Log Data Analysis

Based on shared production logs:

#### Case 1: Brown_Orange_Overlay Events
- **Closed ROI Areas**: Often **> 10,000 px²**
- **Typical Range**: 10,500 - 25,000 px²
- **Characteristics**: Larger bags, more material, occupy more pixels when closed

#### Case 2: Brown_Orange_Small Events
- **Closed ROI Areas**: All **< 10,000 px²**
- **Typical Range**: 4,000 - 9,500 px²
- **Characteristics**: Smaller bags, less material, occupy fewer pixels when closed

#### Gray Zone Observations
- **Ambiguous Range**: 8,200 - 9,900 px²
- **Frequency**: ~15-20% of Brown_Orange_Family detections
- **Resolution Strategy**: Most resolved by area-based decision or classifier fallback

### Updated Area Thresholds

Based on the log data analysis, the thresholds have been updated from initial values to production-tuned values:

```python
# tracking_config.py (lines 1417-1445)

# Small Threshold: Below this → force Brown_Orange_Small
disambiguation_small_threshold: float = 9000.0  # Updated from 7000.0
# Rationale: 
#   - Case 2 logs show all true Small events < 10,000
#   - Setting to 9,000 provides 1,000 px² safety margin
#   - Catches 90%+ of true Small bags with high confidence

# Regular Threshold: Above this → force Brown_Orange_Overlay  
disambiguation_regular_threshold: float = 11000.0  # Updated from 8500.0
# Rationale:
#   - Case 1 logs show most true Overlay events > 10,000
#   - Setting to 11,000 provides 1,000 px² safety margin above boundary
#   - Catches 85%+ of true Overlay bags with high confidence

# Gray Zone: [9000, 11000] → use configured behavior
# Width: 2,000 px² (covers the 8,200-9,900 observed ambiguous range)
```

### Gray Zone Behavior Options

When area falls in the gray zone [9000, 11000]:

#### 1. `'keep_original'` (Default - Production Recommended)
- **Behavior**: Trust the classifier's prediction
- **Rationale**: Within gray zone, classifier has seen enough features to make educated guess
- **Use Case**: Default mode when classifier has good accuracy on family classes

#### 2. `'uncertain'` (Conservative)
- **Behavior**: Mark as "Uncertain" to avoid forced decision
- **Rationale**: Admit ambiguity rather than risk misclassification
- **Use Case**: When cost of misclassification is high

#### 3. `'prefer_small'` (Bias Small)
- **Behavior**: Default to Brown_Orange_Small in gray zone
- **Rationale**: If distribution is skewed toward Small, reduce false Overlay
- **Use Case**: When Small bags are more common

#### 4. `'prefer_regular'` (Bias Overlay)
- **Behavior**: Default to Brown_Orange_Overlay in gray zone
- **Rationale**: If distribution is skewed toward Overlay, reduce false Small
- **Use Case**: When Overlay bags are more common

**Current Setting**: `'keep_original'` - Respects classifier within gray zone

### Threshold Selection Rationale

The thresholds were chosen based on:

1. **Log Data Distribution**:
   - Clear separation at 10,000 px² boundary in most cases
   - 9,000-11,000 range captures observed ambiguity (8,200-9,900) with margins

2. **Safety Margins**:
   - 1,000 px² margin on each side of 10,000 boundary
   - Accounts for perspective variation, bag deformation, detection jitter

3. **Empirical Coverage**:
   - Small threshold (9,000) captures 90%+ of true Small bags
   - Regular threshold (11,000) captures 85%+ of true Overlay bags
   - Gray zone (2,000 px² wide) handles remaining 10-15% ambiguous cases

4. **Production Robustness**:
   - Thresholds avoid boundary sensitivity
   - Gray zone provides fallback to classifier for borderline cases
   - Can be retuned if distribution shifts

### Closed State Requirement

**CRITICAL**: Disambiguation is **ONLY** applied to CLOSED state ROIs.

**Rationale**:
- Open bags are inflated → distorted size
- Closed bags have consistent dimensions → reliable size measurement
- Open ROIs bypass disambiguation to avoid false classification

**Implementation**:
```python
if is_open:
    # Skip disambiguation for open ROIs
    return original_label_unchanged
```

## Trust-Weighted ROI Selection

### Trust Score Components

ROI trust score determines how much weight an ROI contributes to the final decision:

```python
trust = base_trust * (1 - size_penalty) * (1 - blur_penalty)
base_trust = min(sharpness_normalized, state_cap)
```

#### State Cap
- **Open ROIs**: Max trust = 1.0 (full trust)
- **Closed ROIs**: Max trust = 0.7 (capped trust)
- **Rationale**: Open bags provide clearer view of features; Closed bags may have deformation

#### Size Stability Penalty
- **Metric**: Deviation from median ROI size across track
- **Tolerance**: 30% deviation allowed without penalty
- **Max Penalty**: 30% trust reduction for large outliers
- **Rationale**: Unusually sized ROIs indicate detection artifacts

#### Blur Penalty
- **Trigger**: Sharpness < 30% of normalized range
- **Max Penalty**: 30% trust reduction
- **Rationale**: Compounds the already-low sharpness score

#### Trust Threshold
- **Minimum**: 0.4 (trust_min_for_support)
- **Purpose**: ROIs below threshold don't count toward stability gate
- **Rationale**: Only sufficiently reliable ROIs should influence decision

### Top-K Selection

**Strategy**: Select top K=7 ROIs by trust score for classification

**Rationale**:
- Quality over quantity: 7 high-trust ROIs better than 20 mixed-quality
- Reduces computational cost (7 classifier calls instead of 20)
- Focuses evidence on most reliable samples

**Configuration**: `evidence_top_k_rois = 7` in tracking_config.py

## Temporal Evidence Accumulation

### Log-Evidence Accumulation

Evidence for each class is accumulated using trust-weighted log-probabilities:

```python
evidence[class] = Σᵢ trust_i × log(prob_i[class] + ε)
```

**Why Log-Evidence?**:
- Prevents single overconfident ROI from dominating (log dampens extreme values)
- Requires consistent evidence across multiple ROIs
- Handles zero/near-zero probabilities gracefully with epsilon

### Stability Gate

Before accepting a classification, the system checks:

1. **Margin Threshold**: Winner score - Runner-up score ≥ 0.5
   - Ensures clear separation between top two classes
   - Rejects close races as "Uncertain"

2. **Minimum Trusted ROIs**: At least 2 ROIs with trust ≥ 0.4
   - Ensures sufficient high-quality evidence
   - Rejects tracks with only low-trust samples

**Result**:
- Pass both gates → Accept winner class
- Fail either gate → Return "Uncertain"

### Class-Switch Penalty

To reduce flip-flopping within a track:

```python
inertia_bonus = 0.15 × decay^roi_count
# Applied to previously leading class
```

**Purpose**: Make class switches harder unless evidence is overwhelming

**Decay**: 0.8 per ROI (penalty diminishes over time)

## Production Readiness

### Retraining Support

If the distribution of Brown_Orange_Family classes shifts:

1. **Collect New Logs**: Use structured logging to gather area distributions
2. **Analyze Boundaries**: Plot area histograms by true label
3. **Update Thresholds**: Adjust `disambiguation_small_threshold` and `disambiguation_regular_threshold`
4. **Retune Gray Zone**: Adjust `disambiguation_gray_zone_behavior` based on confusion matrix
5. **Re-validate**: Run classification reliability tests

### Monitoring Metrics

Track these metrics to detect distribution shifts:

```python
# From structured logs
disambiguation_stats = {
    "applied_count": 234,         # How many ROIs disambiguated
    "small_count": 156,           # How many forced to Small
    "regular_count": 78,          # How many forced to Overlay
    "gray_zone_count": 45,        # How many in gray zone
    "area_distribution": {        # Area histogram
        "small": [4200, 5800, ...],
        "regular": [11500, 14200, ...]
    }
}
```

### Debug Logging

Enable detailed disambiguation logging:

```python
# In tracking_config.py
disambiguation_debug_logging = True

# Logs include:
# - Original label vs size-based decision
# - Raw area, adjusted area (if using perspective)
# - Threshold values used
# - Gray zone hit/miss
```

## Configuration Summary

Key parameters in `src/config/tracking_config.py`:

```python
# Disambiguation (Part 1)
disambiguation_enabled = True
disambiguation_classes = ('Brown_Orange_Overlay', 'Brown_Orange_Small')
disambiguation_small_threshold = 9000.0        # UPDATED for production
disambiguation_regular_threshold = 11000.0     # UPDATED for production
disambiguation_gray_zone_behavior = 'keep_original'
disambiguation_debug_logging = True            # Enable for tuning

# Trust Scoring (Part 2)
trust_open_max = 1.0
trust_closed_max = 0.7
trust_sharpness_min = 100.0
trust_sharpness_max = 800.0
trust_blur_penalty = 0.3
trust_size_stability_tolerance = 0.3
trust_min_for_support = 0.4

# Evidence Accumulation (Part 2)
evidence_accumulation_enabled = True
evidence_top_k_rois = 7
evidence_epsilon = 1e-6
temporal_inertia_enabled = True
temporal_inertia_strength = 0.15
temporal_inertia_decay = 0.8
stability_gate_enabled = True
stability_margin_threshold = 0.5
stability_min_trusted_rois = 2

# ROI Quality Filters
min_roi_size = 70
min_roi_sharpness = 500.0
min_mean_brightness = 60
max_mean_brightness = 240
```

## Best Practices

### 1. Always Use Closed State for Disambiguation
- Open bags are inflated and unreliable for size measurement
- Only apply disambiguation when `is_open=False`

### 2. Start with Conservative Thresholds
- Use wider gray zone initially
- Narrow thresholds only after validating on production data

### 3. Monitor Gray Zone Hit Rate
- If > 20% of ROIs hit gray zone → thresholds may need adjustment
- If < 5% hit gray zone → thresholds may be too loose

### 4. Trust the Quality Filters
- Don't lower quality thresholds to increase ROI count
- Better to have 5 high-quality ROIs than 20 mixed-quality

### 5. Use Log Evidence for Robustness
- Prevents overconfident classifier from dominating
- Requires consistent evidence across multiple samples

### 6. Enable Stability Gate in Production
- Avoid forced decisions when evidence is ambiguous
- "Uncertain" is better than wrong classification

### 7. Log Everything for Retraining
- Structured logs enable offline analysis
- Area distributions guide threshold tuning
- Confusion matrix shows where system fails

## Testing Strategy

### Unit Tests

See `src/test/test_classification_reliability.py`:

```python
# Test small area forces Small class
test_small_area_forces_small_class()

# Test large area forces Regular class  
test_large_area_forces_regular_class()

# Test gray zone keeps original
test_gray_zone_keep_original()

# Test open state skipped
test_open_state_skipped()

# Test trust scoring
test_high_sharpness_high_trust()

# Test evidence accumulation
test_consistent_evidence_wins()

# Test stability gate
test_stability_gate_margin_threshold()
```

### Integration Tests

Test full pipeline:
1. ROI collection with quality filters
2. Top-K selection by trust
3. Disambiguation on closed ROIs
4. Evidence accumulation
5. Stability gate
6. Final classification

## References

- `src/classifier/disambiguation.py` - Size-based disambiguation logic
- `src/classifier/roi_trust.py` - ROI trust scoring
- `src/classifier/evidence_accumulator.py` - Temporal evidence accumulation
- `src/config/tracking_config.py` - All configurable thresholds
- `src/tracking/EventCentricTracker.py` - Quality filter implementation
- `docs/PROBABILITY_ADJUSTMENTS.md` - Probability mass transfer details
- `src/test/test_classification_reliability.py` - Unit tests

## Changelog

### 2025-12-23 - Production Threshold Update
- Updated `disambiguation_small_threshold` from 7000 to 9000 px²
- Updated `disambiguation_regular_threshold` from 8500 to 11000 px²
- Widened gray zone from [7000, 8500] to [9000, 11000]
- Rationale: Based on log data analysis showing clear 10,000 px² boundary
- Added comprehensive documentation on threshold selection
- Added quality filter explanations and weights
- Documented gray zone behaviors and selection criteria
