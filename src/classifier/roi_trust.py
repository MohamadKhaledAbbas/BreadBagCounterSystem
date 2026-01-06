"""
ROI Trust Scoring Module for Evidence-Weighted Classification.

This module computes trust scores for ROI candidates based on quality metrics
that are independent of the predicted class. The trust score determines how
much weight an ROI's classification contributes to the final track-level decision.

## Trust Score Composition

Trust scoring uses multiple quality dimensions to ensure only high-quality ROIs
influence the final classification:

### Primary Components:
- **Sharpness**: Primary component - sharper images are more reliable
  - Measured by Laplacian variance (higher = sharper)
  - Normalized to [0, 1] using configurable min/max thresholds
  - Weight: Acts as base trust before penalties
  
- **State (Open/Closed)**: Open ROIs can reach max trust; Closed are capped
  - Open max: 1.0 (full view of bag features)
  - Closed max: 0.7 (may have deformation/obscured features)
  - Ensures closed ROIs don't overpower open ROIs in evidence

### Penalties Applied:
- **Motion Blur**: Penalty for blurry ROIs (inferred from low sharpness)
  - Applied when sharpness < 30% of normalized range
  - Max penalty: 30% trust reduction
  
- **Size Stability**: Penalty for outlier sizes vs track median
  - Tolerance: 30% deviation allowed without penalty
  - Max penalty: 30% trust reduction for large outliers
  - Detects detection artifacts or bag flip anomalies

### Trust Threshold:
- **Minimum for Support**: 0.4 (configurable)
  - ROIs below this don't count toward stability gate
  - Ensures only sufficiently reliable ROIs influence decision

## Quality Filters (Pre-Trust Scoring)

Before trust scoring, ROIs pass through multiple quality filters in EventCentricTracker:

1. **Sharpness Filter** (min_roi_sharpness: 500.0):
   - Rejects blurry/out-of-focus ROIs
   - Variance of Laplacian < threshold → rejected

2. **Edge Density** (computed in quality score):
   - Mean absolute Sobel gradient
   - Detects presence of text/texture
   - Normalized by dividing by 25.0

3. **Entropy** (computed in quality score):
   - Histogram entropy (32 bins)
   - Measures information content/richness
   - Normalized by dividing by 5.0

4. **Contrast** (computed in quality score):
   - Standard deviation of grayscale values
   - Ensures usable dynamic range
   - Normalized by dividing by 60.0

5. **Colorfulness** (computed in quality score):
   - Standard deviation of HSV Saturation channel
   - Detects color diversity
   - Normalized by dividing by 20.0

6. **Glare Detection** (computed in quality score):
   - Percentage of near-white pixels (>245)
   - Penalizes specular highlights
   - Max penalty: 0.3 reduction in quality

7. **Size Filter** (min_roi_size: 70px):
   - Rejects ROIs too small for reliable features
   
8. **Brightness Filter** ([60, 240] range):
   - Rejects underexposed or overexposed ROIs

See docs/ROI_FILTERING_AND_THRESHOLDS.md for complete documentation.

## Design Philosophy

The goal is to ensure that **quantity of ROIs is irrelevant - only quality matters**.
Low-quality ROIs cannot outweigh high-quality ones in evidence accumulation.

This is achieved through:
1. Hard rejection filters (sharpness, size, brightness) - eliminate unusable ROIs
2. Trust scoring - weight remaining ROIs by reliability
3. Top-K selection - use only the best K ROIs regardless of total collected
4. Stability gate - require minimum number of high-trust ROIs

## Usage

    from src.classifier.roi_trust import compute_roi_trust, ROITrustResult
    
    result = compute_roi_trust(
        sharpness=450.0,
        is_open=True,
        roi_size=(100, 80),
        median_size=(95, 82),
        config=tracking_config
    )
    
    trust_score = result.trust  # Use for evidence weighting
    is_reliable = result.is_trusted  # True if trust >= min_for_support

All parameters are centralized in tracking_config.py for easy tuning.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from src.utils.AppLogging import logger


@dataclass
class ROITrustResult:
    """Result of ROI trust scoring."""
    trust: float  # Final trust score in [0, 1]
    sharpness_component: float  # Contribution from sharpness
    state_cap: float  # Maximum allowed (open vs closed)
    size_penalty: float  # Penalty from size instability (0 = no penalty)
    blur_penalty: float  # Penalty from detected blur
    is_trusted: bool  # True if trust >= min_for_support threshold
    reason: str  # Human-readable explanation


def normalize_sharpness(
    sharpness: float,
    sharpness_min: float = 100.0,
    sharpness_max: float = 800.0
) -> float:
    """
    Normalize sharpness to [0, 1] range.
    
    Args:
        sharpness: Raw sharpness value (Laplacian variance)
        sharpness_min: Minimum expected sharpness
        sharpness_max: Maximum expected sharpness
        
    Returns:
        Normalized sharpness in [0, 1]
    """
    if sharpness <= sharpness_min:
        return 0.0
    elif sharpness >= sharpness_max:
        return 1.0
    else:
        return (sharpness - sharpness_min) / (sharpness_max - sharpness_min)


def compute_size_deviation(
    roi_size: Tuple[int, int],
    median_size: Optional[Tuple[int, int]]
) -> float:
    """
    Compute size deviation from median as a fraction.
    
    Args:
        roi_size: (width, height) of the ROI
        median_size: (width, height) median size of track's ROIs
        
    Returns:
        Deviation as fraction (0 = same size, 0.5 = 50% different)
    """
    if median_size is None:
        return 0.0
    
    roi_area = roi_size[0] * roi_size[1]
    median_area = median_size[0] * median_size[1]
    
    if median_area <= 0:
        return 0.0
    
    deviation = abs(roi_area - median_area) / median_area
    return min(deviation, 1.0)  # Cap at 1.0


def compute_roi_trust(
    sharpness: float,
    is_open: bool,
    roi_size: Tuple[int, int],
    median_size: Optional[Tuple[int, int]],
    config: Any
) -> ROITrustResult:
    """
    Compute trust score for a single ROI based on quality metrics.
    
    The trust score is class-independent - it measures how reliable
    the ROI is as evidence, regardless of what class it predicts.
    
    Trust is computed as:
        base_trust = min(sharpness_normalized, state_cap)
        trust = base_trust * (1 - size_penalty) * (1 - blur_penalty)
        
    This ensures:
    1. Sharper ROIs get higher trust (primary discriminant)
    2. Closed ROIs are capped to prevent overpowering open ROIs
    3. Size outliers are penalized (likely detection artifacts)
    4. Very blurry ROIs are further penalized beyond low sharpness
    
    Args:
        sharpness: Laplacian variance (higher = sharper)
        is_open: True if ROI is from an Open detection
        roi_size: (width, height) of the ROI in pixels
        median_size: Median (width, height) across track's ROIs (for stability)
        config: TrackingConfig object with trust parameters
        
    Returns:
        ROITrustResult with trust score and component breakdown
        
    Trust Calculation Steps:
        1. Normalize sharpness to [0, 1] using configurable min/max
        2. Apply state cap (Open=1.0, Closed=0.7)
        3. Compute and apply size stability penalty if ROI is outlier
        4. Apply blur penalty if sharpness is very low (<30% normalized)
        5. Clamp final trust to [0, 1]
        6. Check if trust meets minimum for support threshold
    """
    # Get configuration parameters
    open_max = getattr(config, 'trust_open_max', 1.0)
    closed_max = getattr(config, 'trust_closed_max', 0.7)
    sharpness_min = getattr(config, 'trust_sharpness_min', 100.0)
    sharpness_max = getattr(config, 'trust_sharpness_max', 800.0)
    blur_penalty = getattr(config, 'trust_blur_penalty', 0.3)
    size_tolerance = getattr(config, 'trust_size_stability_tolerance', 0.3)
    trust_min_for_support = getattr(config, 'trust_min_for_support', 0.4)
    
    # === STEP 1: Normalize sharpness to [0, 1] ===
    # Maps raw Laplacian variance to normalized score
    # Example: sharpness=450 with min=100, max=800 → 0.5
    sharpness_norm = normalize_sharpness(
        sharpness=sharpness,
        sharpness_min=sharpness_min,
        sharpness_max=sharpness_max
    )
    
    # === STEP 2: Determine state cap ===
    # Open ROIs: Can reach full trust (1.0) - clear view of features
    # Closed ROIs: Capped at 0.7 - may have deformation/obscured features
    # This prevents closed ROIs from overpowering open ROIs in evidence
    state_cap = open_max if is_open else closed_max
    
    # === STEP 3: Compute size deviation penalty ===
    # Penalize ROIs with unusual size compared to track median
    # Rationale: Size outliers often indicate detection artifacts or bag flips
    size_deviation = compute_size_deviation(roi_size, median_size)
    size_penalty = 0.0
    if size_deviation > size_tolerance:
        # Penalize ROIs with size deviation > 30%
        # Example: 50% deviation → 10% penalty (0.5 - 0.3) * 0.5 = 0.1
        excess = size_deviation - size_tolerance
        size_penalty = min(excess * 0.5, 0.3)  # Max 30% penalty
    
    # === STEP 4: Compute blur penalty ===
    # Additional penalty for very blurry ROIs (compounds low sharpness)
    # Applied when normalized sharpness < 30% (very blurry)
    applied_blur_penalty = 0.0
    if sharpness_norm < 0.3:  # Below 30% normalized sharpness
        # Scale penalty from 0 (at 30%) to full penalty (at 0%)
        # Example: sharpness_norm=0.15 → penalty = 0.3 * (1 - 0.15/0.3) = 0.15
        applied_blur_penalty = blur_penalty * (1.0 - sharpness_norm / 0.3)
    
    # === STEP 5: Calculate final trust ===
    # Start with sharpness as base (primary quality indicator)
    base_trust = sharpness_norm
    
    # Apply state cap (open vs closed)
    trust = min(base_trust, state_cap)
    
    # Apply penalties (multiplicative reduction to preserve scale)
    # Example: trust=0.7, size_penalty=0.1, blur_penalty=0.15
    #          → trust = 0.7 * 0.9 * 0.85 = 0.5355
    trust = trust * (1.0 - size_penalty)
    trust = trust * (1.0 - applied_blur_penalty)
    
    # Clamp to [0, 1] for safety (should not exceed 1.0)
    trust = max(0.0, min(1.0, trust))
    
    # === STEP 6: Determine if this ROI is "trusted" ===
    # Trusted ROIs meet the minimum threshold for supporting evidence
    # Used in stability gate to ensure sufficient high-quality evidence
    is_trusted = trust >= trust_min_for_support
    
    # Build reason string
    reasons = []
    if sharpness_norm < 0.5:
        reasons.append(f"low_sharpness({sharpness:.0f})")
    if not is_open:
        reasons.append(f"closed_cap({closed_max})")
    if size_penalty > 0:
        reasons.append(f"size_outlier({size_deviation:.2f})")
    if applied_blur_penalty > 0:
        reasons.append(f"blur_penalty({applied_blur_penalty:.2f})")
    
    reason = ", ".join(reasons) if reasons else "good_quality"
    
    return ROITrustResult(
        trust=trust,
        sharpness_component=sharpness_norm,
        state_cap=state_cap,
        size_penalty=size_penalty,
        blur_penalty=applied_blur_penalty,
        is_trusted=is_trusted,
        reason=reason
    )


def compute_track_trust_scores(
    roi_candidates: List[Dict[str, Any]],
    config: Any
) -> List[Dict[str, Any]]:
    """
    Compute trust scores for all ROI candidates in a track.
    
    Also computes the median size for stability analysis.
    
    Args:
        roi_candidates: List of ROI candidate dicts with 'sharpness', 'size', 'is_open' keys
        config: TrackingConfig object
        
    Returns:
        List of candidates with 'trust' and 'trust_result' added
    """
    if not roi_candidates:
        return []
    
    # Compute median size across all candidates
    sizes = [(c.get('size', (100, 100))[0], c.get('size', (100, 100))[1]) 
             for c in roi_candidates]
    
    if sizes:
        median_width = sorted([s[0] for s in sizes])[len(sizes) // 2]
        median_height = sorted([s[1] for s in sizes])[len(sizes) // 2]
        median_size = (median_width, median_height)
    else:
        median_size = None
    
    results = []
    for candidate in roi_candidates:
        sharpness = candidate.get('sharpness', 0.0)
        is_open = candidate.get('is_open', True) or candidate.get('state') == 'open'
        roi_size = candidate.get('size', (100, 100))
        
        trust_result = compute_roi_trust(
            sharpness=sharpness,
            is_open=is_open,
            roi_size=roi_size,
            median_size=median_size,
            config=config
        )
        
        # Add trust score to candidate
        updated = {
            **candidate,
            'trust': trust_result.trust,
            'trust_result': trust_result
        }
        results.append(updated)
    
    return results


def select_top_k_by_trust(
    roi_candidates: List[Dict[str, Any]],
    k: int,
    config: Any
) -> List[Dict[str, Any]]:
    """
    Select top-K ROI candidates by trust score (quality-first selection).
    
    This ensures that the best quality ROIs are used for classification,
    regardless of how many total ROIs were collected.
    
    Args:
        roi_candidates: List of ROI candidate dicts
        k: Number of top candidates to select
        config: TrackingConfig object
        
    Returns:
        Top-K candidates sorted by trust (highest first)
    """
    # Compute trust scores if not already present
    candidates_with_trust = []
    for candidate in roi_candidates:
        if 'trust' not in candidate:
            # Need to compute trust
            sharpness = candidate.get('sharpness', 0.0)
            is_open = candidate.get('is_open', True) or candidate.get('state') == 'open'
            roi_size = candidate.get('size', (100, 100))
            
            trust_result = compute_roi_trust(
                sharpness=sharpness,
                is_open=is_open,
                roi_size=roi_size,
                median_size=None,  # Individual computation
                config=config
            )
            candidate = {**candidate, 'trust': trust_result.trust}
        
        candidates_with_trust.append(candidate)
    
    # Sort by trust descending
    sorted_candidates = sorted(
        candidates_with_trust,
        key=lambda x: x.get('trust', 0.0),
        reverse=True
    )
    
    # Return top K
    return sorted_candidates[:k]


def count_trusted_rois(
    roi_candidates: List[Dict[str, Any]],
    config: Any
) -> int:
    """
    Count number of ROIs that meet the minimum trust threshold.
    
    Used for stability gate to ensure sufficient high-quality evidence.
    
    Args:
        roi_candidates: List of ROI candidates with 'trust' scores
        config: TrackingConfig object
        
    Returns:
        Number of trusted ROIs
    """
    trust_min = getattr(config, 'trust_min_for_support', 0.4)
    
    count = 0
    for candidate in roi_candidates:
        trust = candidate.get('trust', 0.0)
        if trust >= trust_min:
            count += 1
    
    return count


def select_stratified_top_k(
    roi_candidates: List[Dict[str, Any]],
    top_k: int = 10,
    min_closed: int = 3,
    config: Any = None
) -> List[Dict[str, Any]]:
    """
    Select top K ROIs ensuring minimum closed representation.
    
    Strategy:
    1. Guarantee at least min_closed closed ROIs (if available)
    2. Fill remaining slots with best ROIs from both states by trust
    3. Prevents disambiguation failure from lack of closed ROIs
    
    This addresses the critical issue where top-K selection by trust alone
    may select only open ROIs if they have slightly higher sharpness, leaving
    zero closed ROIs for size-based disambiguation.
    
    Example:
        Given 10 open ROIs (trust 0.8-0.9) and 5 closed ROIs (trust 0.6-0.75):
        - Without stratification: All 10 selected would be open (higher trust)
        - With stratification: At least 3 closed + 7 best remaining = good mix
    
    Args:
        roi_candidates: List of ROI candidate dicts with 'trust' and 'state'/'is_open'
        top_k: Total number of ROIs to select
        min_closed: Minimum number of closed ROIs to guarantee (if available)
        config: TrackingConfig object (unused, kept for API compatibility)
        
    Returns:
        Top-K candidates with stratified selection ensuring min closed representation
        
    Raises:
        ValueError: If min_closed > top_k
    """
    if min_closed > top_k:
        raise ValueError(f"min_closed ({min_closed}) cannot exceed top_k ({top_k})")
    
    if not roi_candidates:
        return []
    
    # Ensure all candidates have trust scores
    candidates_with_trust = []
    for candidate in roi_candidates:
        if 'trust' not in candidate:
            # Compute trust if missing
            if config:
                sharpness = candidate.get('sharpness', 0.0)
                is_open = candidate.get('is_open', True) or candidate.get('state') == 'open'
                roi_size = candidate.get('size', (100, 100))
                
                trust_result = compute_roi_trust(
                    sharpness=sharpness,
                    is_open=is_open,
                    roi_size=roi_size,
                    median_size=None,
                    config=config
                )
                candidate = {**candidate, 'trust': trust_result.trust}
            else:
                # No config, default to 0.5
                candidate = {**candidate, 'trust': 0.5}
        
        candidates_with_trust.append(candidate)
    
    # Separate by state
    open_rois = []
    closed_rois = []
    
    for roi in candidates_with_trust:
        # Determine state from 'is_open' or 'state' field
        is_open = roi.get('is_open', None)
        state = roi.get('state', None)
        
        if is_open is not None:
            if is_open:
                open_rois.append(roi)
            else:
                closed_rois.append(roi)
        elif state is not None:
            if state == 'open':
                open_rois.append(roi)
            elif state == 'closed':
                closed_rois.append(roi)
            else:
                # Unknown state, treat as open by default
                open_rois.append(roi)
        else:
            # No state information, treat as open by default
            open_rois.append(roi)
    
    # Sort by trust descending
    open_sorted = sorted(open_rois, key=lambda x: x.get('trust', 0.0), reverse=True)
    closed_sorted = sorted(closed_rois, key=lambda x: x.get('trust', 0.0), reverse=True)
    
    # STEP 1: Ensure minimum closed representation
    # Take up to min_closed closed ROIs (or all if fewer available)
    selected_closed = closed_sorted[:min(min_closed, len(closed_sorted))]
    remaining_slots = top_k - len(selected_closed)
    
    # STEP 2: Fill remaining slots with best from both states
    # Combine remaining closed (after min_closed) with all open
    remaining_closed = closed_sorted[len(selected_closed):]
    remaining_pool = open_sorted + remaining_closed
    
    # Sort combined pool by trust
    remaining_sorted = sorted(remaining_pool, key=lambda x: x.get('trust', 0.0), reverse=True)
    
    # Select top remaining_slots from combined pool
    selected_remaining = remaining_sorted[:remaining_slots]
    
    # STEP 3: Combine and return
    final_selection = selected_closed + selected_remaining
    
    # Log stratification stats if we have logger
    if len(closed_sorted) > 0:
        actual_closed_selected = len([r for r in final_selection if not r.get('is_open', True) or r.get('state') == 'closed'])
        logger.debug(
            f"Stratified top-K selection: {len(final_selection)} total "
            f"({actual_closed_selected} closed, {len(final_selection) - actual_closed_selected} open) "
            f"from {len(candidates_with_trust)} candidates "
            f"({len(closed_sorted)} closed, {len(open_sorted)} open available)"
        )
    
    return final_selection
