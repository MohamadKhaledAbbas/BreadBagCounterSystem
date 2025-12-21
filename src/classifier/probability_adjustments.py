"""
Probability Mass Transfer Module for Disambiguation.

This module provides a reusable mechanism to apply disambiguation decisions
to probability vectors by transferring probability mass between sibling classes
in a "family" (e.g., Overlay <-> Small) while preserving probabilities for
unrelated classes.

Key Principles:
1. Disambiguation is a decision made outside of classification (e.g., by size)
2. When disambiguation flips the label, we need to adjust the probability vector
   to reflect this decision for evidence accumulation
3. Probability mass is transferred between the two sibling classes only
4. All other class probabilities remain unchanged
5. Final vector is normalized and validated

Algorithm:
    Given:
    - Original probs: {Overlay: 0.6, Small: 0.3, White: 0.05, Bran: 0.05}
    - Disambiguation: Overlay -> Small
    
    Transfer:
    - Take all mass from Overlay (0.6) and Small (0.3) = 0.9 total
    - Give all to Small: {Small: 0.9, Overlay: 0.0, White: 0.05, Bran: 0.05}
    - Or use configurable transfer ratio
    
    Normalize:
    - Ensure sum = 1.0 (handle floating point errors)

Usage:
    from src.classifier.probability_adjustments import apply_probability_adjustment
    
    adjusted_probs, metadata = apply_probability_adjustment(
        original_probs={"Overlay": 0.6, "Small": 0.3, "White": 0.1},
        from_label="Overlay",
        to_label="Small",
        family_classes=["Overlay", "Small"],
        config=tracking_config
    )
    
    # adjusted_probs: {"Overlay": 0.0, "Small": 0.9, "White": 0.1}
    # metadata: {applied: True, from_label: "Overlay", ...}

All parameters are centralized in tracking_config.py for easy tuning.
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from src.utils.AppLogging import logger


@dataclass
class ProbabilityAdjustmentResult:
    """Result of probability mass transfer."""
    adjusted_probs: Dict[str, float]  # Final probability vector after adjustment
    applied: bool  # Whether adjustment was applied
    from_label: Optional[str]  # Source class (if applied)
    to_label: Optional[str]  # Target class (if applied)
    mass_transferred: float  # Amount of probability mass moved
    before_from: float  # Probability of from_label before
    before_to: float  # Probability of to_label before
    after_from: float  # Probability of from_label after
    after_to: float  # Probability of to_label after
    normalization_applied: bool  # Whether renormalization was needed
    reason: str  # Explanation of what happened


def apply_probability_adjustment(
    original_probs: Dict[str, float],
    from_label: Optional[str],
    to_label: Optional[str],
    family_classes: Optional[List[str]] = None,
    config: Optional[Any] = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Apply probability mass transfer based on disambiguation decision.
    
    When disambiguation changes a label (e.g., Overlay -> Small), we adjust
    the probability vector by transferring mass from the source class to the
    target class within the family, keeping other classes unchanged.
    
    Args:
        original_probs: Original probability vector {class: prob}
        from_label: Original class label (before disambiguation)
        to_label: Final class label (after disambiguation)
        family_classes: List of family member classes (e.g., ["Overlay", "Small"])
                       If None, uses [from_label, to_label]
        config: TrackingConfig object with adjustment parameters
        
    Returns:
        Tuple of (adjusted_probs, metadata_dict):
        - adjusted_probs: New probability vector with mass transferred
        - metadata_dict: Details about the adjustment for logging
    """
    # Default metadata for no-adjustment cases
    no_adjustment_metadata = {
        "applied": False,
        "from_label": from_label,
        "to_label": to_label,
        "mass_transferred": 0.0,
        "before_from": original_probs.get(from_label, 0.0) if from_label else 0.0,
        "before_to": original_probs.get(to_label, 0.0) if to_label else 0.0,
        "after_from": original_probs.get(from_label, 0.0) if from_label else 0.0,
        "after_to": original_probs.get(to_label, 0.0) if to_label else 0.0,
        "normalization_applied": False,
        "reason": "no_change_needed"
    }
    
    # Check if adjustment is needed
    if from_label is None or to_label is None:
        no_adjustment_metadata["reason"] = "missing_labels"
        return original_probs.copy(), no_adjustment_metadata
    
    if from_label == to_label:
        no_adjustment_metadata["reason"] = "no_label_change"
        return original_probs.copy(), no_adjustment_metadata
    
    # Check if both labels exist in probability vector
    if from_label not in original_probs:
        no_adjustment_metadata["reason"] = f"from_label_not_in_probs ({from_label})"
        return original_probs.copy(), no_adjustment_metadata
    
    if to_label not in original_probs:
        no_adjustment_metadata["reason"] = f"to_label_not_in_probs ({to_label})"
        return original_probs.copy(), no_adjustment_metadata
    
    # Determine family classes
    if family_classes is None:
        family_classes = [from_label, to_label]
    
    # Get configuration parameters
    transfer_strategy = getattr(config, 'prob_adjustment_strategy', 'full_transfer') if config else 'full_transfer'
    transfer_ratio = getattr(config, 'prob_adjustment_transfer_ratio', 1.0) if config else 1.0
    epsilon = getattr(config, 'prob_adjustment_epsilon', 1e-9) if config else 1e-9
    debug_logging = getattr(config, 'prob_adjustment_debug_logging', False) if config else False
    
    # Record before values
    before_from = original_probs[from_label]
    before_to = original_probs[to_label]
    
    # Create adjusted probability vector (copy original first)
    adjusted_probs = original_probs.copy()
    
    # Apply transfer strategy
    if transfer_strategy == 'full_transfer':
        # Strategy 1: Transfer ALL family mass to target class
        # Sum all probability mass from family members
        family_mass = sum(adjusted_probs.get(cls, 0.0) for cls in family_classes)
        
        # Give all family mass to target class
        adjusted_probs[to_label] = family_mass
        
        # Zero out other family members
        for cls in family_classes:
            if cls != to_label:
                adjusted_probs[cls] = epsilon  # Use epsilon instead of 0 for numerical stability
        
        mass_transferred = family_mass - before_to
        reason = f"full_transfer_to_{to_label}"
    
    elif transfer_strategy == 'proportional_transfer':
        # Strategy 2: Transfer from source only, proportionally
        # Calculate amount to transfer (ratio * from_label's probability)
        transfer_amount = before_from * transfer_ratio
        
        # Transfer from source to target
        adjusted_probs[from_label] = max(epsilon, before_from - transfer_amount)
        adjusted_probs[to_label] = before_to + transfer_amount
        
        mass_transferred = transfer_amount
        reason = f"proportional_transfer (ratio={transfer_ratio})"
    
    elif transfer_strategy == 'swap':
        # Strategy 3: Swap probabilities between from and to
        adjusted_probs[from_label] = before_to
        adjusted_probs[to_label] = before_from
        
        mass_transferred = abs(before_from - before_to)
        reason = "probability_swap"
    
    else:
        # Unknown strategy - no adjustment
        no_adjustment_metadata["reason"] = f"unknown_strategy ({transfer_strategy})"
        return original_probs.copy(), no_adjustment_metadata
    
    # Normalize to ensure sum = 1.0 (handle floating point errors)
    probs_sum = sum(adjusted_probs.values())
    normalization_applied = False
    
    if abs(probs_sum - 1.0) > 1e-6:  # Need normalization
        if probs_sum > epsilon:
            adjusted_probs = {k: v / probs_sum for k, v in adjusted_probs.items()}
            normalization_applied = True
        else:
            # Edge case: all probabilities near zero - reset to uniform
            num_classes = len(adjusted_probs)
            adjusted_probs = {k: 1.0 / num_classes for k in adjusted_probs.keys()}
            normalization_applied = True
            reason += "_uniform_fallback"
    
    # Record after values
    after_from = adjusted_probs[from_label]
    after_to = adjusted_probs[to_label]
    
    # Build metadata
    metadata = {
        "applied": True,
        "from_label": from_label,
        "to_label": to_label,
        "mass_transferred": mass_transferred,
        "before_from": before_from,
        "before_to": before_to,
        "after_from": after_from,
        "after_to": after_to,
        "normalization_applied": normalization_applied,
        "reason": reason,
        "strategy": transfer_strategy,
        "family_classes": family_classes
    }
    
    # Debug logging
    if debug_logging:
        logger.info(
            f"[ProbAdjustment] {from_label} -> {to_label}: "
            f"before=({before_from:.3f}, {before_to:.3f}), "
            f"after=({after_from:.3f}, {after_to:.3f}), "
            f"transferred={mass_transferred:.3f}, strategy={transfer_strategy}"
        )
    
    return adjusted_probs, metadata


def apply_batch_adjustments(
    classifications: List[Dict[str, Any]],
    family_classes: Optional[List[str]] = None,
    config: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Apply probability adjustments to a batch of classifications.
    
    This is useful when processing multiple ROIs from a track where
    each may have been disambiguated.
    
    Args:
        classifications: List of classification dicts with keys:
            - 'probs': Original probability vector
            - 'label': Final label (after disambiguation)
            - 'original_label': Label before disambiguation (optional)
            - 'disambiguated': Whether disambiguation was applied
        family_classes: Family member classes for adjustment
        config: TrackingConfig object
        
    Returns:
        List of classifications with updated 'probs' and 'prob_adjustment' metadata
    """
    results = []
    
    for clf in classifications:
        probs = clf.get('probs', {})
        label = clf.get('label', 'Unknown')
        original_label = clf.get('original_label')
        disambiguated = clf.get('disambiguated', False)
        
        # Only apply adjustment if disambiguation changed the label
        if not disambiguated or original_label is None or original_label == label:
            # No adjustment needed
            results.append({
                **clf,
                'prob_adjustment': {
                    'applied': False,
                    'reason': 'no_disambiguation' if not disambiguated else 'no_label_change'
                }
            })
            continue
        
        # Apply probability adjustment
        adjusted_probs, metadata = apply_probability_adjustment(
            original_probs=probs,
            from_label=original_label,
            to_label=label,
            family_classes=family_classes,
            config=config
        )
        
        # Update classification with adjusted probs
        results.append({
            **clf,
            'probs': adjusted_probs,  # Replace with adjusted probs
            'original_probs': probs,  # Keep original for reference
            'prob_adjustment': metadata
        })
    
    return results


def validate_probability_vector(
    probs: Dict[str, float],
    epsilon: float = 1e-6
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a probability vector is well-formed.
    
    Checks:
    1. All values are non-negative
    2. Sum is approximately 1.0 (within epsilon)
    3. No NaN or Inf values
    
    Args:
        probs: Probability vector to validate
        epsilon: Tolerance for sum check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not probs:
        return False, "empty_probability_vector"
    
    # Check for non-numeric values
    for class_name, prob in probs.items():
        if not isinstance(prob, (int, float)):
            return False, f"non_numeric_probability ({class_name}: {type(prob)})"
        
        if math.isnan(prob):
            return False, f"nan_probability ({class_name})"
        
        if math.isinf(prob):
            return False, f"inf_probability ({class_name})"
        
        if prob < 0:
            return False, f"negative_probability ({class_name}: {prob})"
    
    # Check sum
    total = sum(probs.values())
    if abs(total - 1.0) > epsilon:
        return False, f"sum_not_one (sum={total:.6f})"
    
    return True, None
