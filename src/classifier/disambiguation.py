"""
Size-Based Disambiguation Module for Visually Similar Class Families.

This module implements family-based disambiguation for visually similar
bread bag classes (e.g., Brown_Orange_Overlay vs Brown_Orange_Small) using
raw bounding box area analysis on CLOSED state ROIs only.

## Family-Based Approach

When the classifier returns ANY member of a visually similar class family
(e.g., Brown_Orange_Overlay or Brown_Orange_Small), we treat the detection
as belonging to a "family" (Brown_Orange_Family) and decide the SPECIFIC
class purely based on size measurement.

This approach is future-proof:
- If the classifier is later retrained to return "Brown_Orange_Family" directly,
  the same size-based logic will work seamlessly.
- Size measurement is the primary discriminant, not the classifier's guess.

## Key Insight

These classes differ primarily in physical size. By using raw bounding box
area (in pixels²) on CLOSED state ROIs, we can reliably separate "small" vs
"regular" bags regardless of what the classifier initially predicted.

CLOSED state ROIs are used exclusively because:
- Open bags are inflated and have distorted sizes
- Closed bags have consistent, reliable dimensions
- This avoids false disambiguation from temporary size variations

## Usage

    from src.classifier.disambiguation import disambiguate_by_size
    
    result = disambiguate_by_size(
        original_label="Brown_Orange_Overlay",  # or "Brown_Orange_Small" or "Brown_Orange_Family"
        confidence=0.65,
        bbox=(x1, y1, x2, y2),
        is_open=False,  # Only disambiguate closed ROIs
        config=tracking_config
    )
    
    # For family members in closed state, result.label is ALWAYS decided by size
    # result.disambiguated = True for all family members in closed state
    final_label = result.label
    reason = result.reason  # e.g., "family_size_small (12500 < 15000)"

All parameters are centralized in tracking_config.py for easy tuning.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils.AppLogging import logger


@dataclass
class DisambiguationResult:
    """Result of size-based disambiguation."""
    label: str
    confidence: float
    disambiguated: bool  # True if label was changed from original
    reason: str  # Human-readable explanation
    raw_area: float  # Bbox area in pixels^2



def disambiguate_by_size(
    original_label: str,
    confidence: float,
    bbox: Tuple[float, float, float, float],
    is_open: bool,
    config: Any
) -> DisambiguationResult:
    """
    Disambiguate between visually similar classes using raw bounding box area.
    
    This function implements a "family-based" disambiguation approach:
    - When the classifier returns ANY member of a visually similar class family
      (e.g., Brown_Orange_Overlay or Brown_Orange_Small), we treat them as 
      "Brown_Orange_Family" and decide the specific class PURELY based on size.
    - This approach is future-proof: if the classifier is later trained to return
      "Brown_Orange_Family" directly, the same size-based logic will work.
    
    IMPORTANT: Disambiguation is ONLY applied to CLOSED state ROIs. Open state
    ROIs are skipped because their inflated size would lead to incorrect classification.
    
    Args:
        original_label: Label predicted by the classifier
        confidence: Confidence of the prediction
        bbox: Bounding box (x1, y1, x2, y2)
        is_open: Whether the ROI is in open state (True) or closed state (False)
        config: TrackingConfig object with disambiguation parameters
        
    Returns:
        DisambiguationResult with final label and diagnostic info
        
    Decision Logic:
        1. If is_open=True -> return original unchanged (skip open ROIs)
        2. If label not in target family classes -> return original unchanged
        3. If label IS in target family -> treat as "family" detection
        4. Compute raw bbox area
        5. Decide final class PURELY based on size:
           - raw_area < small_threshold -> Small class
           - raw_area > regular_threshold -> Regular class  
           - gray zone -> apply gray_zone_behavior
        6. Always mark as disambiguated=True for family classes (size-decided)
    """
    # Check if disambiguation is enabled
    if not getattr(config, 'disambiguation_enabled', True):
        return DisambiguationResult(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="disambiguation_disabled",
            raw_area=0.0
        )
    
    # CRITICAL: Skip disambiguation for open state ROIs
    if is_open:
        return DisambiguationResult(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="skipped_open_state",
            raw_area=0.0
        )
    
    # Get disambiguation target classes (family members)
    target_classes = getattr(config, 'disambiguation_classes', 
                             ('Brown_Orange_Overlay', 'Brown_Orange_Small'))
    regular_class, small_class = target_classes
    
    # Get family name for logging/future use (defaults to combined name)
    family_name = getattr(config, 'disambiguation_family_name', 'Brown_Orange_Family')
    
    # Check if original label is a member of the target family
    # This also supports future case where classifier returns "Brown_Orange_Family" directly
    is_family_member = (original_label in target_classes or original_label == family_name)
    
    if not is_family_member:
        return DisambiguationResult(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="not_target_family",
            raw_area=0.0
        )
    
    # === FAMILY MEMBER DETECTED IN CLOSED STATE ===
    # From this point, we IGNORE the classifier's specific class prediction
    # and decide purely based on size measurement
    
    # Get configuration parameters
    small_threshold = getattr(config, 'disambiguation_small_threshold', 10000.0)
    regular_threshold = getattr(config, 'disambiguation_regular_threshold', 20000.0)
    gray_zone_behavior = getattr(config, 'disambiguation_gray_zone_behavior', 'keep_original')
    debug_logging = getattr(config, 'disambiguation_debug_logging', False)
    
    # Compute raw area
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    raw_area = width * height

    logger.debug(f"raw area calculation: width={width}, height={height}, raw_area={raw_area}")

    # === SIZE-BASED DECISION (ignores classifier's specific class) ===
    # For family members in closed state, we ALWAYS use size to decide, so disambiguated=True
    disambiguated = True
    
    if raw_area < small_threshold:
        # Size indicates small class
        final_label = small_class
        reason = f"family_size_small ({raw_area:.0f} < {small_threshold:.0f})"
    
    elif raw_area > regular_threshold:
        # Size indicates regular class
        final_label = regular_class
        reason = f"family_size_regular ({raw_area:.0f} > {regular_threshold:.0f})"
    
    else:
        # Gray zone - apply configured behavior
        if gray_zone_behavior == 'uncertain':
            final_label = "Uncertain"
            reason = f"family_gray_zone_uncertain ({small_threshold:.0f} <= {raw_area:.0f} <= {regular_threshold:.0f})"
        
        elif gray_zone_behavior == 'prefer_small':
            final_label = small_class
            reason = f"family_gray_zone_prefer_small ({raw_area:.0f})"
        
        elif gray_zone_behavior == 'prefer_regular':
            final_label = regular_class
            reason = f"family_gray_zone_prefer_regular ({raw_area:.0f})"
        
        else:  # 'keep_original' - but since we're treating as family, default to regular
            # For family members in gray zone with 'keep_original', we need a decision
            # Default to regular class as the "baseline" family member
            final_label = original_label
            reason = f"family_gray_zone_default_regular ({raw_area:.0f})"
    
    # Debug logging for tuning
    if debug_logging:
        logger.info(
            f"[Disambiguation] family={family_name}, classifier_said={original_label}, "
            f"size_decision={final_label}, bbox={bbox}, "
            f"raw_area={raw_area:.0f}, "
            f"thresholds=({small_threshold:.0f}, {regular_threshold:.0f}), "
            f"reason={reason}"
        )
    
    # Confidence handling for family-based disambiguation
    # Option 1: Always apply penalty (conservative - indicates size-based override)
    # Option 2: Only apply when classifier's guess differs from size-based decision
    confidence_penalty = getattr(config, 'disambiguation_confidence_penalty', 0.9)
    apply_penalty_only_on_change = getattr(config, 'disambiguation_penalty_on_change_only', False)
    
    classifier_was_correct = (original_label == final_label)
    should_apply_penalty = not apply_penalty_only_on_change or not classifier_was_correct
    
    final_confidence = confidence * confidence_penalty if should_apply_penalty else confidence
    
    return DisambiguationResult(
        label=final_label,
        confidence=final_confidence,
        disambiguated=disambiguated,
        reason=reason,
        raw_area=raw_area
    )


def disambiguate_batch(
    classifications: list,
    config: Any
) -> list:
    """
    Apply disambiguation to a batch of classification results.
    
    This is useful when processing multiple ROIs from a track.
    
    Args:
        classifications: List of dicts with 'label', 'confidence', 'bbox', 'is_open' keys
        config: TrackingConfig object
        
    Returns:
        List of classifications with potentially updated labels and metadata
    """
    results = []
    
    for clf in classifications:
        original_label = clf.get('label', 'Unknown')
        confidence = clf.get('confidence', 0.0)
        bbox = clf.get('bbox')
        is_open = clf.get('is_open', True)  # Default to True for safety
        
        if bbox is None:
            # No bbox available, keep original
            results.append({
                **clf,
                'disambiguation': {
                    'applied': False,
                    'reason': 'no_bbox'
                }
            })
            continue
        
        # Apply disambiguation
        result = disambiguate_by_size(
            original_label=original_label,
            confidence=confidence,
            bbox=tuple(bbox),
            is_open=is_open,
            config=config
        )
        
        # Update classification result
        updated_clf = {
            **clf,
            'label': result.label,
            'confidence': result.confidence,
            'disambiguation': {
                'applied': result.disambiguated,
                'original_label': original_label,
                'reason': result.reason,
                'raw_area': result.raw_area
            }
        }
        
        results.append(updated_clf)
    
    return results
