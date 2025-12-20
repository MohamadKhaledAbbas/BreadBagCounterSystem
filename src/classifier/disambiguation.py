"""
Size-Based Disambiguation Module for Visually Similar Class Families.

This module implements family-based disambiguation for visually similar
bread bag classes (e.g., Brown_Orange_Overlay vs Brown_Orange_Small) using
perspective-adjusted bounding box area analysis.

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

These classes differ primarily in physical size, and the camera's fixed 
viewpoint creates a predictable relationship between:
- Y position in frame (higher = farther = appears smaller)
- Apparent bbox area (needs perspective adjustment)

By computing perspective-adjusted area, we can reliably separate "small" vs
"regular" bags regardless of what the classifier initially predicted.

## Usage

    from src.classifier.disambiguation import disambiguate_by_size
    
    result = disambiguate_by_size(
        original_label="Brown_Orange_Overlay",  # or "Brown_Orange_Small" or "Brown_Orange_Family"
        confidence=0.65,
        bbox=(x1, y1, x2, y2),
        image_height=720,
        config=tracking_config
    )
    
    # For family members, result.label is ALWAYS decided by size
    # result.disambiguated = True for all family members
    final_label = result.label
    reason = result.reason  # e.g., "family_size_small (12500 < 15000)"

All parameters are centralized in tracking_config.py for easy tuning.
"""

import math
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
    raw_area: float  # Original bbox area in pixels^2
    adjusted_area: float  # Perspective-adjusted area
    y_norm: float  # Normalized Y position (0=top, 1=bottom)
    scale_factor: float  # Perspective scale factor applied


def compute_perspective_scale(
    y_norm: float,
    model: str = 'linear',
    a: float = 0.5,
    b: float = 1.0,
    p: float = 1.5
) -> float:
    """
    Compute perspective scale factor based on normalized Y position.
    
    The scale factor adjusts the apparent bbox area to account for
    perspective distortion. Objects higher in the frame (lower y_norm)
    appear smaller due to distance.
    
    Args:
        y_norm: Normalized Y position (0.0 = top of frame, 1.0 = bottom)
        model: Scaling model ('linear', 'power', or 'inverse')
        a: Scale coefficient a
        b: Scale coefficient b
        p: Power exponent (only for 'power' model)
        
    Returns:
        Scale factor to multiply with raw area
        
    Examples:
        - y_norm=0.0 (top/far): scale is smaller -> adjusted area is larger
        - y_norm=1.0 (bottom/near): scale is larger -> adjusted area is similar to raw
    """
    if model == 'linear':
        # Linear: scale = a + b * y_norm
        # At y_norm=0: scale = a
        # At y_norm=1: scale = a + b
        scale = a + b * y_norm
    
    elif model == 'power':
        # Power-law: scale = (a + b * y_norm) ** p
        # More aggressive scaling for perspective correction
        scale = (a + b * y_norm) ** p
    
    elif model == 'inverse':
        # Inverse: scale = 1 / (a + b * (1 - y_norm))
        # Stronger correction for objects far from camera
        denominator = a + b * (1.0 - y_norm)
        scale = 1.0 / max(denominator, 0.01)  # Avoid division by zero
    
    else:
        # Default to linear if unknown model
        scale = a + b * y_norm
    
    return max(scale, 0.01)  # Ensure positive scale


def compute_adjusted_area(
    bbox: Tuple[float, float, float, float],
    image_height: int,
    y_feature: str = 'cy',
    scaling_model: str = 'linear',
    scale_a: float = 0.5,
    scale_b: float = 1.0,
    scale_p: float = 1.5
) -> Tuple[float, float, float, float]:
    """
    Compute perspective-adjusted area for a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_height: Height of the image in pixels
        y_feature: Y coordinate feature to use ('cy' for center, 'y2' for bottom)
        scaling_model: Perspective scaling model
        scale_a, scale_b, scale_p: Scaling coefficients
        
    Returns:
        Tuple of (raw_area, adjusted_area, y_norm, scale_factor)
    """
    x1, y1, x2, y2 = bbox
    
    # Compute raw area
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    raw_area = width * height
    
    # Compute Y feature for perspective calculation
    if y_feature == 'y2':
        y_coord = y2  # Bottom of bbox
    else:
        y_coord = (y1 + y2) / 2.0  # Center of bbox
    
    # Normalize Y to [0, 1]
    y_norm = y_coord / max(image_height, 1)
    y_norm = max(0.0, min(1.0, y_norm))  # Clamp to [0, 1]
    
    # Compute perspective scale factor
    scale_factor = compute_perspective_scale(
        y_norm=y_norm,
        model=scaling_model,
        a=scale_a,
        b=scale_b,
        p=scale_p
    )
    
    # Compute adjusted area
    # Objects higher in frame (lower y_norm, lower scale) get multiplied
    # by a larger inverse factor to normalize their apparent size
    adjusted_area = raw_area * (1.0 / scale_factor)
    
    return raw_area, adjusted_area, y_norm, scale_factor


def disambiguate_by_size(
    original_label: str,
    confidence: float,
    bbox: Tuple[float, float, float, float],
    image_height: int,
    config: Any
) -> DisambiguationResult:
    """
    Disambiguate between visually similar classes using perspective-adjusted area.
    
    This function implements a "family-based" disambiguation approach:
    - When the classifier returns ANY member of a visually similar class family
      (e.g., Brown_Orange_Overlay or Brown_Orange_Small), we treat them as 
      "Brown_Orange_Family" and decide the specific class PURELY based on size.
    - This approach is future-proof: if the classifier is later trained to return
      "Brown_Orange_Family" directly, the same size-based logic will work.
    
    Args:
        original_label: Label predicted by the classifier
        confidence: Confidence of the prediction
        bbox: Bounding box (x1, y1, x2, y2)
        image_height: Height of the image in pixels
        config: TrackingConfig object with disambiguation parameters
        
    Returns:
        DisambiguationResult with final label and diagnostic info
        
    Decision Logic:
        1. If label not in target family classes -> return original unchanged
        2. If label IS in target family -> treat as "family" detection
        3. Compute perspective-adjusted area
        4. Decide final class PURELY based on size:
           - adjusted_area < small_threshold -> Small class
           - adjusted_area > regular_threshold -> Regular class  
           - gray zone -> apply gray_zone_behavior
        5. Always mark as disambiguated=True for family classes (size-decided)
    """
    # Check if disambiguation is enabled
    if not getattr(config, 'disambiguation_enabled', True):
        return DisambiguationResult(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="disambiguation_disabled",
            raw_area=0.0,
            adjusted_area=0.0,
            y_norm=0.0,
            scale_factor=1.0
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
            raw_area=0.0,
            adjusted_area=0.0,
            y_norm=0.0,
            scale_factor=1.0
        )
    
    # === FAMILY MEMBER DETECTED ===
    # From this point, we IGNORE the classifier's specific class prediction
    # and decide purely based on size measurement
    
    # Get configuration parameters
    y_feature = getattr(config, 'disambiguation_y_feature', 'cy')
    scaling_model = getattr(config, 'disambiguation_scaling_model', 'linear')
    scale_a = getattr(config, 'disambiguation_scale_a', 0.5)
    scale_b = getattr(config, 'disambiguation_scale_b', 1.0)
    scale_p = getattr(config, 'disambiguation_scale_p', 1.5)
    small_threshold = getattr(config, 'disambiguation_small_threshold', 15000.0)
    regular_threshold = getattr(config, 'disambiguation_regular_threshold', 25000.0)
    gray_zone_behavior = getattr(config, 'disambiguation_gray_zone_behavior', 'keep_original')
    debug_logging = getattr(config, 'disambiguation_debug_logging', False)
    
    # Compute adjusted area
    raw_area, adjusted_area, y_norm, scale_factor = compute_adjusted_area(
        bbox=bbox,
        image_height=image_height,
        y_feature=y_feature,
        scaling_model=scaling_model,
        scale_a=scale_a,
        scale_b=scale_b,
        scale_p=scale_p
    )
    
    # === SIZE-BASED DECISION (ignores classifier's specific class) ===
    # For family members, we ALWAYS use size to decide, so disambiguated=True
    disambiguated = True
    
    if adjusted_area < small_threshold:
        # Size indicates small class
        final_label = small_class
        reason = f"family_size_small ({adjusted_area:.0f} < {small_threshold:.0f})"
    
    elif adjusted_area > regular_threshold:
        # Size indicates regular class
        final_label = regular_class
        reason = f"family_size_regular ({adjusted_area:.0f} > {regular_threshold:.0f})"
    
    else:
        # Gray zone - apply configured behavior
        if gray_zone_behavior == 'uncertain':
            final_label = "Uncertain"
            reason = f"family_gray_zone_uncertain ({small_threshold:.0f} <= {adjusted_area:.0f} <= {regular_threshold:.0f})"
        
        elif gray_zone_behavior == 'prefer_small':
            final_label = small_class
            reason = f"family_gray_zone_prefer_small ({adjusted_area:.0f})"
        
        elif gray_zone_behavior == 'prefer_regular':
            final_label = regular_class
            reason = f"family_gray_zone_prefer_regular ({adjusted_area:.0f})"
        
        else:  # 'keep_original' - but since we're treating as family, default to regular
            # For family members in gray zone with 'keep_original', we need a decision
            # Default to regular class as the "baseline" family member
            final_label = regular_class
            reason = f"family_gray_zone_default_regular ({adjusted_area:.0f})"
    
    # Debug logging for tuning
    if debug_logging:
        logger.info(
            f"[Disambiguation] family={family_name}, classifier_said={original_label}, "
            f"size_decision={final_label}, bbox={bbox}, "
            f"raw_area={raw_area:.0f}, adjusted_area={adjusted_area:.0f}, "
            f"y_norm={y_norm:.3f}, scale={scale_factor:.3f}, "
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
        raw_area=raw_area,
        adjusted_area=adjusted_area,
        y_norm=y_norm,
        scale_factor=scale_factor
    )


def disambiguate_batch(
    classifications: list,
    image_height: int,
    config: Any
) -> list:
    """
    Apply disambiguation to a batch of classification results.
    
    This is useful when processing multiple ROIs from a track.
    
    Args:
        classifications: List of dicts with 'label', 'confidence', 'bbox' keys
        image_height: Height of the image in pixels
        config: TrackingConfig object
        
    Returns:
        List of classifications with potentially updated labels and metadata
    """
    results = []
    
    for clf in classifications:
        original_label = clf.get('label', 'Unknown')
        confidence = clf.get('confidence', 0.0)
        bbox = clf.get('bbox')
        
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
            image_height=image_height,
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
                'raw_area': result.raw_area,
                'adjusted_area': result.adjusted_area,
                'y_norm': result.y_norm,
                'scale_factor': result.scale_factor
            }
        }
        
        results.append(updated_clf)
    
    return results
