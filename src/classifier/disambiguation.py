"""
Size-Based Disambiguation Module for Visually Similar Classes.

This module implements post-detection disambiguation between visually similar
bread bag classes (e.g., Brown_Orange_Overlay vs Brown_Orange_Small) using
perspective-adjusted bounding box area analysis.

The key insight is that these classes differ primarily in physical size, and
the camera's fixed viewpoint creates a predictable relationship between:
- Y position in frame (higher = farther = appears smaller)
- Apparent bbox area (needs perspective adjustment)

By computing perspective-adjusted area, we can reliably separate "small" vs
"regular" bags even when the classifier is uncertain.

Usage:
    from src.classifier.disambiguation import disambiguate_by_size
    
    result = disambiguate_by_size(
        original_label="Brown_Orange_Overlay",
        confidence=0.65,
        bbox=(x1, y1, x2, y2),
        image_height=720,
        config=tracking_config
    )
    
    if result['disambiguated']:
        final_label = result['label']
        reason = result['reason']

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
    
    This function checks if the original_label is one of the disambiguation
    target classes, and if so, uses size-based logic to potentially override
    the classifier's prediction.
    
    Args:
        original_label: Label predicted by the classifier
        confidence: Confidence of the prediction
        bbox: Bounding box (x1, y1, x2, y2)
        image_height: Height of the image in pixels
        config: TrackingConfig object with disambiguation parameters
        
    Returns:
        DisambiguationResult with final label and diagnostic info
        
    Decision Logic:
        1. If label not in target classes -> return original unchanged
        2. Compute perspective-adjusted area
        3. If adjusted_area < small_threshold -> force to small class
        4. If adjusted_area > regular_threshold -> force to regular class
        5. Else (gray zone) -> apply gray_zone_behavior
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
    
    # Get disambiguation target classes
    target_classes = getattr(config, 'disambiguation_classes', 
                             ('Brown_Orange_Overlay', 'Brown_Orange_Small'))
    regular_class, small_class = target_classes
    
    # Check if original label is a target class
    if original_label not in target_classes:
        return DisambiguationResult(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="not_target_class",
            raw_area=0.0,
            adjusted_area=0.0,
            y_norm=0.0,
            scale_factor=1.0
        )
    
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
    
    # Apply decision rules
    final_label = original_label
    disambiguated = False
    reason = "unchanged"
    
    if adjusted_area < small_threshold:
        # Force to small class
        final_label = small_class
        disambiguated = (original_label != small_class)
        reason = f"area_below_small_threshold ({adjusted_area:.0f} < {small_threshold:.0f})"
    
    elif adjusted_area > regular_threshold:
        # Force to regular class
        final_label = regular_class
        disambiguated = (original_label != regular_class)
        reason = f"area_above_regular_threshold ({adjusted_area:.0f} > {regular_threshold:.0f})"
    
    else:
        # Gray zone - apply configured behavior
        if gray_zone_behavior == 'uncertain':
            final_label = "Uncertain"
            disambiguated = True
            reason = f"gray_zone_uncertain ({small_threshold:.0f} <= {adjusted_area:.0f} <= {regular_threshold:.0f})"
        
        elif gray_zone_behavior == 'prefer_small':
            final_label = small_class
            disambiguated = (original_label != small_class)
            reason = f"gray_zone_prefer_small ({adjusted_area:.0f})"
        
        elif gray_zone_behavior == 'prefer_regular':
            final_label = regular_class
            disambiguated = (original_label != regular_class)
            reason = f"gray_zone_prefer_regular ({adjusted_area:.0f})"
        
        else:  # 'keep_original' or unknown
            reason = f"gray_zone_keep_original ({adjusted_area:.0f})"
    
    # Debug logging for tuning
    if debug_logging:
        logger.info(
            f"[Disambiguation] original={original_label}, final={final_label}, "
            f"disambiguated={disambiguated}, bbox={bbox}, "
            f"raw_area={raw_area:.0f}, adjusted_area={adjusted_area:.0f}, "
            f"y_norm={y_norm:.3f}, scale={scale_factor:.3f}, "
            f"thresholds=({small_threshold:.0f}, {regular_threshold:.0f}), "
            f"reason={reason}"
        )
    
    return DisambiguationResult(
        label=final_label,
        confidence=confidence if not disambiguated else confidence * 0.9,  # Slight penalty for override
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
