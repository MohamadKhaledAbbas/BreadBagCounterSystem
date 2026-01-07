"""
Simplified Production-Grade Size-Based Disambiguation Module V2.

This module implements homography-first size-based disambiguation for visually similar
bread bag classes (e.g., Brown_Orange_Overlay vs Brown_Orange_Small).

## V8 Improvements: Homography-First Approach

1. **Homography-based classification (preferred)**:
   - Uses real-world measurements (cm²) when calibrated
   - Perspective-invariant and physically accurate
   - High confidence results

2. **Simple pixel fallback**:
   - Used when homography is not calibrated
   - Lower confidence tier to indicate less reliable measurement
   - Minimal penalty for gray zone cases

3. **Simplified gray zone handling**:
   - No complex strategies needed with homography
   - Simple midpoint-based resolution
   - Always flags as low confidence

4. **Streamlined confidence tiers**:
   - High: Homography-based clear classification
   - Low: Pixel fallback, or any gray zone case

## Usage

    from src.classifier.disambiguation_v2 import disambiguate_v2, DisambiguationV2Result
    
    result = disambiguate_v2(
        original_label="Brown_Orange_Overlay",
        confidence=0.65,
        bbox=(x1, y1, x2, y2),
        is_open=False,
        config=tracking_config,
        context={'track_id': 123, 'frame_index': 45}
    )
    
    # Result includes homography status
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence}")
    print(f"Tier: {result.confidence_tier}")
    print(f"Homography used: {result.metadata['homography_used']}")

Parameters are centralized in tracking_config.py and environment variables.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import math

from src.utils.AppLogging import logger

# V8: Import homography for real-world size measurement
try:
    from src.classifier.homography import get_homography_transform, classify_size_by_area_cm2
    HOMOGRAPHY_AVAILABLE = True
except ImportError:
    HOMOGRAPHY_AVAILABLE = False


@dataclass
class DisambiguationV2Result:
    """Enhanced result of size-based disambiguation with full diagnostics."""
    label: str
    confidence: float
    disambiguated: bool  # True if label was changed from original
    reason: str  # Human-readable explanation
    confidence_tier: str = 'high'  # 'high' or 'low' - flagged as 'low' for gray zone/ambiguous results
    
    # Enhanced metadata for monitoring and debugging
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'label': self.label,
            'confidence': self.confidence,
            'disambiguated': self.disambiguated,
            'reason': self.reason,
            'confidence_tier': self.confidence_tier,
            'metadata': self.metadata
        }


@dataclass
class ValidationResult:
    """Result of bbox validation checks."""
    valid: bool
    reason: Optional[str] = None
    penalty_applied: float = 0.0  # Confidence penalty (0.0 = no penalty, 1.0 = full penalty)
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_bbox(
    bbox: Tuple[float, float, float, float],
    config: Any,
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Validate bounding box geometry for suspicious or degenerate cases.
    
    Checks:
    1. Non-negative dimensions
    2. Reasonable aspect ratio (not too elongated or squished)
    3. Non-zero area
    4. Reasonable area (not unrealistically large/small given typical bag sizes)
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        config: Configuration object with validation thresholds
        context: Optional context for logging (track_id, frame_index, etc.)
        
    Returns:
        ValidationResult with validity status and penalty
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    metadata = {
        'bbox': bbox,
        'width': width,
        'height': height,
        'area': area
    }
    
    # Check 1: Non-negative dimensions
    if width <= 0 or height <= 0:
        return ValidationResult(
            valid=False,
            reason=f"degenerate_bbox (width={width:.1f}, height={height:.1f})",
            penalty_applied=1.0,  # Full penalty for degenerate bbox
            metadata=metadata
        )
    
    # Check 2: Aspect ratio validation
    min_aspect_ratio = getattr(config, 'disambiguation_v2_min_aspect_ratio', 0.3)
    max_aspect_ratio = getattr(config, 'disambiguation_v2_max_aspect_ratio', 3.0)
    
    aspect_ratio = width / height
    metadata['aspect_ratio'] = aspect_ratio
    
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        # Apply partial penalty for suspicious aspect ratio
        penalty = getattr(config, 'disambiguation_v2_aspect_ratio_penalty', 0.3)
        return ValidationResult(
            valid=True,  # Still proceed but with penalty
            reason=f"suspicious_aspect_ratio ({aspect_ratio:.2f} outside [{min_aspect_ratio:.2f}, {max_aspect_ratio:.2f}])",
            penalty_applied=penalty,
            metadata=metadata
        )
    
    # Check 3: Unrealistic area
    min_realistic_area = getattr(config, 'disambiguation_v2_min_realistic_area', 1000.0)
    max_realistic_area = getattr(config, 'disambiguation_v2_max_realistic_area', 100000.0)
    
    if area < min_realistic_area:
        penalty = getattr(config, 'disambiguation_v2_unrealistic_area_penalty', 0.5)
        return ValidationResult(
            valid=True,
            reason=f"unrealistically_small_area ({area:.0f} < {min_realistic_area:.0f})",
            penalty_applied=penalty,
            metadata=metadata
        )
    
    if area > max_realistic_area:
        penalty = getattr(config, 'disambiguation_v2_unrealistic_area_penalty', 0.5)
        return ValidationResult(
            valid=True,
            reason=f"unrealistically_large_area ({area:.0f} > {max_realistic_area:.0f})",
            penalty_applied=penalty,
            metadata=metadata
        )
    
    # All checks passed
    return ValidationResult(
        valid=True,
        reason=None,
        penalty_applied=0.0,
        metadata=metadata
    )

def compute_size_bin(
    raw_area: float,
    config: Any,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute size bin for the given raw area using multi-threshold logic.
    
    V8: Supports homography-based real-world measurements when calibrated.
    
    Bins:
    - 'very_small': area < very_small_threshold
    - 'small': very_small_threshold <= area < small_threshold
    - 'gray_zone': small_threshold <= area <= regular_threshold
    - 'regular': regular_threshold < area <= large_threshold
    - 'large': area > large_threshold
    
    Args:
        raw_area: Raw bounding box area in pixels²
        config: Configuration object with threshold values
        bbox: Optional bounding box for homography transformation
        
    Returns:
        Tuple of (bin_name, metadata_dict)
    """
    # V8: Check if homography is enabled and calibrated
    use_homography = getattr(config, 'homography_enabled', False) and HOMOGRAPHY_AVAILABLE
    area_cm2 = None
    size_cm = None
    
    if use_homography and bbox is not None:
        try:
            homography = get_homography_transform()
            if homography.is_calibrated():
                size_cm = homography.get_bbox_size_cm(bbox)
                area_cm2 = size_cm[0] * size_cm[1]
                
                # Use homography-based thresholds
                small_threshold_cm2 = getattr(config, 'homography_small_threshold_cm2', 100.0)
                large_threshold_cm2 = getattr(config, 'homography_large_threshold_cm2', 150.0)
                
                # Use cm² thresholds to determine bin
                metadata = {
                    'raw_area_px': raw_area,
                    'area_cm2': area_cm2,
                    'size_cm': size_cm,
                    'homography_used': True,
                    'thresholds_cm2': {
                        'small': small_threshold_cm2,
                        'large': large_threshold_cm2
                    }
                }
                
                # Classify using cm² thresholds
                size_class, size_bin = classify_size_by_area_cm2(
                    area_cm2, 
                    small_threshold_cm2,
                    large_threshold_cm2
                )
                
                # Map size_class to bin names
                if size_bin == 'very_small':
                    bin_name = 'very_small'
                elif size_bin == 'small':
                    bin_name = 'small'
                elif size_bin == 'medium':
                    bin_name = 'gray_zone'  # Medium = gray zone
                elif size_bin == 'large':
                    bin_name = 'regular'  # Large in cm = regular bag
                else:
                    bin_name = 'gray_zone'
                
                metadata['bin'] = bin_name
                metadata['size_class'] = size_class
                
                logger.debug(
                    f"[Disambiguation V2] Homography: area={area_cm2:.1f}cm², "
                    f"size={size_cm[0]:.1f}x{size_cm[1]:.1f}cm, bin={bin_name}"
                )
                
                return bin_name, metadata
                
        except Exception as e:
            logger.debug(f"[Disambiguation V2] Homography failed, using pixel area: {e}")
    
    # Fallback: Use pixel-based thresholds
    very_small_threshold = getattr(config, 'disambiguation_v2_very_small_threshold', 5000.0)
    small_threshold = getattr(config, 'disambiguation_small_threshold', 9000.0)
    regular_threshold = getattr(config, 'disambiguation_regular_threshold', 11000.0)
    large_threshold = getattr(config, 'disambiguation_v2_large_threshold', 25000.0)
    
    metadata = {
        'raw_area': raw_area,
        'homography_used': False,
        'thresholds': {
            'very_small': very_small_threshold,
            'small': small_threshold,
            'regular': regular_threshold,
            'large': large_threshold
        }
    }
    
    if raw_area < very_small_threshold:
        bin_name = 'very_small'
    elif raw_area < small_threshold:
        bin_name = 'small'
    elif raw_area <= regular_threshold:
        bin_name = 'gray_zone'
    elif raw_area <= large_threshold:
        bin_name = 'regular'
    else:
        bin_name = 'large'
    
    metadata['bin'] = bin_name
    
    return bin_name, metadata


def resolve_gray_zone(
    original_label: str,
    size_bin_metadata: Dict[str, Any],
    target_classes: Tuple[str, str],
    homography_used: bool = False
) -> Tuple[str, str]:
    """
    Resolve classification in gray zone with simplified logic.
    
    SIMPLIFIED: With homography, gray zones are rare and we just pick by midpoint.
    Without homography (pixel fallback), we use the same simple midpoint logic.
    
    Args:
        original_label: Original classifier prediction
        size_bin_metadata: Metadata from size bin computation
        target_classes: Tuple of (regular_class, small_class)
        homography_used: Whether homography was used for measurement
        
    Returns:
        Tuple of (resolved_label, reason)
    """
    regular_class, small_class = target_classes
    
    # Get area (either cm² or px²)
    if homography_used:
        area = size_bin_metadata.get('area_cm2', 0)
        thresholds = size_bin_metadata.get('thresholds_cm2', {})
        small_threshold = thresholds.get('small', 100.0)
        large_threshold = thresholds.get('large', 150.0)
        unit = 'cm²'
    else:
        area = size_bin_metadata.get('raw_area', 0)
        thresholds = size_bin_metadata.get('thresholds', {})
        small_threshold = thresholds.get('small', 9000.0)
        large_threshold = thresholds.get('regular', 11000.0)
        unit = 'px²'
    
    # Simple midpoint-based resolution
    midpoint = (small_threshold + large_threshold) / 2
    
    if area < midpoint:
        return small_class, f"gray_zone_resolved_to_small (area={area:.1f}{unit}, midpoint={midpoint:.1f})"
    else:
        return regular_class, f"gray_zone_resolved_to_regular (area={area:.1f}{unit}, midpoint={midpoint:.1f})"


def disambiguate_v2(
    original_label: str,
    confidence: float,
    bbox: Tuple[float, float, float, float],
    is_open: bool,
    config: Any,
    context: Optional[Dict[str, Any]] = None
) -> DisambiguationV2Result:
    """
    Simplified homography-first size-based disambiguation.
    
    SIMPLIFIED APPROACH:
    1. Check if enabled and not open state
    2. Check if target family member
    3. Use homography if calibrated (preferred), otherwise pixel fallback
    4. Classify based on thresholds
    5. Set confidence tier (low if pixel fallback or gray zone)
    6. Return result
    
    Args:
        original_label: Label predicted by classifier
        confidence: Confidence of prediction
        bbox: Bounding box (x1, y1, x2, y2)
        is_open: Whether ROI is in open state (True) or closed state (False)
        config: Configuration object with disambiguation parameters
        context: Optional context dict with track_id, frame_index, etc. for logging
        
    Returns:
        DisambiguationV2Result with final label and diagnostics
    """
    # Initialize metadata
    metadata = {
        'original_label': original_label,
        'original_confidence': confidence,
        'is_open': is_open,
        'bbox': bbox,
        'context': context or {}
    }
    
    # Step 1: Check if disambiguation is enabled
    if not getattr(config, 'disambiguation_v2_enabled', False):
        return DisambiguationV2Result(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="disambiguation_v2_disabled",
            metadata=metadata
        )
    
    # Step 2: Skip disambiguation for open state ROIs
    if is_open:
        metadata['skip_reason'] = 'open_state'
        return DisambiguationV2Result(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="skipped_open_state",
            metadata=metadata
        )
    
    # Step 3: Check if original label is in target family
    target_classes = getattr(config, 'disambiguation_classes', 
                             ('Brown_Orange_Overlay', 'Brown_Orange_Small'))
    regular_class, small_class = target_classes
    family_name = getattr(config, 'disambiguation_family_name', 'Brown_Orange_Family')
    
    is_family_member = (original_label in target_classes or original_label == family_name)
    
    if not is_family_member:
        metadata['skip_reason'] = 'not_target_family'
        return DisambiguationV2Result(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="not_target_family",
            metadata=metadata
        )
    
    # === FAMILY MEMBER DETECTED IN CLOSED STATE ===
    
    # Step 4: Compute size using homography (if available) or pixel fallback
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    raw_area = width * height
    
    size_bin, size_metadata = compute_size_bin(raw_area, config, bbox=bbox)
    metadata['size_bin'] = size_bin
    metadata['size_metadata'] = size_metadata
    
    homography_used = size_metadata.get('homography_used', False)
    
    # Step 5: Classify based on size bin
    final_label = original_label
    reason = ""
    confidence_tier = 'high'  # Default to high confidence
    
    if size_bin == 'very_small' or size_bin == 'small':
        # Clearly small
        final_label = small_class
        if homography_used:
            area_cm2 = size_metadata.get('area_cm2', 0)
            reason = f"homography_small (area={area_cm2:.1f}cm²)"
        else:
            reason = f"pixel_small (area={raw_area:.0f}px²)"
        confidence_tier = 'high' if homography_used else 'low'
    
    elif size_bin == 'regular' or size_bin == 'large':
        # Clearly regular/large
        final_label = regular_class
        if homography_used:
            area_cm2 = size_metadata.get('area_cm2', 0)
            reason = f"homography_regular (area={area_cm2:.1f}cm²)"
        else:
            reason = f"pixel_regular (area={raw_area:.0f}px²)"
        confidence_tier = 'high' if homography_used else 'low'
    
    elif size_bin == 'gray_zone':
        # Ambiguous size - resolve using simple midpoint logic
        final_label, reason = resolve_gray_zone(
            original_label, size_metadata, target_classes, homography_used
        )
        confidence_tier = 'low'  # Gray zone is always low confidence
    
    # Step 6: Apply minimal confidence penalty only for pixel fallback + gray zone
    final_confidence = confidence
    if not homography_used and size_bin == 'gray_zone':
        # Only apply penalty when using pixel fallback in gray zone
        penalty_factor = getattr(config, 'disambiguation_confidence_penalty', 0.9)
        final_confidence = confidence * penalty_factor
    
    metadata['final_label'] = final_label
    metadata['final_confidence'] = final_confidence
    metadata['confidence_tier'] = confidence_tier
    metadata['homography_used'] = homography_used
    
    # Debug logging if enabled
    debug_logging = getattr(config, 'disambiguation_v2_debug_logging', False)
    if debug_logging:
        context_str = ""
        if context:
            track_id = context.get('track_id', 'N/A')
            frame_index = context.get('frame_index', 'N/A')
            context_str = f"Track {track_id} Frame {frame_index}: "
        
        logger.info(
            f"[Disambiguation V2] {context_str}"
            f"original={original_label}(conf={confidence:.3f}), "
            f"final={final_label}(conf={final_confidence:.3f}), "
            f"size_bin={size_bin}, homography={homography_used}, "
            f"tier={confidence_tier}, reason={reason}"
        )
    
    return DisambiguationV2Result(
        label=final_label,
        confidence=final_confidence,
        disambiguated=True,
        reason=reason,
        confidence_tier=confidence_tier,
        metadata=metadata
    )


def disambiguate_batch_v2(
    classifications: list,
    config: Any,
    context: Optional[Dict[str, Any]] = None
) -> list:
    """
    Apply V2 disambiguation to a batch of classification results.
    
    This is useful when processing multiple ROIs from a track.
    
    Args:
        classifications: List of dicts with 'label', 'confidence', 'bbox', 'is_open' keys
        config: Configuration object
        context: Optional context dict for logging
        
    Returns:
        List of classifications with potentially updated labels and metadata
    """
    results = []
    
    for idx, clf in enumerate(classifications):
        original_label = clf.get('label', 'Unknown')
        confidence = clf.get('confidence', 0.0)
        bbox = clf.get('bbox')
        is_open = clf.get('is_open', True)  # Default to True for safety
        
        # Add index to context
        roi_context = {**(context or {}), 'roi_index': idx}
        
        if bbox is None:
            # No bbox available, keep original
            results.append({
                **clf,
                'disambiguation_v2': {
                    'applied': False,
                    'reason': 'no_bbox'
                }
            })
            continue
        
        # Apply V2 disambiguation
        result = disambiguate_v2(
            original_label=original_label,
            confidence=confidence,
            bbox=tuple(bbox),
            is_open=is_open,
            config=config,
            context=roi_context
        )
        
        # Update classification result
        updated_clf = {
            **clf,
            'label': result.label,
            'confidence': result.confidence,
            'disambiguation_v2': {
                'applied': result.disambiguated,
                'original_label': original_label,
                'reason': result.reason,
                'metadata': result.metadata
            }
        }
        
        results.append(updated_clf)
    
    return results
