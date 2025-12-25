"""
Production-Grade Size-Based Disambiguation Module V2.

This module implements enhanced family-based disambiguation for visually similar
bread bag classes (e.g., Brown_Orange_Overlay vs Brown_Orange_Small) using
production-tuned thresholds and robust fallback logic.

## Key Improvements Over V1:

1. **Multi-threshold logic with adjustable bins**:
   - Multiple size bins for more granular classification
   - Configurable thresholds for each bin
   - Gray zone handling with multiple strategies

2. **Aspect ratio and area validation**:
   - Validates ROI geometry for suspicious bboxes
   - Penalizes confidence for invalid aspect ratios
   - Detects and handles degenerate bboxes

3. **Gray zone handling strategies**:
   - 'keep_original': Trust classifier in ambiguous cases (default)
   - 'prefer_small': Bias toward small class
   - 'prefer_regular': Bias toward regular class
   - 'use_confidence': Use classifier confidence to break ties

4. **Detailed diagnostic logging**:
   - Before/after labels and confidence
   - Area, aspect ratio, size bin information
   - Resolution reason with full context
   - Confidence change tracking

5. **Configurable confidence penalty**:
   - Penalty when disambiguation changes the class
   - Minimal impact when classifier agrees with size
   - Separate penalty for validation failures

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
    
    # Result includes detailed metadata
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence}")
    print(f"Changed: {result.disambiguated}")
    print(f"Reason: {result.reason}")
    print(f"Metadata: {result.metadata}")

All parameters are centralized in tracking_config.py for easy tuning.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import math

from src.utils.AppLogging import logger


@dataclass
class DisambiguationV2Result:
    """Enhanced result of size-based disambiguation with full diagnostics."""
    label: str
    confidence: float
    disambiguated: bool  # True if label was changed from original
    reason: str  # Human-readable explanation
    
    # Enhanced metadata for monitoring and debugging
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'label': self.label,
            'confidence': self.confidence,
            'disambiguated': self.disambiguated,
            'reason': self.reason,
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
    config: Any
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute size bin for the given raw area using multi-threshold logic.
    
    Bins:
    - 'very_small': area < very_small_threshold
    - 'small': very_small_threshold <= area < small_threshold
    - 'gray_zone': small_threshold <= area <= regular_threshold
    - 'regular': regular_threshold < area <= large_threshold
    - 'large': area > large_threshold
    
    Args:
        raw_area: Raw bounding box area in pixels²
        config: Configuration object with threshold values
        
    Returns:
        Tuple of (bin_name, metadata_dict)
    """
    # Get thresholds from config
    very_small_threshold = getattr(config, 'disambiguation_v2_very_small_threshold', 5000.0)
    small_threshold = getattr(config, 'disambiguation_small_threshold', 9000.0)
    regular_threshold = getattr(config, 'disambiguation_regular_threshold', 11000.0)
    large_threshold = getattr(config, 'disambiguation_v2_large_threshold', 25000.0)
    
    metadata = {
        'raw_area': raw_area,
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
    confidence: float,
    size_bin_metadata: Dict[str, Any],
    config: Any,
    target_classes: Tuple[str, str]
) -> Tuple[str, str]:
    """
    Resolve classification in gray zone using configured strategy.
    
    Strategies:
    - 'keep_original': Trust classifier's prediction (default)
    - 'prefer_small': Bias toward small class
    - 'prefer_regular': Bias toward regular class
    - 'use_confidence': Use confidence to break ties (high conf keeps original, low conf → uncertain)
    
    Args:
        original_label: Original classifier prediction
        confidence: Prediction confidence
        size_bin_metadata: Metadata from size bin computation
        config: Configuration object
        target_classes: Tuple of (regular_class, small_class)
        
    Returns:
        Tuple of (resolved_label, reason)
    """
    gray_zone_behavior = getattr(config, 'disambiguation_gray_zone_behavior', 'keep_original')
    regular_class, small_class = target_classes
    raw_area = size_bin_metadata.get('raw_area', 0)
    
    if gray_zone_behavior == 'uncertain':
        return "Uncertain", f"gray_zone_uncertain (area={raw_area:.0f})"
    
    elif gray_zone_behavior == 'prefer_small':
        return small_class, f"gray_zone_prefer_small (area={raw_area:.0f})"
    
    elif gray_zone_behavior == 'prefer_regular':
        return regular_class, f"gray_zone_prefer_regular (area={raw_area:.0f})"
    
    elif gray_zone_behavior == 'use_confidence':
        # Use confidence threshold to decide
        confidence_threshold = getattr(config, 'disambiguation_v2_gray_zone_confidence_threshold', 0.6)
        if confidence >= confidence_threshold:
            return original_label, f"gray_zone_high_confidence (area={raw_area:.0f}, conf={confidence:.3f})"
        else:
            return "Uncertain", f"gray_zone_low_confidence (area={raw_area:.0f}, conf={confidence:.3f})"
    
    else:  # 'keep_original' or unknown strategy
        return original_label, f"gray_zone_keep_original (area={raw_area:.0f})"


def disambiguate_v2(
    original_label: str,
    confidence: float,
    bbox: Tuple[float, float, float, float],
    is_open: bool,
    config: Any,
    context: Optional[Dict[str, Any]] = None
) -> DisambiguationV2Result:
    """
    Enhanced size-based disambiguation with robust validation and detailed diagnostics.
    
    This function implements production-grade disambiguation with:
    - Multi-threshold size bins
    - Aspect ratio and area validation
    - Gray zone handling with multiple strategies
    - Detailed logging with before/after metadata
    - Configurable confidence penalties
    
    Args:
        original_label: Label predicted by classifier
        confidence: Confidence of prediction
        bbox: Bounding box (x1, y1, x2, y2)
        is_open: Whether ROI is in open state (True) or closed state (False)
        config: Configuration object with disambiguation parameters
        context: Optional context dict with track_id, frame_index, etc. for logging
        
    Returns:
        DisambiguationV2Result with final label and comprehensive diagnostics
    """
    # Initialize metadata
    metadata = {
        'original_label': original_label,
        'original_confidence': confidence,
        'is_open': is_open,
        'bbox': bbox,
        'context': context or {}
    }
    
    # Check if disambiguation is enabled
    if not getattr(config, 'disambiguation_v2_enabled', False):
        # Fall back to V1 if V2 is disabled
        return DisambiguationV2Result(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="disambiguation_v2_disabled",
            metadata=metadata
        )
    
    # CRITICAL: Skip disambiguation for open state ROIs
    if is_open:
        metadata['skip_reason'] = 'open_state'
        return DisambiguationV2Result(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason="skipped_open_state",
            metadata=metadata
        )
    
    # Get target classes (family members)
    target_classes = getattr(config, 'disambiguation_classes', 
                             ('Brown_Orange_Overlay', 'Brown_Orange_Small'))
    regular_class, small_class = target_classes
    family_name = getattr(config, 'disambiguation_family_name', 'Brown_Orange_Family')
    
    # Check if original label is in target family
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
    
    # Step 1: Validate bounding box
    validation_result = validate_bbox(bbox, config, context)
    metadata['validation'] = validation_result.metadata
    metadata['validation_valid'] = validation_result.valid
    metadata['validation_reason'] = validation_result.reason
    
    if not validation_result.valid:
        # Bbox is degenerate, skip disambiguation
        metadata['skip_reason'] = 'validation_failed'
        logger.warning(
            f"[Disambiguation V2] {context} Validation failed: {validation_result.reason}, "
            f"skipping disambiguation"
        )
        return DisambiguationV2Result(
            label=original_label,
            confidence=confidence,
            disambiguated=False,
            reason=f"validation_failed: {validation_result.reason}",
            metadata=metadata
        )
    
    # Apply validation penalty if any
    current_confidence = confidence
    if validation_result.penalty_applied > 0:
        current_confidence = confidence * (1.0 - validation_result.penalty_applied)
        metadata['validation_penalty'] = validation_result.penalty_applied
        metadata['confidence_after_validation'] = current_confidence
    
    # Step 2: Compute size bin
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    raw_area = width * height
    
    size_bin, size_metadata = compute_size_bin(raw_area, config)
    metadata['size_bin'] = size_bin
    metadata['size_metadata'] = size_metadata
    
    # Get thresholds for decision logic
    small_threshold = getattr(config, 'disambiguation_small_threshold', 9000.0)
    regular_threshold = getattr(config, 'disambiguation_regular_threshold', 11000.0)
    
    # Step 3: Make size-based decision
    final_label = original_label
    reason = ""
    
    if size_bin == 'very_small' or size_bin == 'small':
        # Clearly small
        final_label = small_class
        reason = f"family_size_{size_bin} (area={raw_area:.0f} < {small_threshold:.0f})"
    
    elif size_bin == 'regular' or size_bin == 'large':
        # Clearly regular/large
        final_label = regular_class
        reason = f"family_size_{size_bin} (area={raw_area:.0f} > {regular_threshold:.0f})"
    
    elif size_bin == 'gray_zone':
        # Ambiguous size, use gray zone strategy
        final_label, reason = resolve_gray_zone(
            original_label, current_confidence, size_metadata, config, target_classes
        )
    
    # Step 4: Apply confidence penalty if label changed
    label_changed = (final_label != original_label)
    metadata['label_changed'] = label_changed
    
    confidence_penalty = getattr(config, 'disambiguation_confidence_penalty', 0.9)
    penalty_on_change_only = getattr(config, 'disambiguation_penalty_on_change_only', False)
    
    should_apply_penalty = not penalty_on_change_only or label_changed
    
    if should_apply_penalty:
        final_confidence = current_confidence * confidence_penalty
        metadata['confidence_penalty_applied'] = True
        metadata['confidence_penalty_value'] = confidence_penalty
    else:
        final_confidence = current_confidence
        metadata['confidence_penalty_applied'] = False
    
    metadata['final_label'] = final_label
    metadata['final_confidence'] = final_confidence
    
    # Step 5: Debug logging if enabled
    debug_logging = getattr(config, 'disambiguation_v2_debug_logging', False)
    
    if debug_logging:
        context_str = ""
        if context:
            track_id = context.get('track_id', 'N/A')
            frame_index = context.get('frame_index', 'N/A')
            roi_index = context.get('roi_index', 'N/A')
            context_str = f"Track {track_id} Frame {frame_index} ROI {roi_index}: "
        
        logger.info(
            f"[Disambiguation V2] {context_str}"
            f"family={family_name}, "
            f"original={original_label}(conf={confidence:.3f}), "
            f"final={final_label}(conf={final_confidence:.3f}), "
            f"bbox={bbox}, area={raw_area:.0f}, size_bin={size_bin}, "
            f"validation={validation_result.valid}, "
            f"reason={reason}"
        )
    
    # Always mark as disambiguated for family members in closed state
    return DisambiguationV2Result(
        label=final_label,
        confidence=final_confidence,
        disambiguated=True,
        reason=reason,
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
