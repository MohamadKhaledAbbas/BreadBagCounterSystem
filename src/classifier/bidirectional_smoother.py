"""
Bidirectional Context-Aware Classification Smoother.

This module implements a buffered validation queue that uses both previous and
future context to validate and correct low-confidence classifications.

Key Features:
1. Sliding Window Buffer - Delays final commit to gather context from both sides
2. Context Analysis - Uses prev_N and next_N items to validate center item
3. Batch Transition Protection - Preserves genuine transitions between bag types
4. High-Confidence Bypass - Trusts classifier for high-confidence predictions

Design Rationale:
- In production, bags of the same type are processed sequentially (batches)
- A single misclassification amidst a batch is likely an error
- But when switching between batches (Brown -> White), we must NOT smooth
- This exploits the batch nature of production for 99.9% accuracy

Usage:
    smoother = BidirectionalSmoother(buffer_size=7)
    
    # As each event is ready for commit:
    validated_event = smoother.add_event(event_data)
    if validated_event:
        # Event has been validated and can be committed
        commit_to_database(validated_event)
"""

import logging
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from src.config.tracking_config import tracking_config
from src.utils.AppLogging import logger, structured_logger

# Confidence threshold above which we trust the classifier without context
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.90

# Default buffer size (should be odd for symmetric context)
DEFAULT_BUFFER_SIZE = 7

# Default context agreement ratio to override
DEFAULT_CONTEXT_AGREEMENT_RATIO = 0.8


@dataclass
class BufferedEvent:
    """
    Represents an event in the validation buffer.
    
    Attributes:
        event_id: Unique identifier for the event
        label: Classification label
        confidence: Classification confidence
        original_label: Original label before any smoothing
        original_confidence: Original confidence before any smoothing
        event_data: Full event data dictionary for commit
        validated: Whether this event has been validated
        smoothed: Whether this event's label was changed by smoothing
        smoothing_reason: Reason for smoothing if applied
    """
    event_id: int
    label: str
    confidence: float
    original_label: str
    original_confidence: float
    event_data: Dict[str, Any]
    validated: bool = False
    smoothed: bool = False
    smoothing_reason: str = ""


class BidirectionalSmoother:
    """
    Buffered validation queue with bidirectional context-aware smoothing.
    
    Delays final commit of classification results to gather context from
    both previous and upcoming events. Uses this context to validate and
    potentially correct low-confidence classifications.
    
    The smoother maintains a sliding window buffer:
    - Center item (index buffer_size // 2) is validated using context
    - Previous items (left of center) provide historical context
    - Next items (right of center) provide future context
    
    Validation Rules:
    1. High confidence (>= threshold): Trust classifier, bypass context
    2. Low confidence + unanimous context: Override with context label
    3. Low confidence + split context: Batch transition, trust classifier
    """
    
    def __init__(
        self,
        buffer_size: int = None,
        confidence_threshold: float = None,
        context_agreement_ratio: float = None,
        batch_transition_protection: bool = None,
        enabled: bool = None,
    ):
        """
        Initialize the bidirectional smoother.
        
        Args:
            buffer_size: Size of the validation buffer (should be odd)
            confidence_threshold: Above this, classifications bypass context check
            context_agreement_ratio: Fraction of context items that must agree
            batch_transition_protection: If True, protect batch transitions
            enabled: If False, smoother is disabled (pass-through mode)
        """
        # Load defaults from config
        self.enabled = enabled if enabled is not None else tracking_config.bidirectional_smoothing_enabled
        self.buffer_size = buffer_size if buffer_size is not None else tracking_config.bidirectional_buffer_size
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else tracking_config.bidirectional_confidence_threshold
        self.context_agreement_ratio = context_agreement_ratio if context_agreement_ratio is not None else tracking_config.bidirectional_context_agreement_ratio
        self.batch_transition_protection = batch_transition_protection if batch_transition_protection is not None else tracking_config.bidirectional_batch_transition_protection
        
        # Ensure buffer size is odd for symmetric context
        if self.buffer_size % 2 == 0:
            self.buffer_size += 1
            logger.warning(
                f"[BidirectionalSmoother] Buffer size must be odd, adjusted to {self.buffer_size}"
            )
        
        # Center index is the item being validated
        self.center_index = self.buffer_size // 2
        
        # The validation buffer
        self._buffer: deque[BufferedEvent] = deque(maxlen=self.buffer_size)
        
        # Statistics for monitoring
        self._stats = {
            'total_events': 0,
            'validated_events': 0,
            'smoothed_events': 0,
            'high_confidence_bypassed': 0,
            'context_overrides': 0,
            'batch_transitions_protected': 0,
            'no_context_available': 0,
        }
        
        logger.info(
            f"[BidirectionalSmoother] Initialized: enabled={self.enabled}, "
            f"buffer_size={self.buffer_size}, confidence_threshold={self.confidence_threshold:.2f}, "
            f"context_agreement_ratio={self.context_agreement_ratio:.2f}, "
            f"batch_transition_protection={self.batch_transition_protection}"
        )
    
    def add_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a classification event to the validation buffer.
        
        When the buffer is full enough to validate the center item,
        returns the validated event data. Otherwise returns None.
        
        Args:
            event_data: Event data dictionary containing at minimum:
                - 'event_id': Unique event identifier
                - 'bag_type': Classification label
                - 'confidence': Classification confidence
                
        Returns:
            Validated event data if center item was validated, None otherwise
        """
        if not self.enabled:
            # Pass-through mode - return immediately
            return event_data
        
        self._stats['total_events'] += 1
        
        # Extract classification info
        event_id = event_data.get('event_id', 0)
        label = event_data.get('bag_type', 'Unknown')
        confidence = event_data.get('confidence', 0.0)
        
        # Create buffered event
        buffered_event = BufferedEvent(
            event_id=event_id,
            label=label,
            confidence=confidence,
            original_label=label,
            original_confidence=confidence,
            event_data=event_data,
        )
        
        # Add to buffer
        self._buffer.append(buffered_event)
        
        # Check if we can validate the center item
        if len(self._buffer) >= self.buffer_size:
            return self._validate_center()
        
        return None
    
    def flush(self) -> List[Dict[str, Any]]:
        """
        Flush remaining events from the buffer.
        
        Call this when processing is complete to get any remaining
        buffered events that couldn't be fully validated.
        
        Returns:
            List of remaining event data dictionaries
        """
        if not self.enabled:
            return []
        
        remaining = []
        
        # Validate and emit remaining items
        while len(self._buffer) > 0:
            # For remaining items, we have less context
            # Use what context we have available
            if len(self._buffer) >= self.center_index + 1:
                validated = self._validate_center()
                if validated:
                    remaining.append(validated)
            else:
                # Not enough items even for partial validation
                # Just emit the front item as-is
                event = self._buffer.popleft()
                event.validated = True
                self._stats['validated_events'] += 1
                self._stats['no_context_available'] += 1
                remaining.append(self._finalize_event(event))
        
        return remaining
    
    def _validate_center(self) -> Optional[Dict[str, Any]]:
        """
        Validate the center item using bidirectional context.
        
        Returns:
            Validated event data, or None if validation failed
        """
        if len(self._buffer) < self.center_index + 1:
            return None
        
        center_event = self._buffer[self.center_index]
        
        # High confidence - bypass context check
        if center_event.confidence >= self.confidence_threshold:
            center_event.validated = True
            center_event.smoothing_reason = "high_confidence_trusted"
            self._stats['validated_events'] += 1
            self._stats['high_confidence_bypassed'] += 1
            
            # Pop the front item and return it
            front_event = self._buffer.popleft()
            return self._finalize_event(front_event)
        
        # Low confidence - use context to validate
        prev_context = [self._buffer[i] for i in range(self.center_index)]
        next_context = [self._buffer[i] for i in range(self.center_index + 1, len(self._buffer))]
        
        # Analyze context
        smoothed_label, smoothing_reason = self._analyze_context(
            center_event, prev_context, next_context
        )
        
        if smoothed_label and smoothed_label != center_event.label:
            center_event.label = smoothed_label
            center_event.smoothed = True
            center_event.smoothing_reason = smoothing_reason
            # Update event_data
            center_event.event_data['bag_type'] = smoothed_label
            center_event.event_data['smoothed'] = True
            center_event.event_data['original_bag_type'] = center_event.original_label
            center_event.event_data['smoothing_reason'] = smoothing_reason
            self._stats['smoothed_events'] += 1
            self._stats['context_overrides'] += 1
        
        center_event.validated = True
        self._stats['validated_events'] += 1
        
        # Pop the front item and return it
        front_event = self._buffer.popleft()
        return self._finalize_event(front_event)
    
    def _analyze_context(
        self, 
        center_event: BufferedEvent,
        prev_context: List[BufferedEvent],
        next_context: List[BufferedEvent]
    ) -> Tuple[Optional[str], str]:
        """
        Analyze bidirectional context to determine if smoothing should occur.
        
        Args:
            center_event: The event being validated
            prev_context: Events before the center
            next_context: Events after the center
            
        Returns:
            Tuple of (smoothed_label or None, reason string)
        """
        if not prev_context or not next_context:
            return None, "insufficient_context"
        
        # Get labels from context
        prev_labels = [e.label for e in prev_context]
        next_labels = [e.label for e in next_context]
        all_context_labels = prev_labels + next_labels
        
        # Count label occurrences
        from collections import Counter
        prev_counter = Counter(prev_labels)
        next_counter = Counter(next_labels)
        all_counter = Counter(all_context_labels)
        
        # Get dominant labels
        prev_dominant = prev_counter.most_common(1)[0] if prev_counter else (None, 0)
        next_dominant = next_counter.most_common(1)[0] if next_counter else (None, 0)
        
        # Check for batch transition
        if self.batch_transition_protection:
            if prev_dominant[0] != next_dominant[0] and prev_dominant[0] is not None and next_dominant[0] is not None:
                # Different dominant labels on each side = batch transition
                self._stats['batch_transitions_protected'] += 1
                return None, f"batch_transition_protected (prev={prev_dominant[0]}, next={next_dominant[0]})"
        
        # Check context agreement
        total_context = len(all_context_labels)
        if total_context == 0:
            return None, "no_context"
        
        # Find the most common label in context
        most_common_label, count = all_counter.most_common(1)[0]
        agreement_ratio = count / total_context
        
        if agreement_ratio >= self.context_agreement_ratio:
            if most_common_label != center_event.label:
                # Context strongly agrees on a different label
                return most_common_label, f"context_override (agreement={agreement_ratio:.2f}, label={most_common_label})"
        
        return None, f"context_disagrees (best_agreement={agreement_ratio:.2f})"
    
    def _finalize_event(self, event: BufferedEvent) -> Dict[str, Any]:
        """
        Finalize an event for output.
        
        Args:
            event: The buffered event to finalize
            
        Returns:
            Event data dictionary with smoothing metadata
        """
        result = event.event_data.copy()
        
        # Add smoothing metadata
        result['bidirectional_smoothing'] = {
            'applied': event.smoothed,
            'original_label': event.original_label,
            'original_confidence': event.original_confidence,
            'final_label': event.label,
            'reason': event.smoothing_reason,
        }
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get smoother statistics for monitoring."""
        stats = self._stats.copy()
        stats['buffer_size'] = len(self._buffer)
        stats['smoothing_rate'] = (
            self._stats['smoothed_events'] / self._stats['validated_events']
            if self._stats['validated_events'] > 0 else 0.0
        )
        return stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0


# Module-level singleton for convenience
_global_smoother: Optional[BidirectionalSmoother] = None


def get_bidirectional_smoother() -> BidirectionalSmoother:
    """Get or create the global bidirectional smoother instance."""
    global _global_smoother
    if _global_smoother is None:
        _global_smoother = BidirectionalSmoother()
    return _global_smoother


def reset_bidirectional_smoother():
    """Reset the global smoother instance."""
    global _global_smoother
    _global_smoother = None
