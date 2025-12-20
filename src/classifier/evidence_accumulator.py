"""
Evidence Accumulator Module for Track-Level Classification.

This module implements trust-weighted log-evidence accumulation for making
robust, noise-resistant classification decisions across multiple ROIs.

Key Principles:
1. No single ROI can dominate the final decision
2. Low-quality ROIs cannot outweigh high-quality ones
3. Class switching within a track is penalized
4. Forced decisions under ambiguity are avoided (returns "Uncertain")

Algorithm Overview:
    For each class c, accumulate weighted log evidence:
        Score(c) = Σᵢ wᵢ × log(pᵢ(c) + ε)
    
    Where:
    - wᵢ is trust score for ROI i
    - pᵢ(c) is probability of class c from classifier
    - ε prevents log(0)

Log-evidence provides mathematical containment:
- Prevents single-frame dominance
- Rewards repeated consistent evidence
- Suppresses noisy ambiguous frames
- Contains overconfident misclassifications

Usage:
    from src.classifier.evidence_accumulator import EvidenceAccumulator
    
    accumulator = EvidenceAccumulator(config)
    
    for roi, probs, trust in track_rois:
        accumulator.update(roi_id, probs, trust)
    
    result = accumulator.finalize()
    if result.is_certain:
        final_class = result.label
    else:
        # Handle uncertainty

All parameters are centralized in tracking_config.py for easy tuning.
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.utils.AppLogging import logger


@dataclass
class EvidenceState:
    """Internal state for a single class's evidence."""
    log_evidence: float = 0.0
    roi_count: int = 0
    best_confidence: float = 0.0
    best_roi_id: Optional[int] = None
    contributions: List[float] = field(default_factory=list)


@dataclass
class FinalClassificationResult:
    """Result of evidence accumulation and finalization."""
    label: str  # Final class label or "Uncertain"
    confidence: float  # Best confidence from winning class
    is_certain: bool  # True if passed stability gate
    
    # Evidence details
    winner_score: float  # Log-evidence score for winner
    runner_up_label: Optional[str]  # Second-best class
    runner_up_score: float  # Log-evidence score for runner-up
    margin: float  # Winner - runner_up score
    
    # Stability gate details
    gate_passed: bool  # Did result pass stability gate?
    gate_failure_reason: Optional[str]  # Why gate failed (if applicable)
    
    # Diagnostic info
    rois_used: int  # Number of ROIs used
    rois_trusted: int  # Number of trusted ROIs
    trust_stats: Dict[str, float]  # min/max/mean trust
    evidence_per_class: Dict[str, float]  # Log-evidence for each class
    class_switch_penalty_applied: bool  # Was inertia/penalty applied?
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'label': self.label,
            'confidence': self.confidence,
            'is_certain': self.is_certain,
            'winner_score': self.winner_score,
            'runner_up_label': self.runner_up_label,
            'runner_up_score': self.runner_up_score,
            'margin': self.margin,
            'gate_passed': self.gate_passed,
            'gate_failure_reason': self.gate_failure_reason,
            'rois_used': self.rois_used,
            'rois_trusted': self.rois_trusted,
            'trust_stats': self.trust_stats,
            'evidence_per_class': self.evidence_per_class,
            'class_switch_penalty_applied': self.class_switch_penalty_applied
        }


class EvidenceAccumulator:
    """
    Accumulates trust-weighted log-evidence for track-level classification.
    
    This class maintains evidence for a single track and provides methods
    to update evidence as new ROIs are classified, and to finalize the
    classification when the track is complete.
    
    Example:
        accumulator = EvidenceAccumulator(config)
        
        for roi_data in track_rois:
            # probs is full probability vector {class_name: prob}
            accumulator.update(
                roi_id=roi_data['id'],
                probs=roi_data['probs'],
                trust=roi_data['trust'],
                state=roi_data['state']
            )
        
        result = accumulator.finalize()
    """
    
    def __init__(self, config: Any):
        """
        Initialize the evidence accumulator.
        
        Args:
            config: TrackingConfig object with evidence parameters
        """
        self.config = config
        
        # Evidence state per class
        self._evidence: Dict[str, EvidenceState] = defaultdict(EvidenceState)
        
        # Tracking for temporal consistency
        self._leading_class: Optional[str] = None
        self._inertia_strength: float = 0.0
        self._roi_count: int = 0
        
        # Trust tracking
        self._trust_values: List[float] = []
        self._trusted_count: int = 0
        
        # Configuration
        self._epsilon = getattr(config, 'evidence_epsilon', 1e-6)
        self._inertia_enabled = getattr(config, 'temporal_inertia_enabled', True)
        self._inertia_base = getattr(config, 'temporal_inertia_strength', 0.15)
        self._inertia_decay = getattr(config, 'temporal_inertia_decay', 0.8)
        self._stability_gate = getattr(config, 'stability_gate_enabled', True)
        self._margin_threshold = getattr(config, 'stability_margin_threshold', 0.5)
        self._min_trusted = getattr(config, 'stability_min_trusted_rois', 2)
        self._trust_min = getattr(config, 'trust_min_for_support', 0.4)
    
    def update(
        self,
        roi_id: int,
        probs: Dict[str, float],
        trust: float,
        state: str = 'open'
    ) -> None:
        """
        Update evidence with a new ROI classification.
        
        Args:
            roi_id: Identifier for this ROI
            probs: Full probability vector {class_name: probability}
            trust: Trust score for this ROI (0-1)
            state: 'open' or 'closed' state of the ROI
        """
        self._roi_count += 1
        self._trust_values.append(trust)
        
        if trust >= self._trust_min:
            self._trusted_count += 1
        
        # Update evidence for each class
        for class_name, prob in probs.items():
            # Compute weighted log evidence
            log_prob = math.log(prob + self._epsilon)
            weighted_contribution = trust * log_prob
            
            # Accumulate
            self._evidence[class_name].log_evidence += weighted_contribution
            self._evidence[class_name].roi_count += 1
            self._evidence[class_name].contributions.append(weighted_contribution)
            
            # Track best confidence for this class
            if prob > self._evidence[class_name].best_confidence:
                self._evidence[class_name].best_confidence = prob
                self._evidence[class_name].best_roi_id = roi_id
        
        # Update temporal consistency tracking
        self._update_leading_class()
    
    def _update_leading_class(self) -> None:
        """Update the leading class and manage inertia."""
        if not self._evidence:
            return
        
        # Find current leader
        sorted_classes = sorted(
            self._evidence.items(),
            key=lambda x: x[1].log_evidence,
            reverse=True
        )
        
        current_leader = sorted_classes[0][0] if sorted_classes else None
        
        if self._leading_class is None:
            # First ROI - set leader
            self._leading_class = current_leader
            self._inertia_strength = self._inertia_base if self._inertia_enabled else 0.0
        
        elif current_leader != self._leading_class:
            # Class switch detected - decay inertia
            self._inertia_strength *= self._inertia_decay
            
            # Check if switch is sustained
            if self._evidence[current_leader].log_evidence > \
               self._evidence[self._leading_class].log_evidence + self._inertia_strength:
                # Sustained evidence for new class - allow switch
                self._leading_class = current_leader
                self._inertia_strength = self._inertia_base
        
        else:
            # Same leader - maintain inertia (slight decay to prevent lock-in)
            self._inertia_strength = min(
                self._inertia_strength * 0.95 + self._inertia_base * 0.05,
                self._inertia_base * 1.5
            )
    
    def finalize(self) -> FinalClassificationResult:
        """
        Finalize classification using accumulated evidence.
        
        Applies:
        1. Class-switch penalty (if enabled)
        2. Stability gate (if enabled)
        3. Winner determination
        
        Returns:
            FinalClassificationResult with label and diagnostics
        """
        # Handle empty evidence
        if not self._evidence:
            return FinalClassificationResult(
                label="Uncertain",
                confidence=0.0,
                is_certain=False,
                winner_score=0.0,
                runner_up_label=None,
                runner_up_score=0.0,
                margin=0.0,
                gate_passed=False,
                gate_failure_reason="no_evidence",
                rois_used=0,
                rois_trusted=0,
                trust_stats={'min': 0.0, 'max': 0.0, 'mean': 0.0},
                evidence_per_class={},
                class_switch_penalty_applied=False
            )
        
        # Apply class-switch penalty to non-leading classes (if enabled)
        penalty_applied = False
        if self._inertia_enabled and self._leading_class and self._inertia_strength > 0:
            for class_name in self._evidence:
                if class_name != self._leading_class:
                    # Apply penalty to challengers
                    self._evidence[class_name].log_evidence -= self._inertia_strength
                    penalty_applied = True
        
        # Sort by evidence score
        sorted_classes = sorted(
            self._evidence.items(),
            key=lambda x: x[1].log_evidence,
            reverse=True
        )
        
        winner_name = sorted_classes[0][0]
        winner_state = sorted_classes[0][1]
        
        runner_up_name = sorted_classes[1][0] if len(sorted_classes) > 1 else None
        runner_up_state = sorted_classes[1][1] if len(sorted_classes) > 1 else None
        runner_up_score = runner_up_state.log_evidence if runner_up_state else float('-inf')
        
        margin = winner_state.log_evidence - runner_up_score
        
        # Compute trust statistics
        trust_stats = self._compute_trust_stats()
        
        # Build evidence per class dict
        evidence_per_class = {
            name: state.log_evidence 
            for name, state in self._evidence.items()
        }
        
        # Apply stability gate
        gate_passed = True
        gate_failure_reason = None
        
        if self._stability_gate:
            # Check margin threshold
            if margin < self._margin_threshold:
                gate_passed = False
                gate_failure_reason = f"margin_too_small ({margin:.3f} < {self._margin_threshold})"
            
            # Check trusted ROI count
            elif self._trusted_count < self._min_trusted:
                gate_passed = False
                gate_failure_reason = f"too_few_trusted_rois ({self._trusted_count} < {self._min_trusted})"
        
        # Determine final label
        if gate_passed:
            final_label = winner_name
            is_certain = True
        else:
            final_label = "Uncertain"
            is_certain = False
        
        return FinalClassificationResult(
            label=final_label,
            confidence=winner_state.best_confidence,
            is_certain=is_certain,
            winner_score=winner_state.log_evidence,
            runner_up_label=runner_up_name,
            runner_up_score=runner_up_score,
            margin=margin,
            gate_passed=gate_passed,
            gate_failure_reason=gate_failure_reason,
            rois_used=self._roi_count,
            rois_trusted=self._trusted_count,
            trust_stats=trust_stats,
            evidence_per_class=evidence_per_class,
            class_switch_penalty_applied=penalty_applied
        )
    
    def _compute_trust_stats(self) -> Dict[str, float]:
        """Compute min/max/mean trust statistics."""
        if not self._trust_values:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0}
        
        return {
            'min': min(self._trust_values),
            'max': max(self._trust_values),
            'mean': sum(self._trust_values) / len(self._trust_values)
        }
    
    def get_current_leader(self) -> Optional[str]:
        """Get the current leading class (before finalization)."""
        if not self._evidence:
            return None
        
        sorted_classes = sorted(
            self._evidence.items(),
            key=lambda x: x[1].log_evidence,
            reverse=True
        )
        return sorted_classes[0][0] if sorted_classes else None
    
    def get_evidence_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of evidence for all classes."""
        summary = {}
        for class_name, state in self._evidence.items():
            summary[class_name] = {
                'log_evidence': state.log_evidence,
                'roi_count': state.roi_count,
                'best_confidence': state.best_confidence,
                'mean_contribution': (
                    sum(state.contributions) / len(state.contributions)
                    if state.contributions else 0.0
                )
            }
        return summary


def accumulate_track_evidence(
    classifications: List[Dict[str, Any]],
    config: Any
) -> FinalClassificationResult:
    """
    Convenience function to accumulate evidence for a complete track.
    
    Args:
        classifications: List of classification dicts with:
            - 'probs': Full probability vector {class: prob}
            - 'trust': Trust score
            - 'state': 'open' or 'closed'
        config: TrackingConfig object
        
    Returns:
        FinalClassificationResult with track-level classification
    """
    accumulator = EvidenceAccumulator(config)
    
    for i, clf in enumerate(classifications):
        probs = clf.get('probs', {})
        trust = clf.get('trust', 0.5)
        state = clf.get('state', 'open')
        
        accumulator.update(
            roi_id=i,
            probs=probs,
            trust=trust,
            state=state
        )
    
    return accumulator.finalize()
