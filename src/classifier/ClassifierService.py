"""
ClassifierService Module V4 - Evidence-Based Classification.

Production-grade classification system that:
- Runs classifier only on top-K candidate ROIs (selected by sharpness)
- Accumulates evidence across candidates using weighted scoring
- Uses winner vs runner-up ratio for final decision
- Provides explainable logging for every classification

Key Changes from V2:
- REMOVED: Entropy-based filtering
- REMOVED: EMA smoothing
- REMOVED: Dirichlet voting
- ADDED: Evidence accumulation with sharpness and temporal weights
- ADDED: Winner/runner-up ratio threshold
- ADDED: Structural "Unknown" definition (not statistical uncertainty)
"""

import logging
import os
import time
from collections import defaultdict
from typing import Callable, List, Dict, Any, Tuple, Optional

import cv2

from src.classifier.BaseClassifier import BaseClassifier
from src.config.tracking_config import tracking_config
from src.utils.Utils import compute_phash
from src.utils.AppLogging import logger, structured_logger
from src.utils.PipelineMetrics import pipeline_metrics

ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    """
    Evidence-based classifier service for production-grade accuracy.
    
    V4 Design Principles:
    1. Classification is a scarce resource - spend only on best frames
    2. One track → one final decision (classification at track end only)
    3. Evidence accumulation replaces statistical voting
    4. Unknown = structural issue, not statistical uncertainty
    
    Evidence Scoring:
        evidence_score[label] += confidence × sharpness_weight × temporal_weight
    
    Final Decision:
        - Accept winner if: winner_score >= MIN_TOTAL_SCORE AND
                          winner_score / runner_up_score >= RATIO_THRESHOLD
        - Otherwise: Unknown
    """
    
    def __init__(self,
                 classifier: BaseClassifier,
                 data_root: str = "data",
                 save_all_rois: bool = False,
                 min_confidence_threshold: float = 0.3):
        """
        Initialize the evidence-based classifier service.
        
        Args:
            classifier: Base classifier model for predictions
            data_root: Root directory for saving ROI images
            save_all_rois: Whether to save all classified ROIs
            min_confidence_threshold: Minimum confidence for individual predictions
        """
        self.classifier = classifier
        self.data_root = data_root
        self.save_all_rois = save_all_rois
        self.min_confidence_threshold = min_confidence_threshold
        
        # V4: Configuration from centralized config
        self.top_k = tracking_config.top_k_candidates
        self.min_total_evidence = tracking_config.min_total_evidence_score
        self.ratio_threshold = tracking_config.evidence_ratio_threshold
        self.min_candidates = tracking_config.min_candidates_for_classification
        self.min_track_frames = tracking_config.min_track_frames
        self.sharpness_scale = tracking_config.sharpness_weight_scale
        self.temporal_scale = tracking_config.temporal_weight_scale
        self.max_single_weight = tracking_config.max_single_roi_weight

        self.callbacks: List[ResultCallback] = []
        self.running = True
        
        # V4: Track classification statistics
        self._total_classified = 0
        self._unknown_structural = 0
        self._unknown_low_evidence = 0
        self._unknown_ambiguous = 0
        
        # V5: Classification smoothing with history (for sequential bags of same type)
        self.history_size = 5  # Last N bag classifications to maintain
        self.history_vote_threshold = 3  # K out of N required for stable vote
        self.high_conf_threshold = tracking_config.high_confidence_threshold
        self._recent_classifications: List[Tuple[str, float]] = []  # (label, confidence) for recent bags
        
        # V6: Production-grade classification stability heuristics
        self.enable_label_reuse = tracking_config.enable_label_reuse
        self.low_conf_threshold = tracking_config.low_conf_threshold
        self.streak_min_length = tracking_config.streak_min_length
        self.burst_dominance_min_ratio = tracking_config.burst_dominance_min_ratio
        self.burst_window_size = tracking_config.burst_window_size
        self.track_volatility_threshold = tracking_config.track_volatility_threshold
        self.enable_volatility_logging = tracking_config.enable_volatility_logging
        
        # Track per-track label history for volatility analysis
        self._track_label_history: Dict[int, List[Tuple[str, float]]] = defaultdict(list)  # track_id -> [(label, conf)]

        logger.info(
            f"[ClassifierService] Initialized V4 (Evidence-Based) with V5 Classification Smoothing: "
            f"top_k={self.top_k}, min_evidence={self.min_total_evidence}, "
            f"ratio_threshold={self.ratio_threshold}, min_candidates={self.min_candidates}, "
            f"history_size={self.history_size}, history_vote_threshold={self.history_vote_threshold}"
        )

    def register_callback(self, callback: ResultCallback):
        """Register a callback for classification results."""
        self.callbacks.append(callback)

    def _classify_single(self, roi_image, idx: int = 0) -> Tuple[str, float]:
        """
        Classify a single ROI image.
        
        Args:
            roi_image: ROI image to classify
            idx: Candidate index for logging
            
        Returns:
            Tuple of (label, confidence)
        """
        try:
            label, conf = self.classifier.predict(roi_image)
            return label, float(conf)
        except Exception as e:
            structured_logger.pipeline_error(
                component="ClassifierService",
                operation="single_roi_classification",
                error_type=type(e).__name__,
                error_message=str(e),
                affected_ids=[idx],
                context={"candidate_idx": idx}
            )
            return "Unknown", 0.0
    
    def _apply_classification_smoothing(self, label: str, confidence: float) -> Tuple[str, float, Optional[str]]:
        """
        Apply classification smoothing using recent bag classification history.
        
        Exploits the fact that bags are often provided in sequences of the same type.
        
        Decision rules:
        1. If current confidence >= high_conf_threshold, use current label
        2. Else if history has stable vote (>= K out of N agree on same label), use that label
        3. Else use current label
        
        Args:
            label: Current classification label
            confidence: Current classification confidence
            
        Returns:
            Tuple of (final_label, final_confidence, reason):
            - final_label: Label after smoothing
            - final_confidence: Confidence after smoothing
            - reason: Reason for decision ('high_conf', 'history_vote', or 'current')
        """
        # Decision Rule 1: High confidence - use current, no smoothing needed
        if confidence >= self.high_conf_threshold:
            # Still add to history for future low-confidence classifications
            self._recent_classifications.append((label, confidence))
            if len(self._recent_classifications) > self.history_size:
                self._recent_classifications.pop(0)
            return label, confidence, 'high_conf'
        
        # Decision Rule 2: Check if history has stable vote
        if len(self._recent_classifications) >= self.history_vote_threshold:
            # Count votes for each label (from recent history)
            label_votes = defaultdict(int)
            label_confidences = defaultdict(list)
            
            for hist_label, hist_conf in self._recent_classifications:
                label_votes[hist_label] += 1
                label_confidences[hist_label].append(hist_conf)
            
            # Find most voted label
            max_votes = 0
            winning_label = None
            for lbl, votes in label_votes.items():
                if votes > max_votes:
                    max_votes = votes
                    winning_label = lbl
            
            # Check if winning label has stable vote (K out of N)
            if max_votes >= self.history_vote_threshold and winning_label != label:
                # Use winning label with average confidence from history
                avg_confidence = sum(label_confidences[winning_label]) / len(label_confidences[winning_label])
                
                # Log history vote usage
                structured_logger.classification_history_vote(
                    track_id=-1,  # Not track-specific, global history
                    current_label=label,
                    current_confidence=confidence,
                    history_label=winning_label,
                    history_confidence=avg_confidence,
                    vote_count=max_votes,
                    history_size=len(self._recent_classifications),
                    history_buffer=[(lbl, conf) for lbl, conf in self._recent_classifications]
                )
                
                logger.info(
                    f"[ClassifierService] Using history vote: {winning_label} "
                    f"({max_votes}/{len(self._recent_classifications)} votes, avg_conf={avg_confidence:.2f}) "
                    f"instead of current low-confidence: {label} (conf={confidence:.2f})"
                )
                
                # Add history winner to recent classifications (for continuity)
                self._recent_classifications.append((winning_label, avg_confidence))
                if len(self._recent_classifications) > self.history_size:
                    self._recent_classifications.pop(0)
                
                return winning_label, avg_confidence, 'history_vote'
        
        # Decision Rule 3: Use current label (not enough history or no stable vote)
        # Add current to history
        self._recent_classifications.append((label, confidence))
        if len(self._recent_classifications) > self.history_size:
            self._recent_classifications.pop(0)
        
        return label, confidence, 'current'
    
    def _check_label_reuse(self, track_id: int, current_label: str, current_confidence: float,
                          evidence: Dict[str, Dict[str, Any]]) -> Tuple[str, float, Optional[str]]:
        """
        Check if previous label should be reused instead of current low-confidence classification.
        
        Guards:
        (a) Strong streak exists (length >= STREAK_MIN)
        (b) No higher-confidence conflicting candidate
        (c) Matches burst dominance if available
        (d) Current confidence is below LOW_CONF_THRESHOLD
        
        Args:
            track_id: Track ID (for logging)
            current_label: Current classification label
            current_confidence: Current classification confidence
            evidence: Evidence dict with candidate labels and scores
            
        Returns:
            Tuple of (final_label, final_confidence, reason)
            - If reuse: returns previous label with reason
            - If not: returns current label with None reason
        """
        # Feature flag check
        if not self.enable_label_reuse:
            return current_label, current_confidence, None
        
        # Guard (d): Check if confidence is low enough to consider reuse
        if current_confidence >= self.low_conf_threshold:
            return current_label, current_confidence, None
        
        # Check if we have enough history to determine a streak
        if len(self._recent_classifications) < self.streak_min_length:
            return current_label, current_confidence, None
        
        # Guard (a): Check for strong streak in recent classifications
        # Look at last N classifications to see if they form a consistent streak
        recent_labels = [label for label, _ in self._recent_classifications[-self.streak_min_length:]]
        
        # Check if all recent labels are the same (forming a streak)
        if len(set(recent_labels)) != 1:
            # No consistent streak
            return current_label, current_confidence, None
        
        prev_label = recent_labels[0]  # The label from the streak
        
        # If current label matches the streak, no override needed
        if current_label == prev_label:
            return current_label, current_confidence, None
        
        # Guard (b): Check if there's a higher-confidence conflicting candidate
        # Sort evidence by score
        sorted_evidence = sorted(evidence.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # If current label is not the top candidate, there's a higher-confidence alternative
        # Check if that alternative is the prev_label
        if sorted_evidence:
            top_candidate_label = sorted_evidence[0][0]
            top_candidate_conf = sorted_evidence[0][1]["best_confidence"]
            
            # If top candidate is different from prev_label and has higher confidence, don't reuse
            if top_candidate_label != prev_label and top_candidate_conf > current_confidence:
                return current_label, current_confidence, None
        
        # Guard (c): Check burst dominance if we have enough history
        dominance_label = None
        dominance_ratio = None
        
        if len(self._recent_classifications) >= self.burst_window_size:
            # Analyze last N classifications for burst dominance
            burst_window = self._recent_classifications[-self.burst_window_size:]
            label_counts = Counter([label for label, _ in burst_window])
            
            if label_counts:
                dominant = label_counts.most_common(1)[0]
                dominance_label = dominant[0]
                dominance_ratio = dominant[1] / len(burst_window)
                
                # If burst dominance exists but doesn't match prev_label, don't reuse
                if dominance_ratio >= self.burst_dominance_min_ratio:
                    if dominance_label != prev_label:
                        return current_label, current_confidence, None
        
        # All guards passed: reuse previous label
        # Use average confidence from streak as the reused confidence
        streak_confidences = [conf for _, conf in self._recent_classifications[-self.streak_min_length:]]
        reused_confidence = sum(streak_confidences) / len(streak_confidences)
        
        # Get top candidate labels for logging
        candidate_tops = [(label, data["best_confidence"]) for label, data in sorted_evidence[:3]]
        
        # Log the override decision
        structured_logger.label_reuse_override(
            track_id=track_id,
            prev_label=prev_label,
            new_label=current_label,
            new_confidence=current_confidence,
            streak_len=len(recent_labels),
            dominance_label=dominance_label,
            dominance_ratio=dominance_ratio,
            candidate_tops=candidate_tops,
            reason="low_confidence_with_strong_streak"
        )
        
        logger.info(
            f"[ClassifierService] Track {track_id}: Reusing label {prev_label} "
            f"(streak={len(recent_labels)}, avg_conf={reused_confidence:.2f}) "
            f"instead of low-confidence {current_label} (conf={current_confidence:.2f})"
        )
        
        return prev_label, reused_confidence, "label_reuse"
    
    def _calculate_track_volatility(self, track_id: int) -> Optional[float]:
        """
        Calculate label volatility score for a track.
        
        Volatility = (number of label changes) / (track lifespan)
        
        Args:
            track_id: Track identifier
            
        Returns:
            Volatility score (0.0-1.0) or None if track has insufficient history
        """
        label_history = self._track_label_history.get(track_id, [])
        
        if len(label_history) < 2:
            return None
        
        # Count label changes
        label_changes = 0
        for i in range(1, len(label_history)):
            if label_history[i][0] != label_history[i-1][0]:
                label_changes += 1
        
        # Calculate volatility
        lifespan = len(label_history)
        volatility = label_changes / lifespan
        
        return volatility
    
    def _check_and_log_volatility(self, track_id: int):
        """Check track volatility and log if threshold exceeded."""
        if not self.enable_volatility_logging:
            return
        
        volatility = self._calculate_track_volatility(track_id)
        
        if volatility is None:
            return
        
        if volatility > self.track_volatility_threshold:
            label_history = self._track_label_history[track_id]
            label_changes = sum(1 for i in range(1, len(label_history)) 
                              if label_history[i][0] != label_history[i-1][0])
            
            structured_logger.label_volatility_flag(
                track_id=track_id,
                label_changes=label_changes,
                lifespan=len(label_history),
                volatility_score=volatility,
                label_history=[(label, round(conf, 3)) for label, conf in label_history]
            )
            
            logger.warning(
                f"[ClassifierService] High volatility detected for track {track_id}: "
                f"volatility={volatility:.3f}, changes={label_changes}/{len(label_history)}"
            )

    def _compute_sharpness_weight(self, sharpness: float) -> float:
        """
        Compute sharpness-based weight for evidence scoring.
        
        Higher sharpness = sharper image = more reliable classification.
        Weight is normalized and clamped to prevent extreme values.
        
        Args:
            sharpness: Laplacian variance score
            
        Returns:
            Weight between 0.1 and 1.0
        """
        # Normalize sharpness relative to scale
        normalized = sharpness / self.sharpness_scale
        # Sigmoid-like clamping
        weight = min(1.0, max(0.1, normalized))
        return weight

    def _compute_temporal_weight(self, relative_time: float) -> float:
        """
        Compute temporal weight favoring later frames in the track.
        
        Later frames tend to have better views of the bag after it settles.
        
        Args:
            relative_time: Position in track (0.0 = start, 1.0 = end)
            
        Returns:
            Weight between 0.5 and 1.0
        """
        # Linear scaling: later frames get higher weight
        base_weight = 0.5
        temporal_bonus = self.temporal_scale * relative_time
        return min(1.0, base_weight + temporal_bonus)

    def _accumulate_evidence(self, classifications: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Accumulate evidence across all classified candidates.
        
        For each classified candidate:
            evidence_score[label] += confidence × sharpness_weight × temporal_weight
        
        Also tracks the best representative frame for each label.
        
        Args:
            classifications: List of classification results with metadata
            
        Returns:
            Dictionary mapping labels to evidence data:
            {
                "label": {
                    "score": total_evidence_score,
                    "count": number_of_votes,
                    "best_roi": ROI with highest contribution,
                    "best_confidence": highest confidence for this label,
                    "contributions": list of individual contributions
                }
            }
        """
        evidence: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "score": 0.0,
            "count": 0,
            "best_roi": None,
            "best_confidence": 0.0,
            "contributions": []
        })
        
        for clf in classifications:
            label = clf['label']
            confidence = clf['confidence']
            sharpness = clf['sharpness']
            relative_time = clf['relative_time']
            roi = clf['roi']
            
            # Skip Unknown predictions - they don't contribute evidence
            # Note: If ALL predictions are Unknown, the track will be classified as Unknown
            # with reason "no_valid_classifications" in _finalize_classification().
            # This is intentional: Unknown predictions indicate classifier uncertainty,
            # so they should not influence the evidence accumulation.
            if label == "Unknown":
                continue
            
            # Compute weights
            sharpness_weight = self._compute_sharpness_weight(sharpness)
            temporal_weight = self._compute_temporal_weight(relative_time)
            
            # Calculate contribution with clamping
            raw_contribution = confidence * sharpness_weight * temporal_weight
            clamped_contribution = min(raw_contribution, self.max_single_weight)
            
            # Accumulate evidence
            evidence[label]["score"] += clamped_contribution
            evidence[label]["count"] += 1
            evidence[label]["contributions"].append({
                "confidence": confidence,
                "sharpness": sharpness,
                "temporal": relative_time,
                "raw_contribution": raw_contribution,
                "clamped_contribution": clamped_contribution
            })
            
            # Track best representative frame for this label
            if confidence > evidence[label]["best_confidence"]:
                evidence[label]["best_confidence"] = confidence
                evidence[label]["best_roi"] = roi
        
        return dict(evidence)

    def _finalize_classification(self, evidence: Dict[str, Dict[str, Any]], 
                                 event_stats: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, Any]]:
        """
        Finalize classification using evidence accumulation and ratio thresholds.
        
        Decision Rules:
        1. If no evidence → Unknown (no valid classifications)
        2. If winner_score < min_total_evidence → Unknown (insufficient evidence)
        3. If winner/runner_up < ratio_threshold → Unknown (ambiguous)
        4. Otherwise → Accept winner
        
        Args:
            evidence: Accumulated evidence per label
            event_stats: Track statistics for structural validation
            
        Returns:
            Tuple of (final_label, confidence, rejection_reason, metadata)
        """
        metadata = {
            "evidence_per_label": {},
            "total_candidates_classified": sum(e["count"] for e in evidence.values()),
        }
        
        # Check for structural issues first
        total_rois = event_stats.get("total", 0)
        track_duration = event_stats.get("track_duration_frames", 0)
        avg_sharpness = event_stats.get("avg_sharpness", 0)
        
        # Structural check: too few ROIs
        if total_rois < self.min_candidates:
            self._unknown_structural += 1
            return "Unknown", 0.0, f"too_few_rois ({total_rois} < {self.min_candidates})", metadata
        
        # Structural check: track too short
        if track_duration < self.min_track_frames:
            self._unknown_structural += 1
            return "Unknown", 0.0, f"track_too_short ({track_duration} < {self.min_track_frames})", metadata
        
        # No evidence accumulated (all predictions were Unknown)
        if not evidence:
            self._unknown_structural += 1
            return "Unknown", 0.0, "no_valid_classifications", metadata
        
        # Sort labels by evidence score
        sorted_labels = sorted(evidence.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Record evidence for logging
        for label, data in sorted_labels:
            metadata["evidence_per_label"][label] = {
                "score": round(data["score"], 4),
                "count": data["count"],
                "best_confidence": round(data["best_confidence"], 3)
            }
        
        winner_label, winner_data = sorted_labels[0]
        winner_score = winner_data["score"]
        winner_confidence = winner_data["best_confidence"]
        
        # Check minimum evidence threshold
        if winner_score < self.min_total_evidence:
            self._unknown_low_evidence += 1
            return "Unknown", winner_confidence, f"low_evidence ({winner_score:.3f} < {self.min_total_evidence})", metadata
        
        # Calculate ratio against runner-up
        if len(sorted_labels) > 1:
            runner_up_label, runner_up_data = sorted_labels[1]
            runner_up_score = runner_up_data["score"]
            
            # Avoid division by zero - if runner_up has no score, winner wins by default
            if runner_up_score > 1e-9:  # Use small epsilon for floating point comparison
                ratio = winner_score / runner_up_score
                metadata["winner_ratio"] = round(ratio, 3)
                metadata["runner_up"] = {
                    "label": runner_up_label,
                    "score": round(runner_up_score, 4)
                }
                
                if ratio < self.ratio_threshold:
                    self._unknown_ambiguous += 1
                    return "Unknown", winner_confidence, f"ambiguous ({ratio:.2f} < {self.ratio_threshold})", metadata
            else:
                # Runner-up has essentially zero evidence, winner is uncontested
                metadata["winner_ratio"] = float('inf')
                metadata["runner_up"] = {
                    "label": runner_up_label,
                    "score": 0.0
                }
        
        # Accept winner
        metadata["accepted"] = True
        metadata["winner_score"] = round(winner_score, 4)
        return winner_label, winner_confidence, None, metadata

    def _select_best_representative(self, evidence: Dict[str, Dict[str, Any]], 
                                    final_label: str) -> Optional[Any]:
        """
        Select the best representative ROI for the final label.
        
        Args:
            evidence: Accumulated evidence per label
            final_label: The winning label
            
        Returns:
            Best ROI image for the label, or None
        """
        if final_label in evidence:
            return evidence[final_label].get("best_roi")
        return None

    def process(self, track_id: int, candidates_input: List[Dict], context: Optional[Dict[str, Any]] = None):
        """
        Process classification for a completed track.
        
        V4 Evidence-Based Classification Pipeline:
        1. Validate structural requirements (min ROIs, track length)
        2. Classify each candidate ROI
        3. Accumulate evidence with weighted scoring
        4. Finalize using winner/runner-up ratio
        5. Select best representative frame
        6. Invoke callbacks with result
        
        Args:
            track_id: Unique track identifier
            candidates_input: List of candidate dictionaries from BagEvent.get_all_candidates()
            context: Optional context for snapshot saving
        """
        try:
            self._total_classified += 1
            batch_start = time.perf_counter()
            
            # V4: candidates_input is now a list of dicts with metadata
            if isinstance(candidates_input, list) and len(candidates_input) > 0:
                if isinstance(candidates_input[0], dict):
                    candidates = candidates_input
                else:
                    # Backward compatibility: convert old format (list of ROI images)
                    candidates = [{'roi': roi, 'sharpness': 100.0, 'frame_index': 0, 
                                   'bbox_area': 0, 'confidence': 0.8, 'relative_time': 0.5} 
                                  for roi in candidates_input]
            else:
                candidates = []
            
            # Extract event_stats from context if available
            event_stats = context.get("event_stats", {}) if context else {}
            
            # Structural validation
            if not candidates:
                self._invoke_unknown_result(track_id, "no_candidates", context)
                return
            
            # Step 1: Classify each candidate
            classifications = []
            for idx, cand in enumerate(candidates):
                roi = cand['roi']
                label, conf = self._classify_single(roi, idx)
                
                # Calculate contribution for this candidate
                sharpness = cand.get('sharpness', 0)
                relative_time = cand.get('relative_time', 0.5)
                sharpness_weight = self._compute_sharpness_weight(sharpness)
                temporal_weight = self._compute_temporal_weight(relative_time)
                raw_contribution = conf * sharpness_weight * temporal_weight
                clamped_contribution = min(raw_contribution, self.max_single_weight)
                
                # Structured logging for candidate classification
                structured_logger.classification_candidate(
                    track_id=track_id,
                    candidate_idx=idx,
                    label=label,
                    confidence=conf,
                    sharpness=sharpness,
                    relative_time=relative_time,
                    contribution=clamped_contribution,
                    frame_index=cand.get('frame_index', 0)
                )
                
                classifications.append({
                    'label': label,
                    'confidence': conf,
                    'roi': roi,
                    'sharpness': sharpness,
                    'frame_index': cand.get('frame_index', 0),
                    'relative_time': relative_time,
                })
            
            classify_time = (time.perf_counter() - batch_start) * 1000
            
            # Step 2: Accumulate evidence
            evidence = self._accumulate_evidence(classifications)
            
            # Step 3: Finalize classification
            final_label, final_conf, rejection_reason, metadata = self._finalize_classification(
                evidence, event_stats
            )
            
            # Step 3.5: Apply classification smoothing (V5)
            # Only smooth non-Unknown classifications
            if final_label != "Unknown":
                smoothed_label, smoothed_conf, smooth_reason = self._apply_classification_smoothing(
                    final_label, final_conf
                )
                
                # Update if smoothing changed the result
                if smoothed_label != final_label or abs(smoothed_conf - final_conf) > 0.01:
                    metadata["smoothing_applied"] = True
                    metadata["original_label"] = final_label
                    metadata["original_confidence"] = final_conf
                    metadata["smoothing_reason"] = smooth_reason
                    
                    final_label = smoothed_label
                    final_conf = smoothed_conf
                    
                    logger.info(
                        f"[ClassifierService] Track {track_id}: Smoothing changed result: "
                        f"{metadata['original_label']}({metadata['original_confidence']:.2f}) -> "
                        f"{final_label}({final_conf:.2f}), reason={smooth_reason}"
                    )
            
            # Step 3.6: Check for label reuse (V6 Production-grade stability heuristics)
            # Only apply to non-Unknown classifications with low confidence
            if final_label != "Unknown":
                reuse_label, reuse_conf, reuse_reason = self._check_label_reuse(
                    track_id, final_label, final_conf, evidence
                )
                
                # Update if label reuse changed the result
                if reuse_label != final_label:
                    metadata["label_reuse_applied"] = True
                    metadata["pre_reuse_label"] = final_label
                    metadata["pre_reuse_confidence"] = final_conf
                    metadata["reuse_reason"] = reuse_reason
                    
                    final_label = reuse_label
                    final_conf = reuse_conf
                    
                    logger.info(
                        f"[ClassifierService] Track {track_id}: Label reuse changed result: "
                        f"{metadata['pre_reuse_label']}({metadata['pre_reuse_confidence']:.2f}) -> "
                        f"{final_label}({final_conf:.2f}), reason={reuse_reason}"
                    )
            
            # Step 3.7: Track label history for volatility analysis (V6)
            self._track_label_history[track_id].append((final_label, final_conf))
            
            # Step 3.8: Check and log track volatility (V6)
            self._check_and_log_volatility(track_id)
            
            # Step 4: Select best representative ROI
            if final_label != "Unknown":
                best_roi = self._select_best_representative(evidence, final_label)
            else:
                # For Unknown, use the ROI with highest individual confidence
                best_classification = max(classifications, key=lambda x: x['confidence'])
                best_roi = best_classification['roi']
                final_conf = best_classification['confidence']
            
            # Log decision with full explainability
            self._log_classification_decision(
                track_id, final_label, final_conf, rejection_reason, 
                metadata, len(candidates), classify_time
            )
            
            # Include rejection reason in metadata for callbacks
            if rejection_reason:
                metadata["rejection_reason"] = rejection_reason
                
                # Add unknown_kind for Unknown classifications
                if final_label == "Unknown":
                    unknown_kind = "structural"  # default
                    if "low_evidence" in rejection_reason:
                        unknown_kind = "low_evidence"
                    elif "ambiguous" in rejection_reason:
                        unknown_kind = "ambiguous"
                    elif "too_few_rois" in rejection_reason or "track_too_short" in rejection_reason or "no_valid_classifications" in rejection_reason:
                        unknown_kind = "structural"
                    metadata["unknown_kind"] = unknown_kind
            
            # Save ROI and invoke callbacks
            self._save_and_callback(
                track_id, best_roi, final_label, final_conf, 
                len(candidates), metadata, context
            )

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            
            # Structured error logging
            structured_logger.pipeline_error(
                component='ClassifierService',
                operation='track_classification',
                error_type=type(e).__name__,
                error_message=str(e),
                affected_ids=[track_id],
                context={
                    'candidates_count': len(candidates_input) if candidates_input else 0,
                    'event_stats': event_stats if 'event_stats' in locals() else {}
                },
                traceback=error_trace
            )

    def _log_classification_decision(self, track_id: int, label: str, confidence: float,
                                     rejection_reason: Optional[str], metadata: Dict,
                                     num_candidates: int, classify_time_ms: float):
        """
        Log the classification decision with full explainability.
        
        This implements Task 10: Explainable logging per track.
        """
        evidence_summary = metadata.get("evidence_per_label", {})
        winner_ratio = metadata.get("winner_ratio", "N/A")
        
        # Structured logging for analysis
        # Validate winner_ratio for logging (handle float, int, inf, and NaN)
        import math
        valid_ratio = None
        if winner_ratio is not None:
            if isinstance(winner_ratio, (int, float)):
                if math.isfinite(winner_ratio):
                    valid_ratio = float(winner_ratio)
                # Infinite or NaN - log as None
        
        structured_logger.classification_result(
            track_id=track_id,
            label=label,
            confidence=confidence,
            candidates=num_candidates,
            used_voting=True,  # V4 always uses evidence accumulation
            rejection_reason=rejection_reason,
            evidence_scores=evidence_summary,
            winner_ratio=valid_ratio,
            processing_time_ms=classify_time_ms
        )

    def _invoke_unknown_result(self, track_id: int, reason: str, context: Optional[Dict]):
        """Invoke callbacks with Unknown result for structural failures."""
        # Determine unknown_kind based on reason
        unknown_kind = "structural"  # default for structural issues like no_candidates
        if reason and ("low_evidence" in reason or "insufficient" in reason):
            unknown_kind = "low_evidence"
        elif reason and ("ambiguous" in reason):
            unknown_kind = "ambiguous"
        
        result_data = {
            "label": "Unknown",
            "phash": None,  # Changed from "unknown" to None to avoid hex conversion crashes
            "image_path": None,
            "confidence": 0.0,
            "candidates_evaluated": 0,
            "context": context,
            "metadata": {
                "rejection_reason": reason,
                "unknown_kind": unknown_kind  # Added machine-readable category
            },
        }
        
        for cb in self.callbacks:
            try:
                cb(track_id, result_data)
            except Exception as e:
                logger.error(f"[ClassifierService] Callback error: {e}")

    def _save_and_callback(self, track_id: int, best_roi: Any, label: str, 
                           confidence: float, candidates_count: int,
                           metadata: Dict, context: Optional[Dict]):
        """Save ROI image and invoke registered callbacks."""
        if best_roi is None:
            logger.error(f"[ClassifierService] Track {track_id}: No valid ROI!")
            return
        
        # Compute phash (only for non-Unknown or when we have actual ROI data)
        phash_obj = compute_phash(best_roi)
        phash_str = str(phash_obj) if phash_obj else None
        
        # Determine save path
        if label == "Unknown":
            # For Unknown, use a single directory instead of per-phash directories
            target_dir = os.path.join(self.data_root, "unknown", "unknown_samples")
        else:
            target_dir = os.path.join(self.data_root, "classes", label)
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Save logic
        should_save = False
        existing_files = os.listdir(target_dir)
        
        if self.save_all_rois:
            should_save = True
        elif not existing_files:
            should_save = True
        
        image_path = None
        if should_save:
            timestamp = int(time.time())
            filename = f"{timestamp}_{track_id}.jpg"
            save_path = os.path.join(target_dir, filename)
            cv2.imwrite(save_path, best_roi)
            image_path = save_path
        elif existing_files:
            image_path = os.path.join(target_dir, existing_files[0])
        
        # Prepare result data
        result_data = {
            "label": label,
            "phash": phash_str,
            "image_path": image_path,
            "confidence": confidence,
            "candidates_evaluated": candidates_count,
            "context": context,
            "metadata": metadata,
        }
        
        # Record metrics
        pipeline_metrics.record_classification(
            label, confidence, candidates_count, used_voting=True
        )
        
        # Invoke callbacks
        for cb in self.callbacks:
            try:
                cb(track_id, result_data)
            except Exception as e:
                structured_logger.pipeline_error(
                    component="ClassifierService",
                    operation="callback_invocation",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    affected_ids=[track_id],
                    context={"label": label, "confidence": confidence}
                )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics for monitoring.
        
        Returns:
            Dictionary with classification statistics including stability heuristics
        """
        # Calculate volatility statistics across all tracks
        volatility_scores = []
        high_volatility_count = 0
        
        for track_id, label_history in self._track_label_history.items():
            volatility = self._calculate_track_volatility(track_id)
            if volatility is not None:
                volatility_scores.append(volatility)
                if volatility > self.track_volatility_threshold:
                    high_volatility_count += 1
        
        avg_volatility = sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0
        
        return {
            "total_classified": self._total_classified,
            "unknown_structural": self._unknown_structural,
            "unknown_low_evidence": self._unknown_low_evidence,
            "unknown_ambiguous": self._unknown_ambiguous,
            "successful": self._total_classified - (
                self._unknown_structural + self._unknown_low_evidence + self._unknown_ambiguous
            ),
            # V6: Stability heuristics
            "stability_heuristics": {
                "enable_label_reuse": self.enable_label_reuse,
                "low_conf_threshold": self.low_conf_threshold,
                "streak_min_length": self.streak_min_length,
                "burst_dominance_min_ratio": self.burst_dominance_min_ratio,
                "burst_window_size": self.burst_window_size,
                "track_volatility_threshold": self.track_volatility_threshold,
                "tracks_analyzed": len(self._track_label_history),
                "avg_volatility": avg_volatility,
                "high_volatility_tracks": high_volatility_count,
            },
        }
