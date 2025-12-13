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

        logger.info(
            f"[ClassifierService] Initialized V4 (Evidence-Based): "
            f"top_k={self.top_k}, min_evidence={self.min_total_evidence}, "
            f"ratio_threshold={self.ratio_threshold}, min_candidates={self.min_candidates}"
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
            t1 = time.perf_counter()
            label, conf = self.classifier.predict(roi_image)
            t2 = time.perf_counter()
            processing_time = (t2 - t1) * 1000
            logger.debug(f"[ClassifierService] Candidate {idx}: {label} ({conf:.3f}) - {processing_time:.1f}ms")
            return label, float(conf)
        except Exception as e:
            logger.error(f"[ClassifierService] Classification error: {e}")
            return "Unknown", 0.0

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
                    # Warning: This path uses default metadata which may lead to suboptimal
                    # evidence accumulation. Prefer using the new dict format.
                    logger.warning(
                        f"[ClassifierService] Track {track_id}: Using legacy candidate format. "
                        f"Update to new dict format for optimal evidence weighting."
                    )
                    candidates = [{'roi': roi, 'sharpness': 100.0, 'frame_index': 0, 
                                   'bbox_area': 0, 'confidence': 0.8, 'relative_time': 0.5} 
                                  for roi in candidates_input]
            else:
                candidates = []
            
            # Extract event_stats from context if available
            event_stats = context.get("event_stats", {}) if context else {}
            
            logger.info(
                f"[ClassifierService] Track {track_id}: Processing {len(candidates)} candidates "
                f"(top-K={self.top_k})"
            )
            
            # Structural validation
            if not candidates:
                logger.warning(f"[ClassifierService] Track {track_id}: No candidates!")
                self._invoke_unknown_result(track_id, "no_candidates", context)
                return
            
            # Step 1: Classify each candidate
            classifications = []
            for idx, cand in enumerate(candidates):
                roi = cand['roi']
                label, conf = self._classify_single(roi, idx)
                
                classifications.append({
                    'label': label,
                    'confidence': conf,
                    'roi': roi,
                    'sharpness': cand.get('sharpness', 0),
                    'frame_index': cand.get('frame_index', 0),
                    'relative_time': cand.get('relative_time', 0.5),
                })
            
            classify_time = (time.perf_counter() - batch_start) * 1000
            
            # Step 2: Accumulate evidence
            evidence = self._accumulate_evidence(classifications)
            
            # Step 3: Finalize classification
            final_label, final_conf, rejection_reason, metadata = self._finalize_classification(
                evidence, event_stats
            )
            
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
            
            # Save ROI and invoke callbacks
            self._save_and_callback(
                track_id, best_roi, final_label, final_conf, 
                len(candidates), metadata, context
            )

        except Exception as e:
            logger.error(f"[ClassifierService] Process error for track {track_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _log_classification_decision(self, track_id: int, label: str, confidence: float,
                                     rejection_reason: Optional[str], metadata: Dict,
                                     num_candidates: int, classify_time_ms: float):
        """
        Log the classification decision with full explainability.
        
        This implements Task 10: Explainable logging per track.
        """
        evidence_summary = metadata.get("evidence_per_label", {})
        winner_ratio = metadata.get("winner_ratio", "N/A")
        
        if rejection_reason:
            logger.warning(
                f"[ClassifierService] Track {track_id} -> Unknown: {rejection_reason}\n"
                f"  Candidates: {num_candidates}, Time: {classify_time_ms:.1f}ms\n"
                f"  Evidence: {evidence_summary}"
            )
        else:
            logger.info(
                f"[ClassifierService] Track {track_id} -> {label} (conf={confidence:.3f})\n"
                f"  Winner score: {metadata.get('winner_score', 'N/A')}, Ratio: {winner_ratio}\n"
                f"  Candidates: {num_candidates}, Time: {classify_time_ms:.1f}ms\n"
                f"  Evidence: {evidence_summary}"
            )
        
        # Structured logging for analysis
        structured_logger.classification_result(
            track_id=track_id,
            label=label,
            confidence=confidence,
            candidates=num_candidates,
            used_voting=True,  # V4 always uses evidence accumulation
            entropy=0.0,  # Not used in V4
            margin=winner_ratio if isinstance(winner_ratio, float) else 0.0,
        )

    def _invoke_unknown_result(self, track_id: int, reason: str, context: Optional[Dict]):
        """Invoke callbacks with Unknown result for structural failures."""
        result_data = {
            "label": "Unknown",
            "phash": "unknown",
            "image_path": None,
            "confidence": 0.0,
            "candidates_evaluated": 0,
            "context": context,
            "metadata": {"rejection_reason": reason},
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
        
        # Compute phash
        phash_obj = compute_phash(best_roi)
        phash_str = str(phash_obj)
        
        # Determine save path
        if label == "Unknown":
            target_dir = os.path.join(self.data_root, "unknown", phash_str)
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
        
        logger.info(
            f"[ClassifierService] Track {track_id} DONE: {label} "
            f"(conf={confidence:.3f}, candidates={candidates_count})"
        )
        
        # Invoke callbacks
        for cb in self.callbacks:
            try:
                cb(track_id, result_data)
            except Exception as e:
                logger.error(f"[ClassifierService] Callback error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics for monitoring.
        
        Returns:
            Dictionary with classification statistics
        """
        return {
            "total_classified": self._total_classified,
            "unknown_structural": self._unknown_structural,
            "unknown_low_evidence": self._unknown_low_evidence,
            "unknown_ambiguous": self._unknown_ambiguous,
            "successful": self._total_classified - (
                self._unknown_structural + self._unknown_low_evidence + self._unknown_ambiguous
            ),
        }
