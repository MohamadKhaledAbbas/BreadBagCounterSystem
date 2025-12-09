import logging
import os
import time
from collections import Counter, defaultdict
from typing import Callable, List, Dict, Any, Tuple, Optional

from src.classifier.BaseClassifier import BaseClassifier
from src.utils.Utils import compute_phash
from src.utils.AppLogging import logger

ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    # Low confidence thresholds
    LOW_CONFIDENCE_THRESHOLD = 0.5  # Confidence below this is considered low
    LOW_MARGIN_THRESHOLD = 0.2      # Decision margin below this is considered ambiguous
    
    def __init__(self,
                 classifier: BaseClassifier,
                 data_root: str = "data",
                 save_all_rois: bool = False,
                 min_confidence_threshold: float = 0.3,
                 use_voting: bool = True,
                 voting_top_k: Optional[int] = None,
                 voting_accept_norm_threshold: float = 0.6,
                 voting_accept_margin: float = 0.15):

        self.classifier = classifier
        self.data_root = data_root
        self.save_all_rois = save_all_rois
        self.min_confidence_threshold = min_confidence_threshold
        self.use_voting = use_voting
        # Default None processes all candidates; set a positive value to restore legacy top-k behavior.
        self.voting_top_k = voting_top_k
        self.voting_accept_norm_threshold = voting_accept_norm_threshold
        self.voting_accept_margin = voting_accept_margin

        self.callbacks: List[ResultCallback] = []
        self.running = True

        candidate_cap = self._get_candidate_cap()
        candidate_cap_desc = 'all' if candidate_cap is None else candidate_cap

        logger.info(
            f"[ClassifierService] Initialized: voting={use_voting}, "
            f"weighted_conf_threshold={voting_accept_norm_threshold}, "
            f"margin_threshold={voting_accept_margin}, "
            f"candidates_per_vote={candidate_cap_desc}, "
            f"low_conf_threshold={self.LOW_CONFIDENCE_THRESHOLD}, "
            f"low_margin_threshold={self.LOW_MARGIN_THRESHOLD}"
        )

    def _get_candidate_cap(self) -> Optional[int]:
        """
        Optional cap on number of candidates to include in weighted voting based on `voting_top_k`.
        None or any non-positive value means use all candidates (legacy compatibility and new default).
        Centralizes the limit handling so call sites don't duplicate the same checks.
        """
        if self.voting_top_k is None or self.voting_top_k <= 0:
            return None
        return self.voting_top_k

    def _calculate_margin(self, label_scores: Dict[str, float], winning_label: str, winning_score: float,
                          total_score: float) -> float:
        tie_epsilon = 1e-9
        winning_labels = [label for label, score in label_scores.items() if abs(score - winning_score) < tie_epsilon]
        if len(label_scores) == 1:
            return 1.0
        if len(winning_labels) > 1:
            return 0.0

        second_score = max(
            (score for label, score in label_scores.items() if label != winning_label),
            default=0.0
        )
        return (winning_score - second_score) / total_score if total_score > 0 else 0.0

    def register_callback(self, callback: ResultCallback):
        self.callbacks.append(callback)

    def _classify_single(self, roi_image, idx: int = 0) -> Tuple[str, float]:
        """Classify a single ROI."""
        try:
            t1 = time.perf_counter()
            label, conf = self.classifier.predict(roi_image)
            t2 = time.perf_counter()
            processing_time = (t2 - t1) * 1000  # Convert to milliseconds
            logger.debug(f"[ClassifierService] Candidate {idx}: {label} ({conf:.3f}) - {processing_time:.1f}ms")
            return label, conf
        except Exception as e:
            logger.error(f"[ClassifierService] Classification error: {e}")
            return "Unknown", 0.0

    def _select_best_with_voting(self, candidates: List) -> Tuple[Optional[Any], str, float, float]:
        """
        Classify all candidates and use voting to select the best label.
        Returns: (best_roi, winning_label, confidence, decision_margin)
        """
        if not candidates:
            return None, "Unknown", 0.0

        results = []

        batch_start = time.perf_counter()
        logger.info(f"[ClassifierService] Classifying {len(candidates)} candidates...")

        # Classify all candidates
        for idx, roi in enumerate(candidates):
            label, conf = self._classify_single(roi, idx)
            results.append({
                'roi': roi,
                'label': label,
                'conf': conf
            })
        
        batch_end = time.perf_counter()
        total_batch_time = (batch_end - batch_start) * 1000  # Convert to milliseconds

        # Filter out unknowns for voting
        valid_results = [r for r in results if r['label'] != "Unknown"]

        if not valid_results:
            # All unknown - return best unknown with 0 margin (ambiguous)
            logger.warning("[ClassifierService] All candidates Unknown!")
            best_unknown = max(results, key=lambda x: x['conf'])
            return best_unknown['roi'], "Unknown", best_unknown['conf'], 0.0

        # Sort by confidence (highest first) and optionally limit candidates for voting
        valid_results.sort(key=lambda x: x['conf'], reverse=True)
        candidate_cap = self._get_candidate_cap()
        candidates_for_vote = valid_results if candidate_cap is None else valid_results[:candidate_cap]

        # Confidence-weighted voting across chosen candidates
        label_scores: Dict[str, float] = defaultdict(float)
        label_counts = Counter()
        for r in candidates_for_vote:
            label_scores[r['label']] += r['conf']
            label_counts[r['label']] += 1

        winning_label, winning_score = max(label_scores.items(), key=lambda x: x[1])
        total_score = sum(label_scores.values())
        normalized_score = winning_score / total_score if total_score > 0 else 0.0

        # Margin against the second-best weighted score
        margin = self._calculate_margin(label_scores, winning_label, winning_score, total_score)

        winning_results = [r for r in candidates_for_vote if r['label'] == winning_label]
        best_result = max(winning_results, key=lambda x: x['conf'])
        best_overall = candidates_for_vote[0]

        # OR condition is intentional: a decisive margin between classes should pass even if the normalized
        # score (share of total confidence) is moderate, while strong absolute confidence also passes without
        # needing a large margin. Typical defaults: normalized >= 0.6 or margin >= 0.15.
        accepted = (normalized_score >= self.voting_accept_norm_threshold) or (margin >= self.voting_accept_margin)
        final_label = winning_label if accepted else "Unknown"
        selected_result = best_result if accepted else best_overall
        selected_roi = selected_result['roi']
        selected_conf = selected_result['conf']

        if not accepted:
            logger.warning(
                f"[ClassifierService] Weighted voting uncertain - winner={winning_label} "
                f"(norm={normalized_score:.2f}, margin={margin:.2f}); marking as Unknown "
                f"using thresholds (norm>={self.voting_accept_norm_threshold}, margin>={self.voting_accept_margin})"
            )

        logger.info(
            f"[ClassifierService] Weighted voting result: {final_label} "
            f"(winner={winning_label}, best_conf={best_result['conf']:.3f}, "
            f"norm={normalized_score:.2f}, margin={margin:.2f}, "
            f"candidates_used={len(candidates_for_vote)}, total_candidates={len(valid_results)}, "
            f"time={total_batch_time:.1f}ms)"
        )

        # Log weighted distribution and counts for observability
        debug_level = getattr(logger, "DEBUG", logging.DEBUG)
        if logger.isEnabledFor(debug_level):
            weight_dist = ", ".join([
                f"{label}: sum={label_scores[label]:.3f}, count={label_counts[label]}"
                for label in label_scores
            ])
            logger.debug(f"[ClassifierService] Weighted distribution: {weight_dist}")

        return selected_roi, final_label, selected_conf, margin

    def _select_best_by_confidence(self, candidates: List) -> Tuple[Optional[Any], str, float, float]:
        """
        Select the single best candidate by confidence (no voting).
        Returns: (best_roi, label, confidence, margin)
        For single-candidate selection, margin is always 0 (no comparison possible).
        """
        if not candidates:
            return None, "Unknown", 0.0, 0.0

        best_roi = None
        best_label = "Unknown"
        best_confidence = 0.0
        best_unknown_roi = None
        best_unknown_conf = 0.0

        for idx, roi in enumerate(candidates):
            label, conf = self._classify_single(roi, idx)

            if label == "Unknown":
                if conf > best_unknown_conf:
                    best_unknown_roi = roi
                    best_unknown_conf = conf
            else:
                if conf > best_confidence:
                    best_roi = roi
                    best_label = label
                    best_confidence = conf

        # For single-candidate selection, no meaningful margin (no second choice to compare)
        margin = 0.0
        
        if best_roi is not None:
            logger.info(f"[ClassifierService] Best: {best_label} (conf={best_confidence:.3f})")
            return best_roi, best_label, best_confidence, margin
        else:
            return best_unknown_roi, "Unknown", best_unknown_conf, margin

    def process(self, track_id: int, roi_input):
        """Process classification request."""
        try:
            # Handle list vs single image
            if isinstance(roi_input, list):
                candidates = roi_input
            else:
                candidates = [roi_input]

            logger.info(f"[ClassifierService] Track {track_id}: {len(candidates)} candidates")

            if not candidates:
                logger.error(f"[ClassifierService] Track {track_id}: Empty candidates!")
                return

            # Select best candidate (with or without voting)
            if self.use_voting and len(candidates) >= 3:
                best_roi, label, conf, margin = self._select_best_with_voting(candidates)
            else:
                best_roi, label, conf, margin = self._select_best_by_confidence(candidates)

            if best_roi is None:
                logger.error(f"[ClassifierService] Track {track_id}: No valid ROI!")
                return

            # Determine if this is a low confidence classification
            is_low_confidence = (
                conf < self.LOW_CONFIDENCE_THRESHOLD or 
                margin < self.LOW_MARGIN_THRESHOLD
            )
            
            # Low confidence warning
            if is_low_confidence:
                logger.warning(
                    f"[ClassifierService] Track {track_id}: Low confidence classification - "
                    f"{label} (conf={conf:.3f}, margin={margin:.3f})"
                )
            elif conf < self.min_confidence_threshold:
                logger.warning(
                    f"[ClassifierService] Track {track_id}: Below minimum threshold "
                    f"{label} ({conf:.3f} < {self.min_confidence_threshold})"
                )

            # Compute hash
            phash_obj = compute_phash(best_roi)
            phash_str = str(phash_obj)

            # Save logic
            if label == "Unknown":
                target_dir = os.path.join(self.data_root, "unknown", phash_str)
            else:
                target_dir = os.path.join(self.data_root, "classes", label)

            os.makedirs(target_dir, exist_ok=True)

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

                import cv2
                cv2.imwrite(save_path, best_roi)
                image_path = save_path
            elif existing_files:
                image_path = os.path.join(target_dir, existing_files[0])

            # Result
            result_data = {
                "label": label,
                "phash": phash_str,
                "image_path": image_path,
                "confidence": conf,
                "is_low_confidence": is_low_confidence,
                "decision_margin": margin,
                "candidates_evaluated": len(candidates)
            }

            logger.info(
                f"[ClassifierService] Track {track_id} DONE: {label} "
                f"(conf={conf:.3f}, margin={margin:.3f}, low_conf={is_low_confidence}, "
                f"candidates={len(candidates)})"
            )

            # Callbacks
            for cb in self.callbacks:
                try:
                    cb(track_id, result_data)
                except Exception as e:
                    logger.error(f"[ClassifierService] Callback error: {e}")

        except Exception as e:
            logger.error(f"[ClassifierService] Process error: {e}")
            import traceback
            logger.error(traceback.format_exc())
