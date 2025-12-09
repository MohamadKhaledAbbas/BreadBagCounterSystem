import os
import time
from collections import Counter
from typing import Callable, List, Dict, Any, Tuple, Optional

from src.classifier.BaseClassifier import BaseClassifier
from src.utils.Utils import compute_phash
from src.utils.AppLogging import logger

ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    def __init__(self,
                 classifier: BaseClassifier,
                 data_root: str = "data",
                 save_all_rois: bool = False,
                 min_confidence_threshold: float = 0.3,
                 use_voting: bool = True,
                 voting_top_k: int = 5):

        self.classifier = classifier
        self.data_root = data_root
        self.save_all_rois = save_all_rois
        self.min_confidence_threshold = min_confidence_threshold
        self.use_voting = use_voting
        self.voting_top_k = voting_top_k

        self.callbacks: List[ResultCallback] = []
        self.running = True

        logger.info(
            f"[ClassifierService] Initialized: voting={use_voting}, top_k={voting_top_k}"
        )

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

    def _select_best_with_voting(self, candidates: List) -> Tuple[Optional[Any], str, float]:
        """
        Classify all candidates and use voting to select the best label.
        Returns: (best_roi, winning_label, confidence)
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
            # All unknown - return best unknown
            logger.warning("[ClassifierService] All candidates Unknown!")
            best_unknown = max(results, key=lambda x: x['conf'])
            return best_unknown['roi'], "Unknown", best_unknown['conf']

        # Sort by confidence (highest first)
        valid_results.sort(key=lambda x: x['conf'], reverse=True)

        # Confidence-weighted voting across all valid candidates
        label_scores: Dict[str, float] = {}
        label_counts = Counter()
        for r in valid_results:
            label_scores[r['label']] = label_scores.get(r['label'], 0.0) + r['conf']
            label_counts[r['label']] += 1

        winning_label, winning_score = max(label_scores.items(), key=lambda x: x[1])
        total_score = sum(label_scores.values())
        normalized_score = winning_score / total_score if total_score > 0 else 0.0

        # Margin against the second-best weighted score
        second_score = max([score for label, score in label_scores.items() if label != winning_label], default=0.0)
        margin = (winning_score - second_score) / total_score if total_score > 0 else 0.0

        winning_results = [r for r in valid_results if r['label'] == winning_label]
        best_result = max(winning_results, key=lambda x: x['conf'])
        best_overall = valid_results[0]

        accepted = (normalized_score >= 0.6) or (margin >= 0.15)
        final_label = winning_label if accepted else "Unknown"
        selected_roi = best_result['roi'] if accepted else best_overall['roi']
        selected_conf = best_result['conf'] if accepted else best_overall['conf']

        if not accepted:
            logger.warning(
                f"[ClassifierService] Weighted voting uncertain - winner={winning_label} "
                f"(norm={normalized_score:.2f}, margin={margin:.2f}); marking as Unknown"
            )

        logger.info(
            f"[ClassifierService] Weighted voting result: {final_label} "
            f"(winner={winning_label}, best_conf={best_result['conf']:.3f}, "
            f"norm={normalized_score:.2f}, margin={margin:.2f}, "
            f"candidates={len(valid_results)}, time={total_batch_time:.1f}ms)"
        )

        # Log weighted distribution and counts for observability
        weight_dist = ", ".join([
            f"{label}: sum={label_scores[label]:.3f}, count={label_counts[label]}"
            for label in label_scores
        ])
        logger.debug(f"[ClassifierService] Weighted distribution: {weight_dist}")

        return selected_roi, final_label, selected_conf

    def _select_best_by_confidence(self, candidates: List) -> Tuple[Optional[Any], str, float]:
        """
        Select the single best candidate by confidence (no voting).
        """
        if not candidates:
            return None, "Unknown", 0.0

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

        if best_roi is not None:
            logger.info(f"[ClassifierService] Best: {best_label} (conf={best_confidence:.3f})")
            return best_roi, best_label, best_confidence
        else:
            return best_unknown_roi, "Unknown", best_unknown_conf

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
                best_roi, label, conf = self._select_best_with_voting(candidates)
            else:
                best_roi, label, conf = self._select_best_by_confidence(candidates)

            if best_roi is None:
                logger.error(f"[ClassifierService] Track {track_id}: No valid ROI!")
                return

            # Low confidence warning
            if conf < self.min_confidence_threshold:
                logger.warning(
                    f"[ClassifierService] Track {track_id}: Low confidence "
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
                "candidates_evaluated": len(candidates)
            }

            logger.info(
                f"[ClassifierService] Track {track_id} DONE: {label} "
                f"(conf={conf:.3f}, candidates={len(candidates)})"
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
