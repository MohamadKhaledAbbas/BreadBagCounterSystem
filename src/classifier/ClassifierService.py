import os
import time
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
                 voting_top_k: Optional[int] = None,
                 weighted_score_threshold: float = 0.55,
                 weighted_margin_threshold: float = 0.10):

        self.classifier = classifier
        self.data_root = data_root
        self.save_all_rois = save_all_rois
        self.min_confidence_threshold = min_confidence_threshold
        self.use_voting = use_voting
        self.voting_top_k = voting_top_k
        self.weighted_score_threshold = weighted_score_threshold
        self.weighted_margin_threshold = weighted_margin_threshold

        self.callbacks: List[ResultCallback] = []
        self.running = True

        logger.info(
            f"[ClassifierService] Initialized: voting={use_voting}, "
            f"weighted_top_k={voting_top_k or 'all'}, "
            f"norm_thresh={self.weighted_score_threshold}, "
            f"margin_thresh={self.weighted_margin_threshold}"
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

        # Sort by confidence (descending) and optionally cap candidates
        valid_results.sort(key=lambda x: x['conf'], reverse=True)
        if self.voting_top_k is not None and self.voting_top_k > 0:
            selected_results = valid_results[:self.voting_top_k]
            logger.debug(f"[ClassifierService] Applying top_k cap: {self.voting_top_k}")
        else:
            selected_results = valid_results

        label_conf_sums: Dict[str, float] = {}
        label_counts: Dict[str, int] = {}
        label_max_conf: Dict[str, float] = {}

        for r in selected_results:
            label = r['label']
            label_conf_sums[label] = label_conf_sums.get(label, 0.0) + r['conf']
            label_counts[label] = label_counts.get(label, 0) + 1
            label_max_conf[label] = max(label_max_conf.get(label, 0.0), r['conf'])

        total_conf_sum = sum(label_conf_sums.values())
        if total_conf_sum <= 0:
            logger.warning("[ClassifierService] No positive confidences, falling back to max confidence candidate.")
            best_result = max(selected_results, key=lambda x: x['conf'])
            return best_result['roi'], "Unknown", max(best_result['conf'], 0.0)

        winning_label = max(label_conf_sums, key=label_conf_sums.get)
        winning_sum = label_conf_sums[winning_label]
        runner_up_sum = max(
            (v for k, v in label_conf_sums.items() if k != winning_label),
            default=0.0
        )

        normalized_score = winning_sum / total_conf_sum
        runner_up_normalized = runner_up_sum / total_conf_sum
        margin = normalized_score - runner_up_normalized

        winning_results = [r for r in selected_results if r['label'] == winning_label]
        best_result = max(winning_results, key=lambda x: x['conf'])

        breakdown = ", ".join(
            [
                f"{label}: sum={label_conf_sums[label]:.3f}, "
                f"max={label_max_conf[label]:.3f}, count={label_counts[label]}"
                for label in label_conf_sums
            ]
        )

        logger.info(
            f"[ClassifierService] Weighted vote: {winning_label} "
            f"(norm={normalized_score:.3f}, margin={margin:.3f}, "
            f"best_conf={best_result['conf']:.3f}, "
            f"selected={len(selected_results)}/{len(valid_results)}, "
            f"total_time={total_batch_time:.1f}ms)"
        )
        logger.debug(f"[ClassifierService] Score breakdown: {breakdown}")

        if normalized_score < self.weighted_score_threshold and margin < self.weighted_margin_threshold:
            fallback = max(selected_results, key=lambda x: x['conf'])
            logger.warning(
                f"[ClassifierService] Winner below thresholds "
                f"(norm={normalized_score:.3f}, margin={margin:.3f}); "
                f"marking as Unknown."
            )
            return fallback['roi'], "Unknown", fallback['conf']

        return best_result['roi'], winning_label, best_result['conf']

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
