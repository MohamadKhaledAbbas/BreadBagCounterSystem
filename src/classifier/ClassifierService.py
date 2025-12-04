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

        # Sort by confidence and take top K
        valid_results.sort(key=lambda x: x['conf'], reverse=True)
        top_k = valid_results[:self.voting_top_k]

        # Voting: count label occurrences among top K
        label_votes = Counter(r['label'] for r in top_k)
        winning_label, vote_count = label_votes.most_common(1)[0]

        # Get the highest confidence ROI with the winning label
        winning_results = [r for r in top_k if r['label'] == winning_label]
        best_result = max(winning_results, key=lambda x: x['conf'])

        # Calculate voting confidence (votes / total top_k)
        voting_confidence = vote_count / len(top_k)

        logger.info(
            f"[ClassifierService] Voting result: {winning_label} "
            f"({vote_count}/{len(top_k)} votes, conf={best_result['conf']:.3f}, "
            f"voting_conf={voting_confidence:.2f}, time={total_batch_time:.1f}ms)"
        )

        # Log vote distribution
        vote_dist = ", ".join([f"{label}: {count}" for label, count in label_votes.items()])
        logger.debug(f"[ClassifierService] Vote distribution: {vote_dist}")

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
                filename = f"{timestamp}_{track_id}. jpg"
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