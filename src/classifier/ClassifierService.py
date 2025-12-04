# src/classifier/ClassifierService.py
import os
import time
from typing import Callable, List, Dict, Any, Tuple, Optional

from src.classifier.BaseClassifier import BaseClassifier
from src.utils.Utils import compute_phash
from src.utils.AppLogging import logger

# Callback type
ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    def __init__(self,
                 classifier: BaseClassifier,
                 data_root: str = "data",
                 save_all_rois: bool = False,
                 min_confidence_threshold: float = 0.3):

        self.classifier = classifier
        self.data_root = data_root
        self.save_all_rois = save_all_rois
        self.min_confidence_threshold = min_confidence_threshold

        self.callbacks: List[ResultCallback] = []
        self.running = True

        logger.info(
            f"[ClassifierService] Initialized with data_root={data_root}, "
            f"save_all_rois={save_all_rois}, min_confidence={min_confidence_threshold}"
        )

    def register_callback(self, callback: ResultCallback):
        self.callbacks.append(callback)
        logger.debug(f"[ClassifierService] Registered callback (total: {len(self.callbacks)})")

    def _classify_single(self, roi_image) -> Tuple[str, float]:
        """Classify a single ROI and return (label, confidence)."""
        try:
            label, conf = self.classifier.predict(roi_image)
            logger.debug(f"[ClassifierService] Single ROI classified: {label} ({conf:.3f})")
            return label, conf
        except Exception as e:
            logger.error(f"[ClassifierService] Classification failed: {e}")
            return "Unknown", 0.0

    def _select_best_candidate(self, candidates: List) -> Tuple[Optional[Any], str, float]:
        """
        Evaluate all candidate ROIs and select the one with highest confidence.
        Returns: (best_roi, best_label, best_confidence)
        """
        if not candidates:
            logger.warning("[ClassifierService] No candidates to evaluate")
            return None, "Unknown", 0.0

        best_roi = None
        best_label = "Unknown"
        best_confidence = 0.0

        best_unknown_roi = None
        best_unknown_conf = 0.0

        results = []

        logger.debug(f"[ClassifierService] Evaluating {len(candidates)} candidates...")

        for idx, roi in enumerate(candidates):
            label, conf = self._classify_single(roi)
            results.append((roi, label, conf))

            if label == "Unknown":
                if conf > best_unknown_conf:
                    best_unknown_roi = roi
                    best_unknown_conf = conf
            else:
                if conf > best_confidence:
                    best_roi = roi
                    best_label = label
                    best_confidence = conf

        # Log detailed results at debug level
        logger.debug("[ClassifierService] Candidate evaluation results:")
        for i, (_, label, conf) in enumerate(results):
            logger.debug(f"  [{i + 1}/{len(results)}] {label}: {conf:.3f}")

        # Log summary statistics
        non_unknown = [(l, c) for _, l, c in results if l != "Unknown"]
        unknown_count = len(results) - len(non_unknown)

        logger.debug(
            f"[ClassifierService] Summary: {len(non_unknown)} classified, "
            f"{unknown_count} unknown"
        )

        if best_roi is not None:
            logger.info(
                f"[ClassifierService] Selected: {best_label} "
                f"(confidence={best_confidence:.3f}, from {len(candidates)} candidates)"
            )
            return best_roi, best_label, best_confidence
        else:
            logger.info(
                f"[ClassifierService] All candidates Unknown "
                f"(best_conf={best_unknown_conf:.3f})"
            )
            return best_unknown_roi, "Unknown", best_unknown_conf

    def process(self, track_id: int, roi_input):
        """
        Process classification request.
        roi_input can be:
          - A single ROI image (backward compatible)
          - A list of candidate ROI images (new multi-candidate mode)
        """
        try:
            # Handle both single ROI and list of candidates
            if isinstance(roi_input, list):
                candidates = roi_input
                logger.info(
                    f"[ClassifierService] Processing track {track_id}: "
                    f"{len(candidates)} candidates"
                )
            else:
                candidates = [roi_input]
                logger.debug(
                    f"[ClassifierService] Processing track {track_id}: single ROI"
                )

            # Select best candidate using classifier confidence
            best_roi, label, conf = self._select_best_candidate(candidates)

            if best_roi is None:
                logger.warning(
                    f"[ClassifierService] Track {track_id}: No valid ROI found, skipping"
                )
                return

            # Log low confidence predictions
            if conf < self.min_confidence_threshold:
                logger.warning(
                    f"[ClassifierService] Track {track_id}: Low confidence - "
                    f"{label} ({conf:.3f} < {self.min_confidence_threshold})"
                )

            # Compute Hash
            phash_obj = compute_phash(best_roi)
            phash_str = str(phash_obj)
            logger.debug(f"[ClassifierService] Track {track_id}: pHash={phash_str}")

            # Determine save directory
            if label == "Unknown":
                target_dir = os.path.join(self.data_root, "unknown", phash_str)
            else:
                target_dir = os.path.join(self.data_root, "classes", label)

            os.makedirs(target_dir, exist_ok=True)

            # Smart Saving Logic
            should_save = False
            existing_files = os.listdir(target_dir)

            if self.save_all_rois:
                should_save = True
                logger.debug(
                    f"[ClassifierService] Track {track_id}: Saving (save_all_rois=True)"
                )
            elif not existing_files:
                should_save = True
                logger.debug(
                    f"[ClassifierService] Track {track_id}: Saving (first thumbnail)"
                )

            image_path = None
            if should_save:
                timestamp = int(time.time())
                filename = f"{timestamp}_{track_id}. jpg"
                save_path = os.path.join(target_dir, filename)

                import cv2
                cv2.imwrite(save_path, best_roi)
                image_path = save_path
                logger.debug(f"[ClassifierService] Track {track_id}: Saved to {save_path}")
            elif existing_files:
                image_path = os.path.join(target_dir, existing_files[0])
                logger.debug(
                    f"[ClassifierService] Track {track_id}: Using existing {image_path}"
                )

            # Bundle Result
            result_data = {
                "label": label,
                "phash": phash_str,
                "image_path": image_path,
                "confidence": conf,
                "candidates_evaluated": len(candidates)
            }

            # Log final result at INFO level
            logger.info(
                f"[ClassifierService] Track {track_id} COMPLETE: "
                f"{label} (conf={conf:.3f}, candidates={len(candidates)})"
            )

            for cb in self.callbacks:
                try:
                    cb(track_id, result_data)
                except Exception as cb_error:
                    logger.error(
                        f"[ClassifierService] Callback error for track {track_id}: {cb_error}"
                    )

        except Exception as e:
            logger.error(f"[ClassifierService] Error processing track {track_id}: {e}")
            import traceback
            logger.debug(f"[ClassifierService] Traceback:\n{traceback.format_exc()}")