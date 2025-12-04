import os
import time
from typing import Callable, List, Dict, Any, Tuple, Optional

from src.classifier. BaseClassifier import BaseClassifier
from src. utils.Utils import compute_phash
from src. utils.AppLogging import logger

ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    def __init__(self,
                 classifier: BaseClassifier,
                 data_root: str = "data",
                 save_all_rois: bool = False,
                 min_confidence_threshold: float = 0.3):

        self.classifier = classifier
        self. data_root = data_root
        self.save_all_rois = save_all_rois
        self.min_confidence_threshold = min_confidence_threshold

        self.callbacks: List[ResultCallback] = []
        self.running = True

        logger.info(f"[ClassifierService] Initialized")

    def register_callback(self, callback: ResultCallback):
        self.callbacks.append(callback)

    def _classify_single(self, roi_image, idx: int = 0) -> Tuple[str, float]:
        """Classify a single ROI."""
        try:
            logger.debug(f"[ClassifierService] Classifying candidate {idx}...")
            label, conf = self.classifier. predict(roi_image)
            logger. debug(f"[ClassifierService] Candidate {idx}: {label} ({conf:.3f})")
            return label, conf
        except Exception as e:
            logger.error(f"[ClassifierService] Classification error: {e}")
            return "Unknown", 0.0

    def _select_best_candidate(self, candidates: List) -> Tuple[Optional[Any], str, float]:
        """Evaluate all candidates and select best one."""
        if not candidates:
            logger.warning("[ClassifierService] No candidates!")
            return None, "Unknown", 0.0

        best_roi = None
        best_label = "Unknown"
        best_confidence = 0.0

        best_unknown_roi = None
        best_unknown_conf = 0.0

        results = []

        logger.info(f"[ClassifierService] Evaluating {len(candidates)} candidates...")

        for idx, roi in enumerate(candidates):
            logger.debug(f"[ClassifierService] Candidate {idx}: shape={roi.shape if hasattr(roi, 'shape') else 'N/A'}")
            label, conf = self._classify_single(roi, idx)
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

        # Summary
        non_unknown = [(l, c) for _, l, c in results if l != "Unknown"]
        logger.info(f"[ClassifierService] {len(non_unknown)} valid, {len(results) - len(non_unknown)} unknown")

        if best_roi is not None:
            logger.info(f"[ClassifierService] BEST: {best_label} (conf={best_confidence:.3f})")
            return best_roi, best_label, best_confidence
        else:
            logger.warning(f"[ClassifierService] All Unknown! Best unknown conf={best_unknown_conf:.3f}")
            return best_unknown_roi, "Unknown", best_unknown_conf

    def process(self, track_id: int, roi_input):
        """Process classification request."""
        try:
            # Handle list vs single image
            if isinstance(roi_input, list):
                candidates = roi_input
                logger.info(f"[ClassifierService] Track {track_id}: {len(candidates)} candidates")
            else:
                candidates = [roi_input]
                logger.info(f"[ClassifierService] Track {track_id}: single ROI")

            if not candidates:
                logger.error(f"[ClassifierService] Track {track_id}: Empty candidates list!")
                return

            # Log first candidate info
            first = candidates[0]
            if hasattr(first, 'shape'):
                logger.debug(f"[ClassifierService] First candidate shape: {first. shape}")

            # Select best
            best_roi, label, conf = self._select_best_candidate(candidates)

            if best_roi is None:
                logger.error(f"[ClassifierService] Track {track_id}: No valid ROI!")
                return

            # Compute hash
            phash_obj = compute_phash(best_roi)
            phash_str = str(phash_obj)

            # Save logic
            if label == "Unknown":
                target_dir = os.path.join(self.data_root, "unknown", phash_str)
            else:
                target_dir = os. path.join(self.data_root, "classes", label)

            os.makedirs(target_dir, exist_ok=True)

            should_save = False
            existing_files = os.listdir(target_dir)

            if self. save_all_rois:
                should_save = True
            elif not existing_files:
                should_save = True

            image_path = None
            if should_save:
                timestamp = int(time. time())
                filename = f"{timestamp}_{track_id}. jpg"
                save_path = os. path.join(target_dir, filename)

                import cv2
                cv2.imwrite(save_path, best_roi)
                image_path = save_path
                logger.debug(f"[ClassifierService] Saved: {save_path}")
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

            logger.info(f"[ClassifierService] Track {track_id} DONE: {label} (conf={conf:.3f})")

            # Callbacks
            for cb in self.callbacks:
                try:
                    cb(track_id, result_data)
                except Exception as e:
                    logger.error(f"[ClassifierService] Callback error: {e}")

        except Exception as e:
            logger.error(f"[ClassifierService] Process error for track {track_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())