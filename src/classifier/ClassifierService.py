import logging
import os
import time
from collections import Counter, defaultdict
from typing import Callable, List, Dict, Any, Tuple, Optional

import numpy as np
import cv2

from src.classifier.BaseClassifier import BaseClassifier
from src.utils.Utils import compute_phash
from src.utils.AppLogging import logger
from src.utils.PipelineMetrics import pipeline_metrics

ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    def __init__(self,
                 classifier: BaseClassifier,
                 data_root: str = "data",
                 save_all_rois: bool = False,
                 min_confidence_threshold: float = 0.3,
                 use_voting: bool = True,
                 voting_top_k: Optional[int] = None,
                 voting_accept_norm_threshold: float = 0.4,
                 voting_accept_margin: float = 0.15,
                 # Dirichlet / EMA params
                 alpha0: float = 0.5,
                 ema_beta: float = 0.3,
                 sharpness_s0: float = 100.0,
                 sharpness_scale: float = 20.0,
                 best_conf_break: float = 0.99,
                 weight_min: float = 1e-6):
        self.classifier = classifier
        self.data_root = data_root
        self.save_all_rois = save_all_rois
        self.min_confidence_threshold = min_confidence_threshold
        self.use_voting = use_voting
        # Default None processes all candidates; set a positive value to restore legacy top-k behavior.
        self.voting_top_k = voting_top_k
        self.voting_accept_norm_threshold = voting_accept_norm_threshold
        self.voting_accept_margin = voting_accept_margin

        # New algorithm params
        self.alpha0 = alpha0
        self.ema_beta = ema_beta
        self.sharpness_s0 = sharpness_s0
        self.sharpness_scale = sharpness_scale
        self.best_conf_break = best_conf_break
        self.weight_min = weight_min

        self.callbacks: List[ResultCallback] = []
        self.running = True

        # per-track state
        self.track_states: Dict[int, Dict[str, Any]] = {}

        candidate_cap = self._get_candidate_cap()
        candidate_cap_desc = 'all' if candidate_cap is None else candidate_cap

        logger.info(
            f"[ClassifierService] Initialized: voting={use_voting}, "
            f"weighted_conf_threshold={voting_accept_norm_threshold}, "
            f"margin_threshold={voting_accept_margin}, "
            f"candidates_per_vote={candidate_cap_desc}, "
            f"alpha0={self.alpha0}, ema_beta={self.ema_beta}"
        )

    def _get_candidate_cap(self) -> Optional[int]:
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

    def _compute_sharpness(self, img) -> float:
        try:
            if img is None:
                return 0.0
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            var = float(lap.var())
            return var
        except Exception:
            return 0.0

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    # ... (Dirichlet/EMA voting unchanged) ...

    def _select_best_with_voting(self, track_id: int, candidates: List) -> Tuple[Optional[Any], str, float]:
        # (identical to the Dirichlet/EMA implementation provided earlier)
        # For brevity, the body remains the same as in the previously shared version.
        # It returns (selected_roi, final_label, top_prob)
        # ------------------
        if not candidates:
            return None, "Unknown", 0.0

        results = []
        batch_start = time.perf_counter()
        logger.info(f"[ClassifierService] Classifying {len(candidates)} candidates...")

        for idx, roi in enumerate(candidates):
            label, conf = self._classify_single(roi, idx)
            sharpness = self._compute_sharpness(roi)
            results.append({'roi': roi, 'label': label, 'conf': conf, 'sharpness': sharpness})

        batch_end = time.perf_counter()
        total_batch_time = (batch_end - batch_start) * 1000

        valid_results = [r for r in results if r['label'] != "Unknown"]
        if not valid_results:
            logger.warning("[ClassifierService] All candidates Unknown!")
            best_unknown = max(results, key=lambda x: x['conf'])
            return best_unknown['roi'], "Unknown", best_unknown['conf']

        valid_results.sort(key=lambda x: x['conf'], reverse=True)
        candidate_cap = self._get_candidate_cap()
        candidates_for_vote = valid_results if candidate_cap is None else valid_results[:candidate_cap]

        if hasattr(self.classifier, "classes_") and getattr(self.classifier, "classes_") is not None:
            classes = list(getattr(self.classifier, "classes_"))
        else:
            classes = sorted({r['label'] for r in candidates_for_vote})

        K = len(classes)
        label_to_index = {lbl: idx for idx, lbl in enumerate(classes)}
        proba_available = hasattr(self.classifier, "predict_proba") and callable(getattr(self.classifier, "predict_proba"))

        candidate_vectors = []
        for r in candidates_for_vote:
            if proba_available:
                try:
                    probs = self.classifier.predict_proba(r['roi'])
                    if hasattr(self.classifier, "classes_") and getattr(self.classifier, "classes_") is not None:
                        clf_classes = list(getattr(self.classifier, "classes_"))
                        proba_vec = np.zeros(K, dtype=float)
                        for i_c, cname in enumerate(clf_classes):
                            if cname in label_to_index:
                                proba_vec[label_to_index[cname]] = float(probs[i_c])
                        if proba_vec.sum() <= 0:
                            proba_vec = np.ones(K, dtype=float) / float(K)
                        else:
                            proba_vec = proba_vec / proba_vec.sum()
                    else:
                        proba_vec = np.array(probs, dtype=float)
                        if proba_vec.size != K:
                            proba_vec = np.zeros(K, dtype=float)
                            proba_vec[label_to_index.get(r['label'], 0)] = r['conf']
                            remaining = max(0.0, 1.0 - r['conf'])
                            if K > 1:
                                proba_vec += remaining / (K - 1)
                except Exception:
                    proba_vec = np.zeros(K, dtype=float)
                    proba_vec[label_to_index.get(r['label'], 0)] = r['conf']
                    remaining = max(0.0, 1.0 - r['conf'])
                    if K > 1:
                        proba_vec += remaining / (K - 1)
            else:
                proba_vec = np.zeros(K, dtype=float)
                proba_vec[label_to_index.get(r['label'], 0)] = r['conf']
                remaining = max(0.0, 1.0 - r['conf'])
                if K > 1:
                    proba_vec += remaining / (K - 1)

            if proba_vec.sum() <= 0:
                proba_vec = np.ones(K, dtype=float) / float(K)
            else:
                proba_vec = proba_vec / proba_vec.sum()

            candidate_vectors.append(proba_vec)

        weights = []
        for r in candidates_for_vote:
            sharp = float(r.get('sharpness', 0.0))
            sharp_w = float(self._sigmoid((sharp - self.sharpness_s0) / max(1.0, self.sharpness_scale)))
            conf_w = float(r['conf'])
            w = max(conf_w * sharp_w, self.weight_min)
            weights.append(w)

        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / float(len(weights))

        alpha = np.full(K, float(self.alpha0), dtype=float)
        for vec, w in zip(candidate_vectors, weights):
            alpha += w * vec

        posterior = alpha / float(alpha.sum())

        ts = time.time()
        track_state = self.track_states.get(track_id)
        if track_state is None:
            ema_post = posterior.copy()
        else:
            prev_post = track_state.get('posterior')
            if prev_post is not None and prev_post.size == posterior.size:
                ema_post = (1.0 - self.ema_beta) * prev_post + self.ema_beta * posterior
            else:
                ema_post = posterior.copy()

        self.track_states[track_id] = {
            'posterior': ema_post,
            'labels': classes,
            'last_update': ts
        }

        top_idx = int(np.argmax(ema_post))
        top_label = classes[top_idx]
        top_prob = float(ema_post[top_idx])
        second_prob = float(np.partition(ema_post, -2)[-2]) if K > 1 else 0.0
        margin = top_prob - second_prob
        normalized_score = top_prob

        strong_single = False
        for r, w, _ in zip(candidates_for_vote, weights, candidate_vectors):
            if r['conf'] >= self.best_conf_break and w >= (1.0 / len(weights)) * 0.5:
                strong_single = True
                break

        accepted = (normalized_score >= self.voting_accept_norm_threshold) or (margin >= self.voting_accept_margin) or strong_single
        final_label = top_label if accepted else "Unknown"

        if accepted:
            matching = [r for r in candidates_for_vote if r['label'] == final_label]
            if matching:
                selected_result = max(matching, key=lambda x: x['conf'])
            else:
                selected_result = max(candidates_for_vote, key=lambda x: x['conf'])
        else:
            selected_result = max(candidates_for_vote, key=lambda x: x['conf'])

        selected_roi = selected_result['roi']
        selected_conf = selected_result['conf']

        if not accepted:
            logger.warning(
                f"[ClassifierService] Weighted voting uncertain - winner={top_label} "
                f"(norm={normalized_score:.2f}, margin={margin:.2f}); marking as Unknown "
                f"using thresholds (norm>={self.voting_accept_norm_threshold}, margin>={self.voting_accept_margin})"
            )

        logger.info(
            f"[ClassifierService] Weighted voting result: {final_label} "
            f"(winner={top_label}, best_conf={selected_conf:.3f}, "
            f"norm={normalized_score:.2f}, margin={margin:.2f}, "
            f"candidates_used={len(candidates_for_vote)}, total_candidates={len(valid_results)}, "
            f"time={total_batch_time:.1f}ms)"
        )

        label_scores = {lbl: float(alpha[i]) for i, lbl in enumerate(classes)}
        label_counts = Counter([r['label'] for r in candidates_for_vote])
        if logger.isEnabledFor(getattr(logger, "DEBUG", logging.DEBUG)):
            weight_dist = ", ".join([
                f"{label}: alpha={label_scores[label]:.3f}, count={label_counts[label]}"
                for label in classes
            ])
            logger.debug(f"[ClassifierService] Weighted distribution: {weight_dist}")

        return selected_roi, final_label, top_prob

    def _select_best_by_confidence(self, candidates: List) -> Tuple[Optional[Any], str, float]:
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

    def process(self, track_id: int, roi_input, context: Optional[Dict[str, Any]] = None):
        """Process classification request."""
        try:
            if isinstance(roi_input, list):
                candidates = roi_input
            else:
                candidates = [roi_input]

            logger.info(f"[ClassifierService] Track {track_id}: {len(candidates)} candidates")

            if not candidates:
                logger.error(f"[ClassifierService] Track {track_id}: Empty candidates!")
                return

            if self.use_voting and len(candidates) >= 3:
                best_roi, label, conf = self._select_best_with_voting(track_id, candidates)
            else:
                best_roi, label, conf = self._select_best_by_confidence(candidates)

            if best_roi is None:
                logger.error(f"[ClassifierService] Track {track_id}: No valid ROI!")
                return

            if conf < self.min_confidence_threshold:
                logger.warning(
                    f"[ClassifierService] Track {track_id}: Low confidence "
                    f"{label} ({conf:.3f} < {self.min_confidence_threshold})"
                )

            phash_obj = compute_phash(best_roi)
            phash_str = str(phash_obj)

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

                cv2.imwrite(save_path, best_roi)
                image_path = save_path
            elif existing_files:
                image_path = os.path.join(target_dir, existing_files[0])

            result_data = {
                "label": label,
                "phash": phash_str,
                "image_path": image_path,
                "confidence": conf,
                "candidates_evaluated": len(candidates),
                "context": context,  # pass-through for downstream snapshot saving
            }
            
            # Record classification metrics
            used_voting = self.use_voting and len(candidates) >= 3
            pipeline_metrics.record_classification(
                label, conf, len(candidates), used_voting
            )

            logger.info(
                f"[ClassifierService] Track {track_id} DONE: {label} "
                f"(conf={conf:.3f}, candidates={len(candidates)})"
            )

            for cb in self.callbacks:
                try:
                    cb(track_id, result_data)
                except Exception as e:
                    logger.error(f"[ClassifierService] Callback error: {e}")

        except Exception as e:
            logger.error(f"[ClassifierService] Process error: {e}")
            import traceback
            logger.error(traceback.format_exc())