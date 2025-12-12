"""
ClassifierService Module V2 - Enhanced for Production-Grade Accuracy.

Key V2 Improvements:
- Entropy-based uncertainty filtering for rejecting ambiguous predictions
- Class-specific confidence thresholds for handling class-imbalanced scenarios
- Enhanced logging for debugging and pattern detection
- Improved voting algorithm stability
"""

import logging
import math
import os
import time
from collections import Counter, defaultdict
from typing import Callable, List, Dict, Any, Tuple, Optional

import numpy as np
import cv2

from src.classifier.BaseClassifier import BaseClassifier
from src.utils.Utils import compute_phash
from src.utils.AppLogging import logger, structured_logger
from src.utils.PipelineMetrics import pipeline_metrics

ResultCallback = Callable[[int, Dict[str, Any]], None]


class ClassifierService:
    """
    Enhanced classifier service with voting, entropy filtering, and production-grade accuracy.
    
    V2 Features:
    - Dirichlet-EMA weighted voting for robust multi-candidate classification
    - Entropy-based uncertainty filtering to reject ambiguous predictions
    - Class-specific confidence thresholds for fine-grained control
    - Comprehensive structured logging for debugging
    """
    
    # Default class-specific confidence thresholds (can be overridden)
    # Classes that are harder to distinguish may need higher thresholds
    DEFAULT_CLASS_THRESHOLDS = {
        "Unknown": 0.5,  # Higher bar for unknown to reduce false negatives
    }
    
    # Entropy threshold for uncertainty filtering
    # Lower values = more strict (reject more uncertain predictions)
    # Max entropy for K classes is log(K), normalized entropy = entropy / log(K)
    DEFAULT_MAX_NORMALIZED_ENTROPY = 0.7  # Reject if normalized entropy > 0.7
    
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
                 weight_min: float = 1e-6,
                 # V2: Entropy-based filtering
                 use_entropy_filtering: bool = True,
                 max_normalized_entropy: float = 0.7,
                 # V2: Class-specific thresholds
                 class_confidence_thresholds: Optional[Dict[str, float]] = None):
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
        
        # V2: Entropy-based uncertainty filtering
        self.use_entropy_filtering = use_entropy_filtering
        self.max_normalized_entropy = max_normalized_entropy
        
        # V2: Class-specific confidence thresholds
        self.class_confidence_thresholds = class_confidence_thresholds or self.DEFAULT_CLASS_THRESHOLDS.copy()

        self.callbacks: List[ResultCallback] = []
        self.running = True

        # per-track state
        self.track_states: Dict[int, Dict[str, Any]] = {}
        
        # V2: Metrics tracking
        self._entropy_rejections = 0
        self._class_threshold_rejections = 0

        candidate_cap = self._get_candidate_cap()
        candidate_cap_desc = 'all' if candidate_cap is None else candidate_cap

        logger.info(
            f"[ClassifierService] Initialized V2: voting={use_voting}, "
            f"weighted_conf_threshold={voting_accept_norm_threshold}, "
            f"margin_threshold={voting_accept_margin}, "
            f"candidates_per_vote={candidate_cap_desc}, "
            f"alpha0={self.alpha0}, ema_beta={self.ema_beta}, "
            f"entropy_filtering={use_entropy_filtering}, max_entropy={max_normalized_entropy}"
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

    def _compute_entropy(self, proba: np.ndarray) -> float:
        """
        Compute Shannon entropy of a probability distribution.
        
        Entropy measures the uncertainty in the prediction.
        Higher entropy = more uncertain (spread across classes).
        Lower entropy = more certain (concentrated on one class).
        
        Returns:
            float: Shannon entropy in nats (natural log)
        """
        # Filter out zero probabilities to avoid log(0)
        proba = proba[proba > 0]
        if len(proba) == 0:
            return 0.0
        return -np.sum(proba * np.log(proba))
    
    def _compute_normalized_entropy(self, proba: np.ndarray) -> float:
        """
        Compute normalized entropy (0 to 1).
        
        Normalized entropy = entropy / max_entropy
        where max_entropy = log(K) for K classes.
        
        Returns:
            float: Normalized entropy between 0 (certain) and 1 (uniform/uncertain)
        """
        K = len(proba)
        if K <= 1:
            return 0.0
        
        entropy = self._compute_entropy(proba)
        max_entropy = math.log(K)  # Maximum entropy for uniform distribution
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _check_entropy_threshold(self, proba: np.ndarray, label: str) -> Tuple[bool, float]:
        """
        Check if the prediction entropy is below the threshold.
        
        Returns:
            Tuple[bool, float]: (is_acceptable, normalized_entropy)
        """
        normalized_entropy = self._compute_normalized_entropy(proba)
        
        # Prediction is acceptable if entropy is below threshold
        is_acceptable = normalized_entropy <= self.max_normalized_entropy
        
        if not is_acceptable:
            logger.debug(
                f"[ClassifierService] Entropy filter: normalized_entropy={normalized_entropy:.3f} > "
                f"threshold={self.max_normalized_entropy} for label={label}"
            )
            self._entropy_rejections += 1
        
        return is_acceptable, normalized_entropy
    
    def _check_class_threshold(self, label: str, confidence: float) -> bool:
        """
        Check if the confidence meets the class-specific threshold.
        
        Returns:
            bool: True if confidence is acceptable for this class
        """
        threshold = self.class_confidence_thresholds.get(label, self.min_confidence_threshold)
        
        is_acceptable = confidence >= threshold
        
        if not is_acceptable:
            logger.debug(
                f"[ClassifierService] Class threshold filter: conf={confidence:.3f} < "
                f"threshold={threshold} for class={label}"
            )
            self._class_threshold_rejections += 1
        
        return is_acceptable

    def _select_best_with_voting(self, track_id: int, candidates: List) -> Tuple[Optional[Any], str, float, Dict[str, Any]]:
        # (identical to the Dirichlet/EMA implementation provided earlier)
        # V2: Enhanced with entropy-based filtering and improved metadata
        # It returns (selected_roi, final_label, top_prob, metadata)
        # ------------------
        metadata = {}
        
        if not candidates:
            return None, "Unknown", 0.0, metadata

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
            metadata = {"all_unknown": True, "candidates_evaluated": len(candidates)}
            return best_unknown['roi'], "Unknown", best_unknown['conf'], metadata

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
        
        # V2: Compute entropy for uncertainty detection
        normalized_entropy = self._compute_normalized_entropy(ema_post)
        entropy_acceptable = True
        if self.use_entropy_filtering:
            entropy_acceptable, normalized_entropy = self._check_entropy_threshold(ema_post, top_label)

        strong_single = False
        for r, w, _ in zip(candidates_for_vote, weights, candidate_vectors):
            if r['conf'] >= self.best_conf_break and w >= (1.0 / len(weights)) * 0.5:
                strong_single = True
                break

        # V2: Add entropy check to acceptance criteria
        base_accepted = (normalized_score >= self.voting_accept_norm_threshold) or (margin >= self.voting_accept_margin) or strong_single
        accepted = base_accepted and entropy_acceptable
        
        # V2: Check class-specific threshold
        if accepted and not self._check_class_threshold(top_label, top_prob):
            accepted = False
        
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
            rejection_reasons = []
            if not base_accepted:
                rejection_reasons.append(
                    f"voting(norm={normalized_score:.2f}, margin={margin:.2f})"
                )
            if not entropy_acceptable:
                rejection_reasons.append(f"entropy={normalized_entropy:.2f}")
            logger.warning(
                f"[ClassifierService] Uncertain: {top_label} rejected - "
                f"{', '.join(rejection_reasons)}"
            )

        logger.info(
            f"[ClassifierService] Result: {final_label} "
            f"(conf={selected_conf:.3f}, norm={normalized_score:.2f}, "
            f"margin={margin:.2f}, entropy={normalized_entropy:.3f}, "
            f"n={len(candidates_for_vote)}/{len(valid_results)}, {total_batch_time:.1f}ms)"
        )

        label_scores = {lbl: float(alpha[i]) for i, lbl in enumerate(classes)}
        label_counts = Counter([r['label'] for r in candidates_for_vote])
        if logger.isEnabledFor(getattr(logger, "DEBUG", logging.DEBUG)):
            weight_dist = ", ".join([
                f"{label}: alpha={label_scores[label]:.3f}, count={label_counts[label]}"
                for label in classes
            ])
            logger.debug(f"[ClassifierService] Weighted distribution: {weight_dist}")

        # V2: Include comprehensive metadata for debugging
        metadata = {
            "normalized_score": normalized_score,
            "margin": margin,
            "normalized_entropy": normalized_entropy,
            "entropy_acceptable": entropy_acceptable,
            "base_accepted": base_accepted,
            "strong_single": strong_single,
            "candidates_evaluated": len(candidates),
            "candidates_voted": len(candidates_for_vote),
            "processing_time_ms": total_batch_time,
            "label_scores": label_scores,
        }

        return selected_roi, final_label, top_prob, metadata

    def _select_best_by_confidence(self, candidates: List) -> Tuple[Optional[Any], str, float, Dict[str, Any]]:
        """Select best candidate by confidence (fallback when voting not used)."""
        metadata = {"method": "confidence_only"}
        
        if not candidates:
            return None, "Unknown", 0.0, metadata

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

        metadata["candidates_evaluated"] = len(candidates)
        
        if best_roi is not None:
            # V2: Check class-specific threshold
            if not self._check_class_threshold(best_label, best_confidence):
                logger.warning(
                    f"[ClassifierService] Class threshold not met: {best_label} "
                    f"conf={best_confidence:.3f}"
                )
                return best_roi, "Unknown", best_confidence, metadata
            
            logger.info(f"[ClassifierService] Best: {best_label} (conf={best_confidence:.3f})")
            return best_roi, best_label, best_confidence, metadata
        else:
            return best_unknown_roi, "Unknown", best_unknown_conf, metadata

    def process(self, track_id: int, roi_input, context: Optional[Dict[str, Any]] = None):
        """
        Process classification request.
        
        V2 enhancements:
        - Entropy-based uncertainty filtering
        - Class-specific confidence thresholds
        - Structured logging for debugging
        """
        try:
            if isinstance(roi_input, list):
                candidates = roi_input
            else:
                candidates = [roi_input]

            logger.info(f"[ClassifierService] Track {track_id}: {len(candidates)} candidates")

            if not candidates:
                logger.error(f"[ClassifierService] Track {track_id}: Empty candidates!")
                return

            # V2: Methods now return metadata as 4th element
            used_voting = self.use_voting and len(candidates) >= 3
            if used_voting:
                best_roi, label, conf, metadata = self._select_best_with_voting(track_id, candidates)
            else:
                best_roi, label, conf, metadata = self._select_best_by_confidence(candidates)

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
                "metadata": metadata,  # V2: Include classification metadata
            }
            
            # Record classification metrics
            pipeline_metrics.record_classification(
                label, conf, len(candidates), used_voting
            )
            
            # V2: Structured logging for pattern detection
            structured_logger.classification_result(
                track_id=track_id,
                label=label,
                confidence=conf,
                candidates=len(candidates),
                used_voting=used_voting,
                entropy=metadata.get("normalized_entropy", 0.0),
                margin=metadata.get("margin", 0.0),
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