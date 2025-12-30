import cv2
import numpy as np
from typing import Tuple, Dict
from src.classifier.BaseClassifier import BaseClassifier
from src.utils.AppLogging import logger
from src.config.tracking_config import tracking_config

# Import BPU Library safely
try:
    from hobot_dnn import pyeasy_dnn as dnn

    logger.info("[BpuClassifier] Using hobot_dnn")
except ImportError:
    try:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

        logger.info("[BpuClassifier] Using hobot_dnn_rdkx5")
    except ImportError:
        logger.error("[BpuClassifier] hobot_dnn not found!")
        dnn = None


class BpuClassifier(BaseClassifier):
    def __init__(self, model_path: str, class_names: Dict[int, str], input_size=(224, 224)):
        self._class_names = class_names
        self.input_h, self.input_w = input_size
        self.model = None
        
        # Phase 3 Optimization: Pre-allocate NV12 buffer (saves memory allocation per call)
        self.area = self.input_h * self.input_w
        self.nv12_buffer = np.zeros((self.area * 3 // 2,), dtype=np.uint8)

        logger.info(f"[BpuClassifier] Class names: {class_names}")

        if dnn:
            logger.info(f"[BpuClassifier] Loading model: {model_path}")
            self.model = dnn.load(model_path)

            try:
                input_shape = self.model[0].inputs[0].properties.shape
                logger.info(f"[BpuClassifier] Model loaded.  Input shape: {input_shape}")
            except Exception as e:
                logger.warning(f"[BpuClassifier] Could not read model properties: {e}")
        else:
            logger.error("[BpuClassifier] Disabled - missing library")

    def load(self, model_path: str):
        if dnn and self.model is None:
            self.model = dnn.load(model_path)

    def predict(self, image) -> Tuple[str, float]:
        # Validate model
        if self.model is None:
            logger.error("[BpuClassifier] Model not loaded!")
            return "Unknown", 0.0

        # Validate input
        if image is None:
            logger.error("[BpuClassifier] Image is None!")
            return "Unknown", 0.0

        if not isinstance(image, np.ndarray):
            logger.error(f"[BpuClassifier] Image is not ndarray: {type(image)}")
            return "Unknown", 0.0

        if image.size == 0:
            logger.error("[BpuClassifier] Image is empty!")
            return "Unknown", 0.0

        if len(image.shape) != 3:
            logger.error(f"[BpuClassifier] Invalid image shape: {image.shape}")
            return "Unknown", 0.0

        logger.debug(f"[BpuClassifier] Input image shape: {image.shape}, dtype: {image.dtype}")

        try:
            # Phase 3: Add detailed timing logs for CPU operations
            # 1. Preprocess
            t_preprocess_start = cv2.getTickCount()
            input_tensor = self._preprocess(image)
            t_preprocess_end = cv2.getTickCount()
            preprocess_time_ms = (t_preprocess_end - t_preprocess_start) * 1000 / cv2.getTickFrequency()

            # 2. Inference (BPU)
            t_inference_start = cv2.getTickCount()
            outputs = self.model[0].forward(input_tensor)
            t_inference_end = cv2.getTickCount()
            inference_time_ms = (t_inference_end - t_inference_start) * 1000 / cv2.getTickFrequency()

            # 3. Post-Process
            t_postprocess_start = cv2.getTickCount()
            probs = outputs[0].buffer.flatten()

            # Apply softmax if needed (raw logits instead of probabilities)
            if probs.max() > 1.0 or probs.min() < 0.0:
                exp_scores = np.exp(probs - np.max(probs))
                probs = exp_scores / np.sum(exp_scores)

            # Find max
            top_id = int(np.argmax(probs))
            confidence = float(probs[top_id])
            label = self._class_names.get(top_id, "Unknown")
            t_postprocess_end = cv2.getTickCount()
            postprocess_time_ms = (t_postprocess_end - t_postprocess_start) * 1000 / cv2.getTickFrequency()
            
            # Log timing every 50 classifications
            if not hasattr(self, '_classify_counter'):
                self._classify_counter = 0
                self._classify_timing_sum = {'preprocess': 0, 'inference': 0, 'postprocess': 0}
            
            self._classify_counter += 1
            self._classify_timing_sum['preprocess'] += preprocess_time_ms
            self._classify_timing_sum['inference'] += inference_time_ms
            self._classify_timing_sum['postprocess'] += postprocess_time_ms
            
            if self._classify_counter % 50 == 0:
                avg_preprocess = self._classify_timing_sum['preprocess'] / 50
                avg_inference = self._classify_timing_sum['inference'] / 50
                avg_postprocess = self._classify_timing_sum['postprocess'] / 50
                total_avg = avg_preprocess + avg_inference + avg_postprocess
                
                logger.info(
                    f"[BpuClassifier] Avg timing (50 classifications): "
                    f"preprocess={avg_preprocess:.2f}ms ({avg_preprocess/total_avg*100:.1f}%), "
                    f"inference={avg_inference:.2f}ms ({avg_inference/total_avg*100:.1f}%), "
                    f"postprocess={avg_postprocess:.2f}ms ({avg_postprocess/total_avg*100:.1f}%), "
                    f"total={total_avg:.2f}ms"
                )
                # Reset counters
                self._classify_timing_sum = {'preprocess': 0, 'inference': 0, 'postprocess': 0}

            return label, confidence

        except Exception as e:
            logger.error(f"[BpuClassifier] Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "Unknown", 0.0

    def predict_probs(self, image) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class label, confidence, and full probability vector.
        
        Returns normalized probability distribution over all known classes.
        This is required for trust-weighted log-evidence accumulation.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (label, confidence, probs_dict) where:
            - label: predicted class name
            - confidence: probability of predicted class
            - probs_dict: {class_name: probability} for all classes
                         (normalized, non-negative, sums to ~1.0)
        """
        # Validate model
        if self.model is None:
            logger.error("[BpuClassifier] Model not loaded!")
            return "Unknown", 0.0, {"Unknown": 1.0}
        
        # Validate input
        if image is None:
            logger.error("[BpuClassifier] Image is None!")
            return "Unknown", 0.0, {"Unknown": 1.0}
        
        if not isinstance(image, np.ndarray):
            logger.error(f"[BpuClassifier] Image is not ndarray: {type(image)}")
            return "Unknown", 0.0, {"Unknown": 1.0}
        
        if image.size == 0:
            logger.error("[BpuClassifier] Image is empty!")
            return "Unknown", 0.0, {"Unknown": 1.0}
        
        if len(image.shape) != 3:
            logger.error(f"[BpuClassifier] Invalid image shape: {image.shape}")
            return "Unknown", 0.0, {"Unknown": 1.0}
        
        try:
            # 1. Preprocess
            input_tensor = self._preprocess(image)
            
            # 2. Inference
            outputs = self.model[0].forward(input_tensor)
            
            # 3. Post-Process - Get probability vector
            probs = outputs[0].buffer.flatten()
            
            # Apply softmax if needed (raw logits instead of probabilities)
            if probs.max() > 1.0 or probs.min() < 0.0:
                exp_scores = np.exp(probs - np.max(probs))
                probs = exp_scores / np.sum(exp_scores)
            
            # Ensure probabilities are normalized (handle numerical errors)
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs = probs / probs_sum
            
            # Find top prediction
            top_id = int(np.argmax(probs))
            confidence = float(probs[top_id])
            label = self._class_names.get(top_id, "Unknown")
            
            # Build probability dictionary for all known classes
            probs_dict = {}
            for class_id, class_name in self._class_names.items():
                if class_id < len(probs):
                    probs_dict[class_name] = float(probs[class_id])
            
            # Ensure all probabilities are present and valid
            # Handle case where not all class_ids are in probs (defensive)
            if not probs_dict:
                # Fallback: single class with confidence
                probs_dict = {label: confidence}
            
            return label, confidence, probs_dict
            
        except Exception as e:
            logger.error(f"[BpuClassifier] predict_probs error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "Unknown", 0.0, {"Unknown": 1.0}

    def _preprocess(self, img):
        # Phase 2 Optimization: Use INTER_NEAREST for faster resize (acceptable for classification)
        resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)
        return self._bgr2nv12(resized)

    def _bgr2nv12(self, bgr_img):
        """
        Phase 3 Optimization: Optimized NV12 conversion using pre-allocated buffer.
        """
        height, width = bgr_img.shape[:2]

        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((self.area * 3 // 2,))
        
        # Copy Y plane directly to pre-allocated buffer
        self.nv12_buffer[:self.area] = yuv420p[:self.area]
        
        # Interleave UV directly into buffer
        u_start = self.area
        v_start = self.area + (self.area // 4)
        self.nv12_buffer[self.area::2] = yuv420p[u_start: v_start]
        self.nv12_buffer[self.area + 1::2] = yuv420p[v_start:]

        return self.nv12_buffer

    def _validate_image(self, image, idx: int) -> bool:
        """
        Validate that an image is suitable for classification.
        
        Args:
            image: Image to validate
            idx: Index for logging
            
        Returns:
            True if image is valid, False otherwise
        """
        if image is None:
            logger.debug(f"[BpuClassifier] Batch: Image at index {idx} is None, skipping")
            return False
        if not isinstance(image, np.ndarray):
            logger.debug(f"[BpuClassifier] Batch: Image at index {idx} is not ndarray, skipping")
            return False
        if image.size == 0:
            logger.debug(f"[BpuClassifier] Batch: Image at index {idx} is empty, skipping")
            return False
        if len(image.shape) != 3:
            logger.debug(f"[BpuClassifier] Batch: Image at index {idx} has invalid shape {image.shape}, skipping")
            return False
        return True

    def _preprocess_batch(self, images):
        """
        Preprocess a batch of images for inference.
        
        Args:
            images: List of numpy arrays (ROI images)
            
        Returns:
            Tuple of (batch_tensors, valid_indices) where valid_indices maps
            batch position to original image index
        """
        batch_tensors = []
        valid_indices = []
        
        for idx, image in enumerate(images):
            if not self._validate_image(image, idx):
                continue
            
            try:
                # Preprocess and create a copy (since _preprocess reuses buffer)
                input_tensor = self._preprocess(image)
                batch_tensors.append(input_tensor.copy())
                valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"[BpuClassifier] Batch: Preprocess failed at index {idx}: {e}")
                continue
        
        return batch_tensors, valid_indices

    def _apply_softmax(self, probs):
        """
        Apply softmax to probabilities if needed.
        
        Args:
            probs: Raw probability/logit array
            
        Returns:
            Normalized probability array
        """
        if probs.max() > 1.0 or probs.min() < 0.0:
            exp_scores = np.exp(probs - np.max(probs))
            return exp_scores / np.sum(exp_scores)
        return probs

    def _run_batch_inference(self, batch_tensors, use_true_batch=True):
        """
        Run batch inference on preprocessed tensors.
        
        Attempts true batch inference first, falls back to sequential if needed.
        
        Args:
            batch_tensors: List of preprocessed NV12 tensors
            use_true_batch: Whether to attempt true batch inference
            
        Returns:
            Tuple of (batch_probs, true_batch_used) where batch_probs is a list
            of probability arrays (or None for failed inferences)
        """
        batch_probs = []
        true_batch_used = False
        
        if use_true_batch and len(batch_tensors) > 1:
            try:
                # True batch inference: Stack tensors and pass to BPU in single call
                batch_input = np.stack(batch_tensors, axis=0)
                
                # Attempt true batch forward pass
                batch_outputs = self.model[0].forward(batch_input)
                
                # Check if output is batched
                if batch_outputs and len(batch_outputs) > 0:
                    first_output = batch_outputs[0].buffer
                    
                    if len(first_output.shape) > 1 and first_output.shape[0] == len(batch_tensors):
                        # Outputs are batched - extract probabilities for each image
                        true_batch_used = True
                        for i in range(len(batch_tensors)):
                            probs = first_output[i].flatten()
                            probs = self._apply_softmax(probs)
                            batch_probs.append(probs)
                    else:
                        logger.debug(
                            f"[BpuClassifier] Model output shape {first_output.shape} doesn't match "
                            f"batch_size={len(batch_tensors)}, falling back to sequential"
                        )
                        
            except Exception as e:
                logger.debug(f"[BpuClassifier] True batch inference attempt failed: {e}")
        
        # Sequential fallback if true batch wasn't used or failed
        if not true_batch_used:
            batch_probs = []
            for tensor in batch_tensors:
                try:
                    outputs = self.model[0].forward(tensor)
                    probs = outputs[0].buffer.flatten()
                    probs = self._apply_softmax(probs)
                    batch_probs.append(probs)
                except Exception as e:
                    logger.debug(f"[BpuClassifier] Sequential inference failed: {e}")
                    batch_probs.append(None)
        
        return batch_probs, true_batch_used

    def _update_batch_timing_stats(self, batch_size, preprocess_time_ms, inference_time_ms, 
                                    postprocess_time_ms, total_time_ms, true_batch_used):
        """
        Update and log batch timing statistics.
        
        Args:
            batch_size: Number of images in this batch
            preprocess_time_ms: Preprocessing time in milliseconds
            inference_time_ms: Inference time in milliseconds
            postprocess_time_ms: Postprocessing time in milliseconds
            total_time_ms: Total batch time in milliseconds
            true_batch_used: Whether true batch inference was used
        """
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
            self._batch_timing_sum = {
                'preprocess': 0, 'inference': 0, 'postprocess': 0,
                'total': 0, 'batch_sizes': [], 'true_batch_count': 0
            }
        
        self._batch_counter += 1
        self._batch_timing_sum['preprocess'] += preprocess_time_ms
        self._batch_timing_sum['inference'] += inference_time_ms
        self._batch_timing_sum['postprocess'] += postprocess_time_ms
        self._batch_timing_sum['total'] += total_time_ms
        self._batch_timing_sum['batch_sizes'].append(batch_size)
        if true_batch_used:
            self._batch_timing_sum['true_batch_count'] += 1
        
        # Log every 20 batches
        if self._batch_counter % 20 == 0:
            total_images = sum(self._batch_timing_sum['batch_sizes'])
            avg_batch_size = total_images / len(self._batch_timing_sum['batch_sizes'])
            avg_total = self._batch_timing_sum['total'] / self._batch_counter
            avg_per_image = avg_total / avg_batch_size if avg_batch_size > 0 else 0
            true_batch_pct = (self._batch_timing_sum['true_batch_count'] / self._batch_counter) * 100
            
            logger.info(
                f"[BpuClassifier] Batch stats (20 batches): "
                f"avg_batch_size={avg_batch_size:.1f}, "
                f"avg_time_per_image={avg_per_image:.2f}ms, "
                f"avg_batch_time={avg_total:.2f}ms, "
                f"true_batch_used={true_batch_pct:.1f}%"
            )
            
            # Reset counters
            self._batch_timing_sum = {
                'preprocess': 0, 'inference': 0, 'postprocess': 0,
                'total': 0, 'batch_sizes': [], 'true_batch_count': 0
            }

    def predict_batch(self, images, use_true_batch=None):
        """
        V7: Batch classification for multiple ROI images.
        
        Processes multiple images in batches for improved throughput when classifying
        multiple ROIs (e.g., during track classification).
        
        Args:
            images: List of numpy arrays (ROI images) to classify
            use_true_batch: If True, attempt true batch inference via BPU.
                           If False or BPU batch fails, use sequential processing.
                           If None, uses tracking_config.classification_true_batch_enabled.
            
        Returns:
            List of tuples (label, confidence), one per image
        """
        # Use config value if not explicitly specified
        if use_true_batch is None:
            use_true_batch = tracking_config.classification_true_batch_enabled
        
        if self.model is None:
            return [("Unknown", 0.0) for _ in images]
        
        batch_size = len(images)
        if batch_size == 0:
            return []
        
        # Single image - use regular predict
        if batch_size == 1:
            return [self.predict(images[0])]
        
        t_batch_start = cv2.getTickCount()
        
        # 1. Preprocess all images
        t_preprocess_start = cv2.getTickCount()
        batch_tensors, valid_indices = self._preprocess_batch(images)
        t_preprocess_end = cv2.getTickCount()
        preprocess_time_ms = (t_preprocess_end - t_preprocess_start) * 1000 / cv2.getTickFrequency()
        
        # If no valid images, return all Unknown
        if len(batch_tensors) == 0:
            return [("Unknown", 0.0) for _ in images]
        
        # 2. Batch inference
        t_inference_start = cv2.getTickCount()
        batch_probs, true_batch_used = self._run_batch_inference(batch_tensors, use_true_batch)
        t_inference_end = cv2.getTickCount()
        inference_time_ms = (t_inference_end - t_inference_start) * 1000 / cv2.getTickFrequency()
        
        # 3. Post-process and build results
        t_postprocess_start = cv2.getTickCount()
        
        # Initialize results with Unknown for all images
        results = [("Unknown", 0.0) for _ in images]
        
        # Map batch results back to original indices
        for batch_idx, orig_idx in enumerate(valid_indices):
            if batch_idx < len(batch_probs) and batch_probs[batch_idx] is not None:
                probs = batch_probs[batch_idx]
                top_id = int(np.argmax(probs))
                confidence = float(probs[top_id])
                label = self._class_names.get(top_id, "Unknown")
                results[orig_idx] = (label, confidence)
        
        t_postprocess_end = cv2.getTickCount()
        postprocess_time_ms = (t_postprocess_end - t_postprocess_start) * 1000 / cv2.getTickFrequency()
        
        # 4. Log batch timing metrics
        t_batch_end = cv2.getTickCount()
        total_batch_time_ms = (t_batch_end - t_batch_start) * 1000 / cv2.getTickFrequency()
        
        self._update_batch_timing_stats(
            batch_size, preprocess_time_ms, inference_time_ms,
            postprocess_time_ms, total_batch_time_ms, true_batch_used
        )
        
        return results

    def predict_batch_probs(self, images, use_true_batch=None):
        """
        V7: Batch classification with full probability vectors.
        
        Similar to predict_batch but returns full probability vectors for each image,
        required for trust-weighted log-evidence accumulation.
        
        Args:
            images: List of numpy arrays (ROI images) to classify
            use_true_batch: If True, attempt true batch inference via BPU.
                           If None, uses tracking_config.classification_true_batch_enabled.
            
        Returns:
            List of tuples (label, confidence, probs_dict), one per image
        """
        # Use config value if not explicitly specified
        if use_true_batch is None:
            use_true_batch = tracking_config.classification_true_batch_enabled
        
        if self.model is None:
            return [("Unknown", 0.0, {"Unknown": 1.0}) for _ in images]
        
        batch_size = len(images)
        if batch_size == 0:
            return []
        
        # Single image - use regular predict_probs
        if batch_size == 1:
            return [self.predict_probs(images[0])]
        
        # 1. Preprocess all images (using shared helper)
        batch_tensors, valid_indices = self._preprocess_batch(images)
        
        if len(batch_tensors) == 0:
            return [("Unknown", 0.0, {"Unknown": 1.0}) for _ in images]
        
        # 2. Batch inference (using shared helper)
        batch_probs, _ = self._run_batch_inference(batch_tensors, use_true_batch)
        
        # 3. Build results with probability dictionaries
        results = [("Unknown", 0.0, {"Unknown": 1.0}) for _ in images]
        
        for batch_idx, orig_idx in enumerate(valid_indices):
            if batch_idx < len(batch_probs) and batch_probs[batch_idx] is not None:
                probs = batch_probs[batch_idx]
                
                # Ensure probabilities are normalized
                probs_sum = np.sum(probs)
                if probs_sum > 0:
                    probs = probs / probs_sum
                
                top_id = int(np.argmax(probs))
                confidence = float(probs[top_id])
                label = self._class_names.get(top_id, "Unknown")
                
                # Build probability dictionary
                probs_dict = {}
                for class_id, class_name in self._class_names.items():
                    if class_id < len(probs):
                        probs_dict[class_name] = float(probs[class_id])
                
                if not probs_dict:
                    probs_dict = {label: confidence}
                
                results[orig_idx] = (label, confidence, probs_dict)
        
        return results