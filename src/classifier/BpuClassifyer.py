import cv2
import numpy as np
from typing import Tuple, Dict
from src.classifier.BaseClassifier import BaseClassifier
from src.utils.AppLogging import logger

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
            # 1. Preprocess
            input_tensor = self._preprocess(image)
            logger.debug(f"[BpuClassifier] Preprocessed tensor shape: {input_tensor.shape}")

            # 2.  Inference
            t1 = cv2.getTickCount()
            outputs = self.model[0].forward(input_tensor)
            t2 = cv2.getTickCount()

            latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
            logger.debug(f"[BpuClassifier] Inference time: {latency:.2f}ms")

            # 3. Post-Process
            probs = outputs[0].buffer.flatten()

            logger.debug(f"[BpuClassifier] Raw probs shape: {probs.shape}, range: [{probs.min():.4f}, {probs.max():.4f}]")

            # Apply softmax if needed (raw logits instead of probabilities)
            if probs.max() > 1.0 or probs.min() < 0.0:
                logger.debug("[BpuClassifier] Applying softmax")
                exp_scores = np.exp(probs - np.max(probs))
                probs = exp_scores / np.sum(exp_scores)

            # Find max
            top_id = int(np.argmax(probs))
            confidence = float(probs[top_id])

            label = self._class_names.get(top_id, "Unknown")

            logger.info(f"[BpuClassifier] Result: {label} (conf={confidence:.3f}, id={top_id})")

            return label, confidence

        except Exception as e:
            logger.error(f"[BpuClassifier] Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "Unknown", 0.0

    def _preprocess(self, img):
        resized = cv2.resize(img, (self.input_w, self.input_h))
        return self._bgr2nv12(resized)

    def _bgr2nv12(self, bgr_img):
        height, width = bgr_img.shape[:2]
        area = height * width

        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_packed = yuv420p[area:].reshape((2, area // 4)).transpose((1, 0)).reshape((area // 2,))

        nv12 = np.zeros_like(yuv420p)
        nv12[:area] = y
        nv12[area:] = uv_packed

        return nv12