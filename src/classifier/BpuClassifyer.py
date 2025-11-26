import cv2
import numpy as np
from typing import Tuple, Dict
from src.classifier.BaseClassifier import BaseClassifier

# Import BPU Library safely
try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    try:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
    except ImportError:
        print("⚠️ Warning: hobot_dnn not found. BpuClassifier will fail.")
        dnn = None


class BpuClassifier(BaseClassifier):
    def __init__(self, model_path: str, class_names: Dict[int, str], input_size=(224, 224)):
        """
        Args:
            model_path: Path to .bin file.
            class_names: Dict mapping ID to Name.
            input_size: Model input size (usually 224x224 for Classifiers).
        """
        self._class_names = class_names
        self.input_h, self.input_w = input_size
        self.model = None

        if dnn:
            print(f"Loading BPU Classifier: {model_path}")
            self.model = dnn.load(model_path)

            # Debug: Print Input Shape
            # Note: We access this AFTER loading, unlike the previous broken code
            try:
                input_shape = self.model[0].inputs[0].properties.shape
                print(f"Classifier loaded. Expects input shape: {input_shape}")
            except Exception as e:
                print(f"Warning: Could not read model properties: {e}")
        else:
            print("BPU Classifier disabled due to missing library.")

    def load(self, model_path: str):
        # Wrapper if needed, but __init__ handles it
        if dnn and self.model is None:
            self.model = dnn.load(model_path)

    def predict(self, image) -> Tuple[str, float]:
        if self.model is None:
            return "Error", 0.0

        if image is None or image.size == 0:
            return "Unknown", 0.0

        # 1. Preprocess (Resize + NV12)
        # We manually convert to NV12 to avoid SegFaults
        input_tensor = self._preprocess(image)

        # 2. Inference
        t1 = cv2.getTickCount()
        outputs = self.model[0].forward(input_tensor)
        t2 = cv2.getTickCount()

        latency = (t2 - t1) * 1000 / cv2.getTickFrequency()
        print(f"BPU Classify Inference Time: {latency:.2f} ms")

        # 3. Post-Process
        # Output[0] is the probability vector (or logits)
        probs = outputs[0].buffer.flatten()

        # Find Max
        top_id = int(np.argmax(probs))
        confidence = float(probs[top_id])

        # Optional: If your model outputs raw logits (not 0-1), apply Softmax
        # Check if confidence is huge (like 15.4) or normal (0.95)
        # if confidence > 1.0 or confidence < 0.0:
        #    exp_scores = np.exp(probs - np.max(probs))
        #    probs = exp_scores / np.sum(exp_scores)
        #    top_id = int(np.argmax(probs))
        #    confidence = float(probs[top_id])

        label = self._class_names.get(top_id, "Unknown")

        return label, confidence

    def _preprocess(self, img):
        # 1. Resize
        # Standard Classification typically squashes image to 224x224
        resized = cv2.resize(img, (self.input_w, self.input_h))

        # 2. Convert BGR to NV12 (Required by BPU hardware)
        return self._bgr2nv12(resized)

    def _bgr2nv12(self, bgr_img):
        height, width = bgr_img.shape[:2]
        area = height * width

        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]

        # UV Packing logic
        uv_packed = yuv420p[area:].reshape((2, area // 4)).transpose((1, 0)).reshape((area // 2,))

        nv12 = np.zeros_like(yuv420p)
        nv12[:area] = y
        nv12[area:] = uv_packed

        return nv12