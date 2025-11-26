import cv2
import numpy as np
from typing import Dict, List
from scipy.special import softmax
from src.detection.BaseDetection import BaseDetector
from src.utils.PerformanceChecker import run_with_timing

# Import BPU Library
try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    try:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
    except ImportError:
        print("⚠️ Warning: hobot_dnn not found. This will fail on RDK X5.")
        dnn = None


class BpuDetector(BaseDetector):
    def __init__(self, model_path: str, class_names: Dict[int, str], input_size=(640, 640)):
        self._class_names = class_names
        self.input_h, self.input_w = input_size
        # --- PRE-ALLOCATE MEMORY BUFFERS (Optimization) ---
        # We allocate the NV12 buffer ONCE here, instead of every frame
        self.nv12_buffer = np.zeros((self.input_h * self.input_w * 3 // 2,), dtype=np.uint8)
        self.area = self.input_h * self.input_w

        if dnn:
            print(f"Loading BPU model: {model_path}")
            self.quantize_model = dnn.load(model_path)
        else:
            self.quantize_model = None

        # Configs matching your model training
        self.classes_num = len(class_names)
        self.nms_thres = 0.45
        self.score_thres = 0.25
        self.reg = 16
        self.strides = [8, 16, 32]

        # Pre-calculate constants
        self.conf_thres_raw = -np.log(1 / self.score_thres - 1)
        self.weights_static = np.array([i for i in range(self.reg)]).astype(np.float32)[np.newaxis, np.newaxis, :]

        # Generate Anchors
        self.grids = []
        for stride in self.strides:
            grid_h, grid_w = self.input_h // stride, self.input_w // stride
            self.grids.append(np.stack([
                np.tile(np.linspace(0.5, grid_h - 0.5, grid_h), reps=grid_h),
                np.repeat(np.arange(0.5, grid_w + 0.5, 1), grid_w)
            ], axis=0).transpose(1, 0))

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    def predict(self, frame):
        if self.quantize_model is None:
            return [BpuResultWrapper([], [], [])]

        # 1. Preprocess (Resize + BGR2NV12)
        input_tensor, x_scale, y_scale, x_shift, y_shift = run_with_timing("detection pre-process", self._preprocess, frame)

        # 2. Forward
        outputs = run_with_timing("forward", self.quantize_model[0].forward, input_tensor)

        # 3. Convert to Numpy
        output_arrays = [out.buffer for out in outputs]

        # 4. Post-Process (Decode)
        results = run_with_timing("detection post-process",self._postprocess, output_arrays, x_scale, y_scale, x_shift, y_shift, frame.shape)

        # 5. Format Results
        boxes, scores, class_ids = [], [], []
        for cid, score, x1, y1, x2, y2 in results:
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cid)

        return [BpuResultWrapper(np.array(boxes), np.array(scores), np.array(class_ids))]

    def _preprocess(self, img):
        """
        Optimized Preprocessing:
        1. Uses INTER_NEAREST (Faster than Linear)
        2. Uses Pre-allocated buffer (Saves Memory Alloc)
        3. Uses Slicing for NV12 Interleaving (Faster than Reshape/Transpose)
        """
        img_h, img_w = img.shape[:2]
        x_scale = min(1.0 * self.input_h / img_h, 1.0 * self.input_w / img_w)
        y_scale = x_scale

        new_w = int(img_w * x_scale)
        new_h = int(img_h * y_scale)

        x_shift = (self.input_w - new_w) // 2
        y_shift = (self.input_h - new_h) // 2

        # Optimization 1: INTER_NEAREST is significantly faster for detection
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        padded = cv2.copyMakeBorder(resized, y_shift, self.input_h - new_h - y_shift,
                                    x_shift, self.input_w - new_w - x_shift,
                                    cv2.BORDER_CONSTANT, value=127)

        # Optimization 2 & 3: Efficient NV12 Conversion
        # Convert to YUV I420 (Planar: YYYYY...UU...VV...)
        yuv_i420 = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)

        # Flatten allows us to treat it as a 1D buffer
        yuv_flat = yuv_i420.reshape(-1)

        # Copy Y plane directly
        self.nv12_buffer[:self.area] = yuv_flat[:self.area]

        # Interleave UV (NV12: UVUVUV...)
        # U is at: yuv_flat[area : area + area/4]
        # V is at: yuv_flat[area + area/4 : ]
        u_start = self.area
        v_start = self.area + (self.area // 4)

        # Magic Slicing: Assign U to even indices, V to odd indices
        self.nv12_buffer[self.area::2] = yuv_flat[u_start: v_start]
        self.nv12_buffer[self.area + 1::2] = yuv_flat[v_start:]

        return self.nv12_buffer, x_scale, y_scale, x_shift, y_shift
    def _postprocess(self, outputs, x_scale, y_scale, x_shift, y_shift, orig_shape):
        # YOLOv8 Headless Decoding Logic
        clses = [outputs[0].reshape(-1, self.classes_num), outputs[2].reshape(-1, self.classes_num),
                 outputs[4].reshape(-1, self.classes_num)]
        bboxes = [outputs[1].reshape(-1, self.reg * 4), outputs[3].reshape(-1, self.reg * 4),
                  outputs[5].reshape(-1, self.reg * 4)]

        dbboxes, ids, scores = [], [], []

        for cls, bbox, stride, grid in zip(clses, bboxes, self.strides, self.grids):
            max_scores = np.max(cls, axis=1)
            valid_mask = max_scores >= self.conf_thres_raw
            if not np.any(valid_mask): continue

            ids.append(np.argmax(cls[valid_mask, :], axis=1))
            scores.append(1 / (1 + np.exp(-max_scores[valid_mask])))

            pred_dist = softmax(bbox[valid_mask].reshape(-1, 4, self.reg), axis=2)
            ltrb = np.sum(pred_dist * self.weights_static, axis=2)

            grid_val = grid[valid_mask]
            x1y1 = grid_val - ltrb[:, 0:2]
            x2y2 = grid_val + ltrb[:, 2:4]
            dbboxes.append(np.hstack([x1y1, x2y2]) * stride)

        if not dbboxes: return []

        dbboxes = np.concatenate(dbboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        ids = np.concatenate(ids, axis=0)

        # NMS
        xywh = dbboxes.copy()
        xywh[:, 2:4] = xywh[:, 2:4] - xywh[:, 0:2]

        final_results = []
        for i in range(self.classes_num):
            mask = ids == i
            if not np.any(mask): continue
            indices = cv2.dnn.NMSBoxes(xywh[mask].tolist(), scores[mask].tolist(), self.score_thres, self.nms_thres)
            if len(indices) > 0:
                indices = indices.flatten()
                selected_boxes = dbboxes[mask][indices]
                selected_scores = scores[mask][indices]

                for box, score in zip(selected_boxes, selected_scores):
                    x1, y1, x2, y2 = box
                    x1 = max(0, min(orig_shape[1], (x1 - x_shift) / x_scale))
                    y1 = max(0, min(orig_shape[0], (y1 - y_shift) / y_scale))
                    x2 = max(0, min(orig_shape[1], (x2 - x_shift) / x_scale))
                    y2 = max(0, min(orig_shape[0], (y2 - y_shift) / y_scale))
                    final_results.append((i, score, x1, y1, x2, y2))
        return final_results


# --- HELPER CLASSES (Mimicking Ultralytics API) ---

class BpuResultWrapper:
    def __init__(self, boxes, scores, class_ids):
        self.boxes = BoxWrapper(boxes, scores, class_ids)


class BoxWrapper:
    def __init__(self, boxes, scores, class_ids):
        # We wrap the data in TensorAdapter to support .cpu().numpy()
        self.xyxy = TensorAdapter(boxes)
        self.conf = TensorAdapter(scores)
        self.cls = TensorAdapter(class_ids)

    def __len__(self):
        # This fixes your "object has no len()" error
        return len(self.xyxy)


class TensorAdapter:
    """
    A fake Tensor class that allows code using .cpu().numpy() to work
    without needing PyTorch installed on the board.
    """

    def __init__(self, data):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data

    def cpu(self):
        return self  # We are already on CPU

    def numpy(self):
        return self.data  # Return the underlying numpy array

    def __len__(self):
        return len(self.data)














    def _postprocess(self, outputs, x_scale, y_scale, x_shift, y_shift, orig_shape):
        # YOLOv8 Headless Decoding Logic
        clses = [outputs[0].reshape(-1, self.classes_num), outputs[2].reshape(-1, self.classes_num),
                 outputs[4].reshape(-1, self.classes_num)]
        bboxes = [outputs[1].reshape(-1, self.reg * 4), outputs[3].reshape(-1, self.reg * 4),
                  outputs[5].reshape(-1, self.reg * 4)]

        dbboxes, ids, scores = [], [], []

        for cls, bbox, stride, grid in zip(clses, bboxes, self.strides, self.grids):
            max_scores = np.max(cls, axis=1)
            valid_mask = max_scores >= self.conf_thres_raw
            if not np.any(valid_mask): continue

            ids.append(np.argmax(cls[valid_mask, :], axis=1))
            scores.append(1 / (1 + np.exp(-max_scores[valid_mask])))

            pred_dist = softmax(bbox[valid_mask].reshape(-1, 4, self.reg), axis=2)
            ltrb = np.sum(pred_dist * self.weights_static, axis=2)

            grid_val = grid[valid_mask]
            x1y1 = grid_val - ltrb[:, 0:2]
            x2y2 = grid_val + ltrb[:, 2:4]
            dbboxes.append(np.hstack([x1y1, x2y2]) * stride)

        if not dbboxes: return []

        dbboxes = np.concatenate(dbboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        ids = np.concatenate(ids, axis=0)

        # NMS
        xywh = dbboxes.copy()
        xywh[:, 2:4] = xywh[:, 2:4] - xywh[:, 0:2]

        final_results = []
        for i in range(self.classes_num):
            mask = ids == i
            if not np.any(mask): continue
            indices = cv2.dnn.NMSBoxes(xywh[mask].tolist(), scores[mask].tolist(), self.score_thres, self.nms_thres)
            if len(indices) > 0:
                indices = indices.flatten()
                selected_boxes = dbboxes[mask][indices]
                selected_scores = scores[mask][indices]

                for box, score in zip(selected_boxes, selected_scores):
                    x1, y1, x2, y2 = box
                    x1 = max(0, min(orig_shape[1], (x1 - x_shift) / x_scale))
                    y1 = max(0, min(orig_shape[0], (y1 - y_shift) / y_scale))
                    x2 = max(0, min(orig_shape[1], (x2 - x_shift) / x_scale))
                    y2 = max(0, min(orig_shape[0], (y2 - y_shift) / y_scale))
                    final_results.append((i, score, x1, y1, x2, y2))
        return final_results
