from typing import Any, Dict, Tuple

from ultralytics import YOLO

from src.classifier.BaseClassifier import BaseClassifier


class UltralyticsClassifier(BaseClassifier):
    """Concrete implementation using Ultralytics YOLO for classification."""
    
    def __init__(self, model_path: str, class_names: Dict[int, str] = None):
        """Initialize Ultralytics YOLO classifier.
        
        Args:
            model_path: Path to YOLO classification model (.pt or .onnx)
            class_names: Optional custom class names. If not provided, uses model's built-in names.
        """
        self.model = YOLO(model_path)
        self._class_names = class_names

    def load(self, model_path: str):
        """Load a new model (for compatibility with BaseClassifier interface).
        
        Note: This resets class_names to use the model's built-in names.
        """
        self.model = YOLO(model_path)
        # Reset class names to None so the new model's names are used
        self._class_names = None

    def predict(self, image) -> Tuple[str, float]:
        """Predict class label and confidence for the given image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (label, confidence)
        """
        results = self.model(image, verbose=False)

        label = "Unknown"
        conf = 0.0

        # Case A: It is a Classification Model (has .probs)
        if results[0].probs is not None:
            top1_idx = results[0].probs.top1
            conf = float(results[0].probs.top1conf)  # Extract confidence
            
            # Use custom class names if provided, otherwise use model's names
            if self._class_names is not None:
                label = self._class_names.get(top1_idx, "Unknown")
            else:
                label = results[0].names[top1_idx]

        # Case B: It is a Detection Model used as Classifier (has .boxes)
        elif len(results[0].boxes) > 0:
            # Take the highest confidence box
            conf = float(results[0].boxes.conf[0])
            cls_id = int(results[0].boxes.cls[0])
            
            # Use custom class names if provided, otherwise use model's names
            if self._class_names is not None:
                label = self._class_names.get(cls_id, "Unknown")
            else:
                label = results[0].names[cls_id]

        return label, conf
    
    def predict_probs(self, image) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class label, confidence, and full probability vector.
        
        This method extracts the full softmax probability vector from
        the YOLO classification model.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (label, confidence, probs_dict) where probs_dict is
            {class_name: probability} for all classes
        """
        results = self.model(image, verbose=False)
        
        label = "Unknown"
        conf = 0.0
        probs_dict = {}
        
        # Case A: It is a Classification Model (has .probs)
        if results[0].probs is not None:
            top1_idx = results[0].probs.top1
            conf = float(results[0].probs.top1conf)
            
            # Get full probability vector
            probs_tensor = results[0].probs.data
            
            # Use custom class names if provided, otherwise use model's names
            if self._class_names is not None:
                label = self._class_names.get(top1_idx, "Unknown")
                for idx, prob in enumerate(probs_tensor):
                    class_name = self._class_names.get(idx, f"class_{idx}")
                    probs_dict[class_name] = float(prob)
            else:
                label = results[0].names[top1_idx]
                for idx, prob in enumerate(probs_tensor):
                    class_name = results[0].names[idx]
                    probs_dict[class_name] = float(prob)
        
        # Case B: It is a Detection Model used as Classifier (has .boxes)
        elif len(results[0].boxes) > 0:
            # Take the highest confidence box
            conf = float(results[0].boxes.conf[0])
            cls_id = int(results[0].boxes.cls[0])
            
            # Use custom class names if provided, otherwise use model's names
            if self._class_names is not None:
                label = self._class_names.get(cls_id, "Unknown")
            else:
                label = results[0].names[cls_id]
            
            # For detection models, we don't have a probability vector
            # Return the detection confidence as the probability for the detected class
            probs_dict[label] = conf
        
        else:
            # No valid results - return empty probs
            probs_dict["Unknown"] = 0.0
        
        return label, conf, probs_dict