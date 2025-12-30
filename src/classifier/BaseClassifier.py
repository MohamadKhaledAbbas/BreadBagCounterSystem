from abc import abstractmethod, ABC
from typing import Any, Dict, Tuple


class BaseClassifier(ABC):
    """Abstract base class for classification models."""
    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, image) -> tuple[str | Any, float]:
        """Returns the predicted class label as a string."""
        pass
    
    def predict_probs(self, image) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class label, confidence, and full probability vector.
        
        This method is optional for subclasses. The default implementation
        returns the top-1 prediction with a uniform probability vector.
        
        Subclasses should override this method to return the actual
        probability vector from the model.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (label, confidence, probs_dict) where probs_dict is
            {class_name: probability} for all classes
        """
        # Default implementation: call predict and return uniform probs
        label, conf = self.predict(image)
        # Return single-class probs dict (fallback behavior)
        return label, conf, {label: conf}
