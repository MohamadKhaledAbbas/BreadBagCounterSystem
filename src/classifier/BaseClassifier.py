from abc import abstractmethod, ABC
from typing import Any, Dict, Tuple, List


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

    def predict_batch(self, images) -> List[Tuple[str, float]]:
        """
        Batch classification for multiple images.
        
        This method is optional for subclasses. The default implementation
        processes images sequentially using predict().
        
        Subclasses should override this method to provide optimized batch
        inference when available.
        
        Args:
            images: List of numpy arrays (images) to classify
            
        Returns:
            List of tuples (label, confidence), one per image
        """
        # Default implementation: sequential processing
        return [self.predict(img) for img in images]

    def predict_batch_probs(self, images) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Batch classification with full probability vectors.
        
        This method is optional for subclasses. The default implementation
        processes images sequentially using predict_probs().
        
        Subclasses should override this method to provide optimized batch
        inference when available.
        
        Args:
            images: List of numpy arrays (images) to classify
            
        Returns:
            List of tuples (label, confidence, probs_dict), one per image
        """
        # Default implementation: sequential processing
        return [self.predict_probs(img) for img in images]
