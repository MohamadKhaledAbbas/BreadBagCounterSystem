from abc import abstractmethod, ABC


class BaseClassifier(ABC):
    """Abstract base class for classification models."""
    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, image) -> str:
        """Returns the predicted class label as a string."""
        pass
