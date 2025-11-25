from abc import abstractmethod, ABC
from typing import Any


class BaseClassifier(ABC):
    """Abstract base class for classification models."""
    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, image) -> tuple[str | Any, float]:
        """Returns the predicted class label as a string."""
        pass
