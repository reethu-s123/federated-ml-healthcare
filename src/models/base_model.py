"""Base model class for all disease prediction models."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for disease prediction models."""

    def __init__(self, model_name: str, input_features: int):
        """Initialize base model.

        Args:
            model_name: Name of the model
            input_features: Number of input features
        """
        self.model_name = model_name
        self.input_features = input_features
        self.model = None
        self.is_trained = False

    @abstractmethod
    def build(self) -> None:
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10) -> Dict[str, Any]:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs

        Returns:
            Training history
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        pass

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Model parameters
        """
        return {}

    def set_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set model parameters.

        Args:
            parameters: Parameters to set
        """
        pass
