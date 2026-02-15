"""
Base model class
Author: Abhinav
"""

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
from com.abhi.ml.src.evaluation.metrics import ModelEvaluator
from com.abhi.ml.src.utils.file_handler import FileHandler
from com.abhi.ml.src.utils.logger import get_logger
from com.abhi.ml.src.config.settings import MODELS_DIR

logger = get_logger(__name__)


class BaseMLModel(ABC):
    """Abstract base class for all ML models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: BaseEstimator = None
        self.evaluator = ModelEvaluator()
        self.file_handler = FileHandler()
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def build_model(self) -> BaseEstimator:
        """Build and return the model instance"""
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name}...")

        if self.model is None:
            self.model = self.build_model()

        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training completed")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions

        Args:
            X: Features to predict

        Returns:
            Tuple of (predictions, prediction probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        return y_pred, y_pred_proba

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        self.metrics = self.evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        self.metrics['Model'] = self.model_name

        # Print metrics and confusion matrix
        self.evaluator.print_metrics(self.metrics, self.model_name)
        cm = self.evaluator.get_confusion_matrix(y_true, y_pred)
        self.evaluator.print_confusion_matrix(cm)

        return self.metrics

    def save(self) -> None:
        """Save trained model to disk"""
        filename = f"{self.model_name.lower().replace(' ', '_')}_model.pkl"
        filepath = MODELS_DIR / filename
        self.file_handler.save_pickle(self.model, filepath)

    def load(self) -> None:
        """Load trained model from disk"""
        filename = f"{self.model_name.lower().replace(' ', '_')}_model.pkl"
        filepath = MODELS_DIR / filename
        self.model = self.file_handler.load_pickle(filepath)

    def get_metrics(self) -> Dict[str, float]:
        """Return calculated metrics"""
        return self.metrics