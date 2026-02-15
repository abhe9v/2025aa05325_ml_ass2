"""
Model evaluation metrics
Author: Abhinav
"""

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import numpy as np
from typing import Dict
from com.abhi.ml.src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Calculates and manages model evaluation metrics"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate all required metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'AUC': round(roc_auc_score(y_true, y_pred_proba), 4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'Recall': round(recall_score(y_true, y_pred), 4),
            'F1': round(f1_score(y_true, y_pred), 4),
            'MCC': round(matthews_corrcoef(y_true, y_pred), 4)
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str) -> None:
        """Print metrics in formatted way"""
        print(f"\n{model_name} Performance:")
        print("-" * 50)
        for metric, value in metrics.items():
            if metric == 'Model':
                continue
            print(f"  {metric:12s}: {value:.4f}")

    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def print_confusion_matrix(cm: np.ndarray) -> None:
        """Print confusion matrix"""
        print("\nConfusion Matrix:")
        print(cm)