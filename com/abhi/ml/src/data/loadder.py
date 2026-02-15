"""
Data loading module
Author: Abhinav
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
from com.abhi.ml.src.config.settings import RANDOM_STATE, TEST_SIZE
from com.abhi.ml.src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles data loading operations"""

    def __init__(self):
        self.dataset_info = {}

    def load_breast_cancer_data(self) -> Tuple[pd.DataFrame, pd.Series, dict]:
        """
        Load breast cancer dataset from sklearn

        Returns:
            Tuple of (features DataFrame, target Series, dataset info dict)
        """
        logger.info("Loading Breast Cancer Wisconsin Dataset...")

        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')

        self.dataset_info = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_names': list(data.feature_names),
            'target_names': list(data.target_names),
            'description': data.DESCR
        }

        logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Class distribution - Malignant: {(y == 0).sum()}, Benign: {(y == 1).sum()}")

        return X, y, self.dataset_info

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train and test sets

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data: train/test = {1 - TEST_SIZE}/{TEST_SIZE}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test