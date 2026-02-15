"""
Data preprocessing module
Author: Abhinav
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Tuple
from com.abhi.ml.src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Handles data preprocessing operations"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler on training data and transform both train and test

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (scaled X_train, scaled X_test)
        """
        logger.info("Scaling features using StandardScaler...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Feature scaling completed")

        return X_train_scaled, X_test_scaled

    def get_scaler(self) -> StandardScaler:
        """Return fitted scaler"""
        return self.scaler