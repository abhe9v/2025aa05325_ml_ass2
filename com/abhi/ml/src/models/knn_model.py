"""
K-Nearest Neighbors Model
Author: Abhinav
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from com.abhi.ml.src.models.base_model import BaseMLModel
from com.abhi.ml.src.config.settings import KNNConfig


class KNNModel(BaseMLModel):
    """K-Nearest Neighbors classifier"""

    def __init__(self):
        super().__init__(model_name="kNN")

    def build_model(self) -> BaseEstimator:
        """
        Build KNN model

        Returns:
            Configured KNeighborsClassifier instance
        """
        return KNeighborsClassifier(
            n_neighbors=KNNConfig.N_NEIGHBORS,
            weights=KNNConfig.WEIGHTS,
            algorithm=KNNConfig.ALGORITHM
        )