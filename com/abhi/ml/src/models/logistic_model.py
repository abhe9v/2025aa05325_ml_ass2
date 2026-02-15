"""
Logistic Regression Model
Author: Abhinav
"""

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from com.abhi.ml.src.models.base_model import BaseMLModel
from com.abhi.ml.src.config.settings import LogisticConfig


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression classifier"""

    def __init__(self):
        super().__init__(model_name="Logistic Regression")

    def build_model(self) -> BaseEstimator:
        """
        Build Logistic Regression model

        Returns:
            Configured LogisticRegression instance
        """
        return LogisticRegression(
            max_iter=LogisticConfig.MAX_ITER,
            solver=LogisticConfig.SOLVER,
            random_state=LogisticConfig.RANDOM_STATE
        )