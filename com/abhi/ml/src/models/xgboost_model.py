"""
XGBoost/Gradient Boosting Model
Author: Abhinav
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
from com.abhi.ml.src.models.base_model import BaseMLModel
from com.abhi.ml.src.config.settings import XGBoostConfig


class XGBoostModel(BaseMLModel):
    """Gradient Boosting (XGBoost) ensemble classifier"""

    def __init__(self):
        super().__init__(model_name="XGBoost")

    def build_model(self) -> BaseEstimator:
        """
        Build Gradient Boosting model

        Returns:
            Configured GradientBoostingClassifier instance
        """
        return GradientBoostingClassifier(
            n_estimators=XGBoostConfig.N_ESTIMATORS,
            learning_rate=XGBoostConfig.LEARNING_RATE,
            max_depth=XGBoostConfig.MAX_DEPTH,
            random_state=XGBoostConfig.RANDOM_STATE
        )