"""
Random Forest Model
Author: Abhinav
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from com.abhi.ml.src.models.base_model import BaseMLModel
from com.abhi.ml.src.config.settings import RandomForestConfig


class RandomForestModel(BaseMLModel):
    """Random Forest ensemble classifier"""

    def __init__(self):
        super().__init__(model_name="Random Forest")

    def build_model(self) -> BaseEstimator:
        """
        Build Random Forest model

        Returns:
            Configured RandomForestClassifier instance
        """
        return RandomForestClassifier(
            n_estimators=RandomForestConfig.N_ESTIMATORS,
            criterion=RandomForestConfig.CRITERION,
            max_depth=RandomForestConfig.MAX_DEPTH,
            random_state=RandomForestConfig.RANDOM_STATE
        )