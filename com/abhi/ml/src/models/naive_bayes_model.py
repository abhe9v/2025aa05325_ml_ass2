"""
Naive Bayes Model
Author: Abhinav
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
from com.abhi.ml.src.models.base_model import BaseMLModel
from com.abhi.ml.src.config.settings import NaiveBayesConfig


class NaiveBayesModel(BaseMLModel):
    """Gaussian Naive Bayes classifier"""

    def __init__(self):
        super().__init__(model_name="Naive Bayes")

    def build_model(self) -> BaseEstimator:
        """
        Build Naive Bayes model

        Returns:
            Configured GaussianNB instance
        """
        return GaussianNB(
            var_smoothing=NaiveBayesConfig.VAR_SMOOTHING
        )