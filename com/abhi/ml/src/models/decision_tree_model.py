"""
Decision Tree Model
Author: Abhinav
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from com.abhi.ml.src.models.base_model import BaseMLModel
from com.abhi.ml.src.config.settings import DecisionTreeConfig


class DecisionTreeModel(BaseMLModel):
    """Decision Tree classifier"""

    def __init__(self):
        super().__init__(model_name="Decision Tree")

    def build_model(self) -> BaseEstimator:
        """
        Build Decision Tree model

        Returns:
            Configured DecisionTreeClassifier instance
        """
        return DecisionTreeClassifier(
            criterion=DecisionTreeConfig.CRITERION,
            max_depth=DecisionTreeConfig.MAX_DEPTH,
            random_state=DecisionTreeConfig.RANDOM_STATE
        )