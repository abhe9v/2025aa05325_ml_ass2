"""
Main training orchestrator
Author: Abhinav
"""

import pandas as pd
from typing import List, Dict
from com.abhi.ml.src.data.loader import DataLoader
from com.abhi.ml.src.data.preprocessor import DataPreprocessor
from com.abhi.ml.src.models.logistic_model import LogisticRegressionModel
from com.abhi.ml.src.models.decision_tree_model import DecisionTreeModel
from com.abhi.ml.src.models.knn_model import KNNModel
from com.abhi.ml.src.models.naive_bayes_model import NaiveBayesModel
from com.abhi.ml.src.models.random_forest_model import RandomForestModel
from com.abhi.ml.src.models.xgboost_model import XGBoostModel
from com.abhi.ml.src.utils.file_handler import FileHandler
from com.abhi.ml.src.utils.logger import get_logger
from com.abhi.ml.src.config.settings import (
    TEST_DATA_FILE, RESULTS_FILE, SCALER_FILE, DATASET_INFO_FILE
)

logger = get_logger(__name__)


class ModelTrainingPipeline:
    """Main pipeline for training all models"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.file_handler = FileHandler()
        self.models = []
        self.results = []

    def initialize_models(self) -> None:
        """Initialize all 6 models"""
        logger.info("Initializing all models...")

        self.models = [
            LogisticRegressionModel(),
            DecisionTreeModel(),
            KNNModel(),
            NaiveBayesModel(),
            RandomForestModel(),
            XGBoostModel()
        ]

        logger.info(f"Initialized {len(self.models)} models")

    def load_and_prepare_data(self):
        """Load and prepare dataset"""
        # Load data
        X, y, dataset_info = self.data_loader.load_breast_cancer_data()

        # Split data
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y)

        # Preprocess
        X_train_scaled, X_test_scaled = self.preprocessor.fit_transform(X_train, X_test)

        # Save test data
        test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_data['target'] = y_test.values
        self.file_handler.save_csv(test_data, TEST_DATA_FILE)

        # Save scaler and dataset info
        scaler = self.preprocessor.get_scaler()
        self.file_handler.save_pickle(scaler, SCALER_FILE)
        self.file_handler.save_pickle(dataset_info, DATASET_INFO_FILE)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_all_models(self, X_train, X_test, y_train, y_test) -> None:
        """Train and evaluate all models"""

        print("\n" + "=" * 80)
        print("TRAINING ALL MODELS")
        print("=" * 80)

        for model in self.models:
            print(f"\n{'=' * 80}")
            print(f"Model: {model.model_name}")
            print(f"{'=' * 80}")

            # Train
            model.train(X_train, y_train)

            # Predict
            y_pred, y_pred_proba = model.predict(X_test)

            # Evaluate
            metrics = model.evaluate(y_test, y_pred, y_pred_proba)
            self.results.append(metrics)

            # Save model
            model.save()

        # Save results
        results_df = pd.DataFrame(self.results)
        self.file_handler.save_csv(results_df, RESULTS_FILE)

        # Print summary
        self.print_summary(results_df)

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print final results summary"""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("\n" + "=" * 80)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults saved to: {RESULTS_FILE}")
        print(f"Test data saved to: {TEST_DATA_FILE}")
        print(f"Models saved to: resources/models/")

    def run(self) -> None:
        """Execute complete training pipeline"""
        try:
            # Initialize models
            self.initialize_models()

            # Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()

            # Train all models
            self.train_all_models(X_train, X_test, y_train, y_test)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Entry point for training"""
    print("=" * 80)
    print("BREAST CANCER CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)

    pipeline = ModelTrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()