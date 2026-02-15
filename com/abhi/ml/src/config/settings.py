"""
Configuration and Constants
Author: Abhinav
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
DATA_DIR = RESOURCES_DIR / "data"
MODELS_DIR = RESOURCES_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Data split configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.0  # For future use

# Model hyperparameters
class LogisticConfig:
    MAX_ITER = 10000
    SOLVER = 'lbfgs'
    RANDOM_STATE = RANDOM_STATE

class DecisionTreeConfig:
    CRITERION = 'gini'
    MAX_DEPTH = None
    RANDOM_STATE = RANDOM_STATE

class KNNConfig:
    N_NEIGHBORS = 5
    WEIGHTS = 'uniform'
    ALGORITHM = 'auto'

class NaiveBayesConfig:
    VAR_SMOOTHING = 1e-9

class RandomForestConfig:
    N_ESTIMATORS = 100
    CRITERION = 'gini'
    MAX_DEPTH = None
    RANDOM_STATE = RANDOM_STATE

class XGBoostConfig:
    N_ESTIMATORS = 100
    LEARNING_RATE = 0.1
    MAX_DEPTH = 3
    RANDOM_STATE = RANDOM_STATE

# File paths
TEST_DATA_FILE = DATA_DIR / "test_data.csv"
RESULTS_FILE = MODELS_DIR / "model_results.csv"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
DATASET_INFO_FILE = MODELS_DIR / "dataset_info.pkl"

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'