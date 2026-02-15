"""
File I/O operations
Author: Abhinav
"""

import pickle
from pathlib import Path
from typing import Any
import pandas as pd
from com.abhi.ml.src.utils.logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Handles file operations for models and data"""

    @staticmethod
    def save_pickle(obj: Any, filepath: Path) -> None:
        """Save object as pickle file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            logger.info(f"Saved pickle: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            raise

    @staticmethod
    def load_pickle(filepath: Path) -> Any:
        """Load pickle file"""
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            logger.info(f"Loaded pickle: {filepath}")
            return obj
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise

    @staticmethod
    def save_csv(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
        """Save DataFrame to CSV"""
        try:
            df.to_csv(filepath, index=index)
            logger.info(f"Saved CSV: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save CSV {filepath}: {e}")
            raise

    @staticmethod
    def load_csv(filepath: Path) -> pd.DataFrame:
        """Load CSV file"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV {filepath}: {e}")
            raise