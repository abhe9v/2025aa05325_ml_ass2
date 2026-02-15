"""
Logging utility
Author: Abhinav
"""

import logging
from com.abhi.ml.src.config.settings import LOG_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger