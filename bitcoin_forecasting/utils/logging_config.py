"""
Logging configuration module for the Bitcoin Probabilistic Forecasting System.

This module provides centralized logging configuration with timestamps, log levels,
and formatted output for all system components.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return the root logger for the forecasting system.

    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO)
        Options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    log_file : str, optional
        Path to log file. If None, logs only to console (default: None)
    format_string : str, optional
        Custom format string for log messages. If None, uses default format (default: None)

    Returns
    -------
    logging.Logger
        Configured root logger instance

    Examples
    --------
    >>> logger = setup_logging(level=logging.DEBUG)
    >>> logger.info("System initialized")
    2024-01-15 10:30:45,123 - INFO - System initialized

    >>> logger = setup_logging(level=logging.INFO, log_file="forecast.log")
    >>> logger.warning("Using fallback volatility estimate")
    """
    # Default format includes timestamp, level, and message
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Parameters
    ----------
    name : str
        Name of the module (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance for the specified module

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)
