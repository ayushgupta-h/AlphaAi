"""
Unit tests for logging configuration module.
"""

import logging
import tempfile
import os
from bitcoin_forecasting.utils.logging_config import setup_logging, get_logger


def test_setup_logging_default():
    """Test that setup_logging creates a logger with default settings."""
    logger = setup_logging()
    
    assert logger is not None
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_setup_logging_custom_level():
    """Test that setup_logging respects custom log level."""
    logger = setup_logging(level=logging.DEBUG)
    
    assert logger.level == logging.DEBUG


def test_setup_logging_with_file():
    """Test that setup_logging creates file handler when log_file is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")
        logger = setup_logging(log_file=log_file)
        
        # Log a test message
        logger.info("Test message")
        
        # Close all handlers to release file locks (Windows requirement)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Verify file was created and contains the message
        assert os.path.exists(log_file)
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content


def test_get_logger():
    """Test that get_logger returns a logger instance."""
    logger = get_logger("test_module")
    
    assert logger is not None
    assert logger.name == "test_module"


def test_logging_format_includes_timestamp():
    """Test that log messages include timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")
        logger = setup_logging(log_file=log_file)
        
        logger.info("Test timestamp")
        
        # Close all handlers to release file locks (Windows requirement)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        with open(log_file, 'r') as f:
            content = f.read()
            # Check for timestamp format (YYYY-MM-DD HH:MM:SS)
            assert "-" in content  # Date separators
            assert ":" in content  # Time separators
            assert "INFO" in content
            assert "Test timestamp" in content
