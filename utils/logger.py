"""
Logging utility for Crisis Forecasting System
"""
import logging
import sys
from pathlib import Path
import yaml

def setup_logger(name: str = "crisis_forecasting", config_path: str = "config/config.yaml") -> logging.Logger:
    """
    Set up logger with configuration
    
    Args:
        name: Logger name
        config_path: Path to config file
    
    Returns:
        Configured logger instance
    """
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        log_config = config.get('logging', {})
    except:
        log_config = {}
    
    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(log_config.get('level', 'INFO'))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = log_config.get('file', 'crisis_forecasting.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Default logger
logger = setup_logger()
