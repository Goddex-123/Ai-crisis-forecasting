import pytest
import sys
import os

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        import main
        import app
        import config
        import data_sources
        import models
        import preprocessing
        import risk_assessment
        import utils
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")

def test_directory_structure():
    """Test that critical directories exist."""
    required_dirs = [
        'config',
        'dashboard',
        'data_sources',
        'database',
        'models',
        'preprocessing',
        'risk_assessment',
        'utils'
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Directory {d} is missing"
