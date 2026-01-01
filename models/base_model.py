"""
Base Model Interface
Defines common interface for all crisis prediction models
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BaseModel(ABC):
    """Abstract base class for all crisis prediction models"""
    
    def __init__(self, model_name: str):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_importance = None
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def save_model(self, path: str):
        """Save model to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, path: str):
        """Load model from disk"""
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance
