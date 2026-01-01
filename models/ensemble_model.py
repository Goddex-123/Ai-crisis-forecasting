"""
Ensemble Model
Combines multiple models for better predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from utils.logger import logger
import yaml

class EnsembleModel(BaseModel):
    """Ensemble of multiple crisis prediction models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ensemble model"""
        super().__init__("ensemble")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['ensemble']
        self.weights = self.config.get('weights', {})
        
        # Initialize individual models
        self.models = {
            'random_forest': RandomForestModel(config_path),
            'xgboost': XGBoostModel(config_path),
            'lstm': LSTMModel(config_path)
        }
        
        # Default equal weights if not specified
        if not self.weights:
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models.keys()}
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Train all models in the ensemble
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name} with {len(self.models)} models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.train(X_train, y_train, **kwargs)
        
        self.is_trained = True
        logger.info(f"✓ Ensemble model training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crisis class using weighted voting"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Get probabilities from all models
        proba = self.predict_proba(X)
        
        # Convert to binary predictions
        return (proba[:, 1] >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crisis probability using weighted average"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Collect predictions from all models
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    proba = model.predict_proba(X)
                    predictions.append(proba)
                    weights.append(self.weights.get(name, 1.0 / len(self.models)))
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No models successfully made predictions!")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble_proba = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_proba += pred * weight
        
        return ensemble_proba
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model"""
        predictions = {}
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    predictions[name] = model.predict_proba(X)[:, 1]
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
        
        return predictions
