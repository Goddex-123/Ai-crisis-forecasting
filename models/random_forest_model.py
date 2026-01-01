"""
Random Forest Model for Crisis Classification
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from utils.logger import logger
import yaml

class RandomForestModel(BaseModel):
    """Random Forest classifier for crisis prediction"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Random Forest model"""
        super().__init__("random_forest")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['random_forest']
        
        if self.config.get('enabled', True):
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', 20),
                min_samples_split=self.config.get('min_samples_split', 5),
                class_weight=self.config.get('class_weight', 'balanced'),
                random_state=42,
                n_jobs=-1
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name} model...")
        
        # Remove non-numeric columns
        X_numeric = X_train.select_dtypes(include=[np.number])
        
        # Train
        self.model.fit(X_numeric, y_train)
        self.is_trained = True
        
        # Store feature importance
        self.feature_importance = dict(zip(
            X_numeric.columns,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"✓ {self.model_name} trained on {len(X_numeric.columns)} features")
        
        # Log top features
        top_features = list(self.feature_importance.items())[:5]
        logger.info(f"Top 5 features: {', '.join([f'{k}: {v:.4f}' for k, v in top_features])}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crisis class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_numeric = X.select_dtypes(include=[np.number])
        return self.model.predict(X_numeric)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crisis probability"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_numeric = X.select_dtypes(include=[np.number])
        return self.model.predict_proba(X_numeric)
