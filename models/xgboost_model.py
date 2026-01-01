"""
XGBoost Model for Crisis Classification
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from utils.logger import logger
import yaml

class XGBoostModel(BaseModel):
    """XGBoost classifier for crisis prediction"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize XGBoost model"""
        super().__init__("xgboost")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['xgboost']
        
        if self.config.get('enabled', True):
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 300),
                max_depth=self.config.get('max_depth', 8),
                learning_rate=self.config.get('learning_rate', 0.05),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name} model...")
        
        # Remove non-numeric columns
        X_numeric = X_train.select_dtypes(include=[np.number])
        
        # Train
        self.model.fit(
            X_numeric, 
            y_train,
            verbose=False
        )
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
