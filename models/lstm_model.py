"""
LSTM Model for Crisis Time Series Prediction
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseModel
from utils.logger import logger
import yaml

class LSTMModel(BaseModel):
    """LSTM neural network for crisis time series prediction"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize LSTM model"""
        super().__init__("lstm")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['lstm']
        self.sequence_length = self.config.get('sequence_length', 12)
        self.scaler = StandardScaler()
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Create sequences for LSTM"""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        if y is not None:
            return np.array(X_seq), np.array(y_seq)
        return np.array(X_seq)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name} model...")
        
        # Remove non-numeric columns
        X_numeric = X_train.select_dtypes(include=[np.number])
        n_features = X_numeric.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_train.values)
        
        logger.info(f"Created {len(X_seq)} sequences of length {self.sequence_length}")
        
        # Build model
        units = self.config.get('units', [128, 64])
        dropout = self.config.get('dropout', 0.2)
        
        self.model = keras.Sequential([
            layers.LSTM(units[0], return_sequences=True, input_shape=(self.sequence_length, n_features)),
            layers.Dropout(dropout),
            layers.LSTM(units[1], return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        # Train
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        self.is_trained = True
        
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        
        logger.info(f"✓ {self.model_name} trained - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crisis class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crisis probability"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad beginning to match original length
        padded_predictions = np.pad(
            predictions.flatten(),
            (self.sequence_length, 0),
            mode='edge'
        )
        
        # Return as 2D array for compatibility
        return np.column_stack([1 - padded_predictions, padded_predictions])
