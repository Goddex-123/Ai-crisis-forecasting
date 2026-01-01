"""
Model Evaluation Framework
Evaluates and compares crisis prediction models
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

class ModelEvaluator:
    """Evaluates model performance"""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
        """
        Evaluate model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            metrics['true_negatives'] = int(cm[0, 0])
            metrics['false_positives'] = int(cm[0, 1])
            metrics['false_negatives'] = int(cm[1, 0])
            metrics['true_positives'] = int(cm[1, 1])
        
        return metrics
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = self.evaluate(y_test.values, y_pred, y_proba)
        
        return metrics
    
    def compare_models(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Comparison dataframe
        """
        logger.info(f"Comparing {len(models)} models...")
        
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            try:
                metrics = self.evaluate_model(model, X_test, y_test)
                metrics['model'] = name
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        comparison_df = comparison_df[[col for col in cols if col in comparison_df.columns]]
        
        # Sort by F1 score
        if 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        logger.info("Model comparison complete")
        
        return comparison_df
    
    def get_metrics_summary(self, metrics: Dict) -> str:
        """Get formatted summary of metrics"""
        summary = f"""
Model Performance:
  Accuracy:  {metrics.get('accuracy', 0):.4f}
  Precision: {metrics.get('precision', 0):.4f}
  Recall:    {metrics.get('recall', 0):.4f}
  F1 Score:  {metrics.get('f1_score', 0):.4f}
  ROC AUC:   {metrics.get('roc_auc', 0):.4f}
"""
        
        if 'true_positives' in metrics:
            summary += f"""
Confusion Matrix:
  True Positives:  {metrics['true_positives']}
  True Negatives:  {metrics['true_negatives']}
  False Positives: {metrics['false_positives']}
  False Negatives: {metrics['false_negatives']}
"""
        
        return summary
