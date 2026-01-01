"""
Crisis Event Labeler
Labels historical crisis events for supervised learning
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
from config.constants import CrisisType

class CrisisLabeler:
    """Labels crisis events in historical data"""
    
    def __init__(self, threshold_multiplier: float = 2.0):
        """
        Initialize crisis labeler
        
        Args:
            threshold_multiplier: Multiplier for standard deviation threshold
        """
        self.threshold_multiplier = threshold_multiplier
    
    def label_crises(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label crisis events based on composite indicators
        
        Args:
            df: Dataframe with composite crisis indicators
        
        Returns:
            Dataframe with crisis labels
        """
        logger.info("Labeling crisis events...")
        
        df_labeled = df.copy()
        
        # Define crisis indicators
        crisis_indicators = {
            'pandemic': 'pandemic_risk_index',
            'food_shortage': 'food_crisis_index',
            'climate_disaster': 'climate_disaster_index',
            'economic_collapse': 'economic_crisis_index',
            'composite': 'composite_crisis_score'
        }
        
        # Label each crisis type
        for crisis_type, indicator in crisis_indicators.items():
            if indicator in df.columns:
                # Calculate threshold (mean + N * std)
                mean_val = df[indicator].mean()
                std_val = df[indicator].std()
                threshold = mean_val + self.threshold_multiplier * std_val
                
                # Label crisis events
                df_labeled[f'crisis_{crisis_type}'] = (df[indicator] > threshold).astype(int)
                
                n_crises = df_labeled[f'crisis_{crisis_type}'].sum()
                logger.info(f"Labeled {n_crises} {crisis_type} events (threshold: {threshold:.2f})")
        
        # Overall crisis label (any crisis type)
        crisis_cols = [col for col in df_labeled.columns if col.startswith('crisis_') and col != 'crisis_composite']
        if crisis_cols:
            df_labeled['crisis_any'] = df_labeled[crisis_cols].max(axis=1)
        
        return df_labeled
    
    def get_crisis_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of crisis events
        
        Args:
            df: Labeled dataframe
        
        Returns:
            Summary dataframe
        """
        crisis_cols = [col for col in df.columns if col.startswith('crisis_')]
        
        summary = pd.DataFrame({
            'crisis_type': crisis_cols,
            'n_events': [df[col].sum() for col in crisis_cols],
            'pct_time': [(df[col].sum() / len(df)) * 100 for col in crisis_cols]
        })
        
        return summary
