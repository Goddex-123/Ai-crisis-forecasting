"""
Risk Assessment Module
Calculates risk scores and generates alerts
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
from config.constants import AlertLevel
import yaml

class RiskScorer:
    """Calculates crisis risk scores"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize risk scorer"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['risk_assessment']
        self.weights = self.config['weights']
        self.thresholds = self.config['thresholds']
    
    def calculate_risk_score(self, probability: float, severity: float,
                            urgency: float, uncertainty: float) -> float:
        """
        Calculate overall risk score
        
        Args:
            probability: Crisis probability (0-1)
            severity: Estimated severity (0-100)
            urgency: Urgency based on time to crisis (0-100)
            uncertainty: Model uncertainty (0-100)
        
        Returns:
            Risk score (0-100)
        """
        score = (
            self.weights['probability'] * (probability * 100) +
            self.weights['severity'] * severity +
            self.weights['urgency'] * urgency +
            self.weights['uncertainty'] * uncertainty
        )
        
        return np.clip(score, 0, 100)
    
    def get_alert_level(self, risk_score: float) -> str:
        """
        Determine alert level from risk score
        
        Args:
            risk_score: Risk score (0-100)
        
        Returns:
            Alert level string
        """
        if risk_score >= self.thresholds['critical']:
            return AlertLevel.CRITICAL.value
        elif risk_score >= self.thresholds['high']:
            return AlertLevel.HIGH.value
        elif risk_score >= self.thresholds['medium']:
            return AlertLevel.MEDIUM.value
        else:
            return AlertLevel.LOW.value
    
    def score_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score crisis predictions
        
        Args:
            predictions_df: DataFrame with columns [date, region, probability, ...]
        
        Returns:
            DataFrame with added risk scores and alert levels
        """
        logger.info("Calculating risk scores...")
        
        df = predictions_df.copy()
        
        # Calculate severity (based on composite indicators if available)
        if 'composite_crisis_score' in df.columns:
            df['severity'] = df['composite_crisis_score']
        else:
            df['severity'] = 50.0  # Default
        
        # Calculate urgency (higher if prediction is soon)
        # Assuming predictions are sorted by date
        df['urgency'] = 70.0  # Default high urgency
        
        # Calculate uncertainty (1 - confidence)
        if 'confidence_lower' in df.columns and 'confidence_upper' in df.columns:
            df['uncertainty'] = (df['confidence_upper'] - df['confidence_lower']) / 2
        else:
            df['uncertainty'] = 20.0  # Default moderate uncertainty
        
        # Calculate risk scores
        df['risk_score'] = df.apply(
            lambda row: self.calculate_risk_score(
                row['probability'],
                row['severity'],
                row['urgency'],
                row['uncertainty']
            ),
            axis=1
        )
        
        # Determine alert levels
        df['alert_level'] = df['risk_score'].apply(self.get_alert_level)
        
        logger.info(f"✓ Calculated risk scores for {len(df)} predictions")
        
        return df


class AlertGenerator:
    """Generates crisis alerts"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize alert generator"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['risk_assessment']
        self.alert_cooldown_days = self.config.get('alert_cooldown_days', 7)
    
    def generate_alerts(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate alerts from risk scores
        
        Args:
            risk_df: DataFrame with risk scores and alert levels
        
        Returns:
            DataFrame of alerts
        """
        logger.info("Generating alerts...")
        
        # Filter for medium or higher alerts
        alerts = risk_df[
            risk_df['alert_level'].isin([
                AlertLevel.MEDIUM.value,
                AlertLevel.HIGH.value,
                AlertLevel.CRITICAL.value
            ])
        ].copy()
        
        # Create alert descriptions
        alerts['description'] = alerts.apply(self._create_alert_description, axis=1)
        alerts['is_active'] = 1
        alerts['alert_date'] = pd.to_datetime(alerts.get('date', datetime.now()))
        
        logger.info(f"✓ Generated {len(alerts)} active alerts")
        
        return alerts
    
    def _create_alert_description(self, row) -> str:
        """Create human-readable alert description"""
        crisis_type = row.get('crisis_type', 'composite')
        region = row.get('region', 'Unknown')
        risk_score = row.get('risk_score', 0)
        probability = row.get('probability', 0) * 100
        
        description = (
            f"{row['alert_level'].upper()} ALERT: {crisis_type} crisis risk in {region}. "
            f"Risk score: {risk_score:.1f}/100 (probability: {probability:.1f}%)"
        )
        
        return description
