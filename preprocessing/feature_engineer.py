"""
Feature Engineering Module
Creates advanced features for crisis prediction
"""
import pandas as pd
import numpy as np
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import create_rolling_features, create_lag_features, create_rate_of_change_features
from utils.logger import logger
import yaml

class FeatureEngineer:
    """Engineers features from fused data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['feature_engineering']
        self.rolling_windows = self.config['rolling_windows']
        self.lag_periods = self.config['lag_periods']
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features for crisis prediction
        
        Args:
            df: Fused dataframe with composite indicators
        
        Returns:
            Dataframe with engineered features
        """
        logger.info("Engineering features...")
        
        df_engineered = df.copy()
        
        # Get numeric columns (excluding date)
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        
        # Focus on most important indicators
        key_indicators = [
            'pandemic_risk_index',
            'food_crisis_index',
            'climate_disaster_index',
            'economic_crisis_index',
            'composite_crisis_score'
        ]
        
        # Add domain-specific key features
        domain_features = []
        for col in numeric_cols:
            if any(keyword in col for keyword in ['temperature', 'disease', 'gdp', 'crop', 
                                                   'unemployment', 'food_prices', 'mortality']):
                domain_features.append(col)
        
        features_to_engineer = [col for col in key_indicators + domain_features if col in numeric_cols]
        
        logger.info(f"Engineering features for {len(features_to_engineer)} key indicators...")
        
        # Sort by date and region for proper time series operations
        df_engineered = df_engineered.sort_values(['region', 'date']).reset_index(drop=True)
        
        # Process each region separately
        regions = df_engineered['region'].unique()
        engineered_dfs = []
        
        for region in regions:
            region_df = df_engineered[df_engineered['region'] == region].copy()
            
            # 1. Rolling window features
            if self.config.get('rolling_windows'):
                region_df = create_rolling_features(region_df, features_to_engineer, self.rolling_windows)
            
            # 2. Lag features
            if self.config.get('lag_periods'):
                region_df = create_lag_features(region_df, features_to_engineer, self.lag_periods)
            
            # 3. Rate of change features
            region_df = create_rate_of_change_features(region_df, features_to_engineer, [1, 3, 6])
            
            # 4. Interaction terms
            if self.config.get('interaction_terms'):
                region_df = self._create_interaction_terms(region_df)
            
            # 5. Seasonal features
            if self.config.get('seasonal_decomposition'):
                region_df = self._create_seasonal_features(region_df)
            
            engineered_dfs.append(region_df)
        
        df_final = pd.concat(engineered_dfs, ignore_index=True)
        
        # Remove rows with too many NaN values (from lag/rolling operations)
        threshold = 0.3  # Remove rows with >30% NaN
        nan_ratio = df_final.isnull().sum(axis=1) / len(df_final.columns)
        df_final = df_final[nan_ratio < threshold]
        
        # Fill remaining NaN with 0 or forward fill
        df_final = df_final.fillna(method='ffill').fillna(0)
        
        logger.info(f"✓ Engineered {len(df_final.columns)} total features from {len(numeric_cols)} base features")
        logger.info(f"Final dataset: {len(df_final)} records")
        
        return df_final
    
    def _create_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between domains"""
        df_copy = df.copy()
        
        # Climate × Food (droughts affect crop yields)
        if 'climate_temperature_anomaly' in df.columns and 'food_crop_yields' in df.columns:
            df_copy['climate_food_interaction'] = (
                df_copy['climate_temperature_anomaly'] * (100 - df_copy['food_crop_yields'])
            )
        
        # Health × Economic (pandemics impact economy)
        if 'health_disease_outbreaks' in df.columns and 'economic_gdp_growth' in df.columns:
            df_copy['health_economic_interaction'] = (
                df_copy['health_disease_outbreaks'] * -df_copy['economic_gdp_growth']
            )
        
        # Economic × Food (economic crisis affects food prices)
        if 'economic_unemployment_rate' in df.columns and 'food_food_prices' in df.columns:
            df_copy['economic_food_interaction'] = (
                df_copy['economic_unemployment_rate'] * df_copy['food_food_prices']
            )
        
        return df_copy
    
    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal features from date"""
        df_copy = df.copy()
        
        if 'date' in df.columns:
            df_copy['month'] = pd.to_datetime(df_copy['date']).dt.month
            df_copy['quarter'] = pd.to_datetime(df_copy['date']).dt.quarter
            df_copy['year'] = pd.to_datetime(df_copy['date']).dt.year
            
            # Cyclical encoding for month
            df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
            df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        
        return df_copy
