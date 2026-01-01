"""
Data Fusion Module
Combines multi-source data into unified crisis indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
from config.constants import CLIMATE_FEATURES, HEALTH_FEATURES, FOOD_FEATURES, ECONOMIC_FEATURES

class DataFusion:
    """Fuses multi-domain data"""
    
    def fuse_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Fuse datasets from multiple domains
        
        Args:
            datasets: Dictionary of aligned dataframes (domain -> df)
        
        Returns:
            Fused dataframe with all features
        """
        logger.info("Fusing multi-domain data...")
        
        # Start with first dataset
        base_domain = list(datasets.keys())[0]
        fused_df = datasets[base_domain][['date', 'region']].copy()
        
        # Add features from each domain
        for domain, df in datasets.items():
            logger.info(f"Merging {domain} features...")
            
            # Get feature columns for this domain
            feature_cols = [col for col in df.columns if col not in ['date', 'region', 'country']]
            
            # Prefix columns with domain name
            df_features = df[['date', 'region'] + feature_cols].copy()
            for col in feature_cols:
                df_features = df_features.rename(columns={col: f'{domain}_{col}'})
            
            # Merge
            fused_df = fused_df.merge(df_features, on=['date', 'region'], how='left')
        
        logger.info(f"✓ Fused data: {len(fused_df)} records with {len(fused_df.columns)} features")
        
        return fused_df
    
    def create_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite crisis indicators from multi-domain features
        
        Args:
            df: Fused dataframe
        
        Returns:
            Dataframe with added composite indicators
        """
        logger.info("Creating composite crisis indicators...")
        
        df_copy = df.copy()
        
        # Pandemic Risk Index
        # Combines health indicators with economic and climate factors
        health_cols = [col for col in df.columns if 'health_' in col and 'disease' in col or 'mortality' in col]
        if health_cols:
            df_copy['pandemic_risk_index'] = 0
            if 'health_disease_outbreaks' in df.columns:
                df_copy['pandemic_risk_index'] += df_copy['health_disease_outbreaks'].fillna(0) * 10
            if 'health_mortality_rates' in df.columns:
                df_copy['pandemic_risk_index'] += (df_copy['health_mortality_rates'].fillna(0) - 8) * 5
            if 'health_hospital_capacity' in df.columns:
                df_copy['pandemic_risk_index'] += (100 - df_copy['health_hospital_capacity'].fillna(100)) * 0.5
        
        # Food Crisis Index
        # Combines food supply, climate, and economic indicators
        df_copy['food_crisis_index'] = 0
        if 'food_crop_yields' in df.columns:
            df_copy['food_crisis_index'] += (100 - df_copy['food_crop_yields'].fillna(100)) * 0.5
        if 'food_food_prices' in df.columns:
            df_copy['food_crisis_index'] += (df_copy['food_food_prices'].fillna(100) - 100) * 0.3
        if 'food_grain_reserves' in df.columns:
            df_copy['food_crisis_index'] += (100 - df_copy['food_grain_reserves'].fillna(100)) * 0.4
        if 'climate_extreme_weather_events' in df.columns:
            df_copy['food_crisis_index'] += df_copy['climate_extreme_weather_events'].fillna(0) * 5
        
        # Climate Disaster Index
        # Combines extreme weather, temperature, and precipitation
        df_copy['climate_disaster_index'] = 0
        if 'climate_temperature_anomaly' in df.columns:
            df_copy['climate_disaster_index'] += np.abs(df_copy['climate_temperature_anomaly'].fillna(0)) * 10
        if 'climate_extreme_weather_events' in df.columns:
            df_copy['climate_disaster_index'] += df_copy['climate_extreme_weather_events'].fillna(0) * 8
        if 'climate_precipitation_anomaly' in df.columns:
            df_copy['climate_disaster_index'] += np.abs(df_copy['climate_precipitation_anomaly'].fillna(0)) * 0.5
        
        # Economic Crisis Index
        # Combines growth, unemployment, debt, and volatility
        df_copy['economic_crisis_index'] = 0
        if 'economic_gdp_growth' in df.columns:
            df_copy['economic_crisis_index'] += np.maximum(0, -df_copy['economic_gdp_growth'].fillna(2)) * 10
        if 'economic_unemployment_rate' in df.columns:
            df_copy['economic_crisis_index'] += (df_copy['economic_unemployment_rate'].fillna(5) - 5) * 3
        if 'economic_stock_market_volatility' in df.columns:
            df_copy['economic_crisis_index'] += (df_copy['economic_stock_market_volatility'].fillna(15) - 15) * 1.5
        if 'economic_debt_to_gdp_ratio' in df.columns:
            df_copy['economic_crisis_index'] += np.maximum(0, df_copy['economic_debt_to_gdp_ratio'].fillna(60) - 80) * 0.5
        
        # Composite Crisis Score (Overall)
        df_copy['composite_crisis_score'] = (
            df_copy.get('pandemic_risk_index', 0) * 0.25 +
            df_copy.get('food_crisis_index', 0) * 0.25 +
            df_copy.get('climate_disaster_index', 0) * 0.25 +
            df_copy.get('economic_crisis_index', 0) * 0.25
        )
        
        # Normalize to 0-100 scale
        for indicator in ['pandemic_risk_index', 'food_crisis_index', 'climate_disaster_index', 
                         'economic_crisis_index', 'composite_crisis_score']:
            if indicator in df_copy.columns:
                max_val = df_copy[indicator].max()
                if max_val > 0:
                    df_copy[indicator] = (df_copy[indicator] / max_val) * 100
        
        logger.info("✓ Created 5 composite crisis indicators")
        
        return df_copy
