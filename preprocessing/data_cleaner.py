"""
Data Preprocessing Module
Cleans and validates multi-source crisis data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import handle_missing_values, detect_outliers, normalize_data
from utils.logger import logger

class DataCleaner:
    """Handles data cleaning and validation"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_dataset(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """
        Clean a dataset from a specific domain
        
        Args:
            df: Input dataframe
            domain: Data domain (climate, health, food, economic)
        
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Cleaning {domain} data ({len(df)} records)...")
        
        df_clean = df.copy()
        original_rows = len(df_clean)
        
        # 1. Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['date', 'region'], keep='last')
        duplicates_removed = original_rows - len(df_clean)
        
        # 2. Sort by date
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        # 3. Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        missing_before = df_clean[numeric_columns].isnull().sum().sum()
        
        df_clean = handle_missing_values(df_clean, strategy='interpolate')
        
        missing_after = df_clean[numeric_columns].isnull().sum().sum()
        
        # 4. Detect and cap outliers
        for col in numeric_columns:
            outlier_col = f'{col}_outlier'
            if outlier_col in df_clean.columns:
                continue
            
            df_with_outliers = detect_outliers(df_clean[[col]], [col], method='iqr', threshold=3)
            
            # Cap outliers at 99th percentile
            if df_with_outliers[outlier_col].sum() > 0:
                upper_bound = df_clean[col].quantile(0.99)
                lower_bound = df_clean[col].quantile(0.01)
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        # Store stats
        self.cleaning_stats[domain] = {
            'original_rows': original_rows,
            'duplicates_removed': duplicates_removed,
            'missing_values_filled': missing_before - missing_after,
            'final_rows': len(df_clean)
        }
        
        logger.info(f"✓ Cleaned {domain} data: {duplicates_removed} duplicates removed, "
                   f"{missing_before - missing_after} missing values filled")
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame, domain: str) -> bool:
        """
        Validate data quality
        
        Args:
            df: Dataframe to validate
            domain: Data domain
        
        Returns:
            True if valid, False otherwise
        """
        # Check for required columns
        required_cols = ['date', 'region']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in {domain} data")
            return False
        
        # Check for nulls in critical columns
        if df[required_cols].isnull().any().any():
            logger.error(f"Null values in required columns in {domain} data")
            return False
        
        # Check date range
        if df['date'].min() > pd.Timestamp('2000-01-01'):
            logger.warning(f"Data starts late: {df['date'].min()}")
        
        # Check for sufficient data
        if len(df) < 12:  # At least 1 year of monthly data
            logger.error(f"Insufficient data in {domain}: only {len(df)} records")
            return False
        
        logger.info(f"✓ {domain} data validation passed")
        return True
    
    def get_stats(self) -> Dict:
        """Get cleaning statistics"""
        return self.cleaning_stats
