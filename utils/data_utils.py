"""
Data utilities for Crisis Forecasting System
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

def normalize_data(df: pd.DataFrame, columns: List[str], method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize specified columns
    
    Args:
        df: Input dataframe
        columns: Columns to normalize
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if method == 'zscore':
            df_copy[col] = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
        elif method == 'minmax':
            df_copy[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in dataframe
    
    Args:
        df: Input dataframe
        strategy: Strategy ('interpolate', 'forward_fill', 'mean', 'drop')
    
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == 'interpolate':
        df_copy = df_copy.interpolate(method='linear', limit_direction='both')
    elif strategy == 'forward_fill':
        df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'mean':
        df_copy = df_copy.fillna(df_copy.mean())
    elif strategy == 'drop':
        df_copy = df_copy.dropna()
    
    return df_copy

def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window features
    
    Args:
        df: Input dataframe
        columns: Columns to create rolling features for
        windows: Window sizes
    
    Returns:
        DataFrame with added rolling features
    """
    df_copy = df.copy()
    
    for col in columns:
        for window in windows:
            df_copy[f'{col}_rolling_mean_{window}'] = df_copy[col].rolling(window=window).mean()
            df_copy[f'{col}_rolling_std_{window}'] = df_copy[col].rolling(window=window).std()
            df_copy[f'{col}_rolling_min_{window}'] = df_copy[col].rolling(window=window).min()
            df_copy[f'{col}_rolling_max_{window}'] = df_copy[col].rolling(window=window).max()
    
    return df_copy

def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features
    
    Args:
        df: Input dataframe
        columns: Columns to create lag features for
        lags: Lag periods
    
    Returns:
        DataFrame with added lag features
    """
    df_copy = df.copy()
    
    for col in columns:
        for lag in lags:
            df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
    
    return df_copy

def create_rate_of_change_features(df: pd.DataFrame, columns: List[str], periods: List[int]) -> pd.DataFrame:
    """
    Create rate of change features
    
    Args:
        df: Input dataframe
        columns: Columns to calculate rate of change for
        periods: Periods for calculation
    
    Returns:
        DataFrame with added rate of change features
    """
    df_copy = df.copy()
    
    for col in columns:
        for period in periods:
            df_copy[f'{col}_roc_{period}'] = df_copy[col].pct_change(periods=period)
    
    return df_copy

def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in specified columns
    
    Args:
        df: Input dataframe
        columns: Columns to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outlier indicators
    """
    df_copy = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            df_copy[f'{col}_outlier'] = ((df_copy[col] < (Q1 - threshold * IQR)) | 
                                          (df_copy[col] > (Q3 + threshold * IQR))).astype(int)
        elif method == 'zscore':
            z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
            df_copy[f'{col}_outlier'] = (z_scores > threshold).astype(int)
    
    return df_copy

def generate_date_range(start_date: str, end_date: str, freq: str = 'MS') -> pd.DatetimeIndex:
    """
    Generate date range
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        freq: Frequency ('D', 'W', 'MS', 'M', 'Y')
    
    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def align_temporal_data(dfs: List[pd.DataFrame], date_col: str = 'date') -> List[pd.DataFrame]:
    """
    Align multiple dataframes to common date range
    
    Args:
        dfs: List of dataframes
        date_col: Name of date column
    
    Returns:
        List of aligned dataframes
    """
    # Find common date range
    min_date = max([df[date_col].min() for df in dfs])
    max_date = min([df[date_col].max() for df in dfs])
    
    # Filter all dataframes to common range
    aligned_dfs = []
    for df in dfs:
        aligned_df = df[(df[date_col] >= min_date) & (df[date_col] <= max_date)].copy()
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs

def calculate_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix
    
    Args:
        df: Input dataframe
        columns: Columns to include (None for all numeric)
    
    Returns:
        Correlation matrix
    """
    if columns:
        return df[columns].corr()
    else:
        return df.select_dtypes(include=[np.number]).corr()
