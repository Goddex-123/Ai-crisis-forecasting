"""
Temporal Alignment Module
Aligns data from different sources to common time grid
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

class TemporalAligner:
    """Aligns multi-source temporal data"""
    
    def align_datasets(self, datasets: Dict[str, pd.DataFrame], 
                       date_col: str = 'date',
                       freq: str = 'MS') -> Dict[str, pd.DataFrame]:
        """
        Align multiple datasets to common date range and frequency
        
        Args:
            datasets: Dictionary of dataframes (domain -> df)
            date_col: Name of date column
            freq: Frequency for alignment ('D', 'W', 'MS', 'M')
        
        Returns:
            Dictionary of aligned dataframes
        """
        logger.info(f"Aligning {len(datasets)} datasets to common temporal grid...")
        
        # Find global date range
        all_dates = []
        for domain, df in datasets.items():
            if date_col in df.columns:
                all_dates.extend(df[date_col].tolist())
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        logger.info(f"Common date range: {min_date} to {max_date}")
        
        # Create common date grid
        common_dates = pd.date_range(start=min_date, end=max_date, freq=freq)
        
        # Align each dataset
        aligned_datasets = {}
        for domain, df in datasets.items():
            logger.info(f"Aligning {domain} data...")
            
            # Group by region
            regions = df['region'].unique()
            aligned_dfs = []
            
            for region in regions:
                region_df = df[df['region'] == region].copy()
                region_df = region_df.set_index(date_col)
                
                # Reindex to common dates
                region_df = region_df.reindex(common_dates)
                region_df['region'] = region
                region_df[date_col] = region_df.index
                
                # Fill missing values with interpolation
                numeric_cols = region_df.select_dtypes(include=[np.number]).columns
                region_df[numeric_cols] = region_df[numeric_cols].interpolate(method='linear', limit_direction='both')
                
                aligned_dfs.append(region_df)
            
            aligned_datasets[domain] = pd.concat(aligned_dfs, ignore_index=True)
            logger.info(f"✓ Aligned {domain}: {len(aligned_datasets[domain])} records")
        
        logger.info("Temporal alignment complete")
        return aligned_datasets
