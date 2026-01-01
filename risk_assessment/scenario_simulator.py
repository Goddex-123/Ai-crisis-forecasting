"""
Scenario Simulation Module
Simulates best/worst case scenarios using Monte Carlo
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
import yaml

class ScenarioSimulator:
    """Simulates crisis scenarios"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize scenario simulator"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['scenario_simulation']
        self.n_simulations = self.config.get('n_simulations', 1000)
        self.confidence_intervals = self.config.get('confidence_intervals', [0.05, 0.50, 0.95])
    
    def simulate_scenarios(self, model, X_future: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Simulate multiple scenarios
        
        Args:
            model: Trained model
            X_future: Future features for prediction
        
        Returns:
            Dictionary of scenario dataframes
        """
        logger.info(f"Running {self.n_simulations} scenario simulations...")
        
        # Base prediction
        base_proba = model.predict_proba(X_future)[:, 1]
        
        # Monte Carlo simulations
        simulations = []
        
        for i in range(self.n_simulations):
            # Add random noise to predictions
            noise = np.random.normal(0, 0.1, len(base_proba))
            sim_proba = np.clip(base_proba + noise, 0, 1)
            simulations.append(sim_proba)
        
        simulations = np.array(simulations)
        
        # Calculate percentiles
        scenarios = {}
        
        # Best case (5th percentile - lower crisis probability)
        scenarios['best_case'] = X_future.copy()
        scenarios['best_case']['probability'] = np.percentile(simulations, 5, axis=0)
        scenarios['best_case']['scenario'] = 'best_case'
        
        # Expected case (50th percentile - median)
        scenarios['expected'] = X_future.copy()
        scenarios['expected']['probability'] = np.percentile(simulations, 50, axis=0)
        scenarios['expected']['scenario'] = 'expected'
        
        # Worst case (95th percentile - higher crisis probability)
        scenarios['worst_case'] = X_future.copy()
        scenarios['worst_case']['probability'] = np.percentile(simulations, 95, axis=0)
        scenarios['worst_case']['scenario'] = 'worst_case'
        
        # Add confidence intervals
        for scenario_name, scenario_df in scenarios.items():
            scenario_df['confidence_lower'] = np.percentile(simulations, 5, axis=0)
            scenario_df['confidence_upper'] = np.percentile(simulations, 95, axis=0)
            scenario_df['uncertainty'] = scenario_df['confidence_upper'] - scenario_df['confidence_lower']
        
        logger.info("✓ Scenario simulation complete")
        
        return scenarios
    
    def compare_scenarios(self, scenarios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare scenarios
        
        Args:
            scenarios: Dictionary of scenario dataframes
        
        Returns:
            Comparison dataframe
        """
        comparison = []
        
        for name, df in scenarios.items():
            comparison.append({
                'scenario': name,
                'mean_probability': df['probability'].mean(),
                'max_probability': df['probability'].max(),
                'mean_uncertainty': df['uncertainty'].mean(),
                'high_risk_periods': (df['probability'] > 0.7).sum()
            })
        
        return pd.DataFrame(comparison)
