"""
Data Simulation Module for Crisis Forecasting System
Generates realistic synthetic data for demonstration purposes
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import REGIONS, COUNTRIES, CLIMATE_FEATURES, HEALTH_FEATURES, FOOD_FEATURES, ECONOMIC_FEATURES

class DataSimulator:
    """Generates synthetic crisis-related data"""
    
    def __init__(self, start_date: str = "2015-01-01", end_date: str = "2025-01-01", seed: int = 42):
        """
        Initialize data simulator
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            seed: Random seed for reproducibility
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.seed = seed
        np.random.seed(seed)
        
        # Generate date range (monthly data)
        self.dates = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        self.n_periods = len(self.dates)
    
    def _add_trend(self, base_value: float, trend_rate: float) -> np.ndarray:
        """Add linear trend to data"""
        trend = base_value + np.arange(self.n_periods) * trend_rate
        return trend
    
    def _add_seasonality(self, amplitude: float, period: int = 12) -> np.ndarray:
        """Add seasonal component"""
        t = np.arange(self.n_periods)
        seasonality = amplitude * np.sin(2 * np.pi * t / period)
        return seasonality
    
    def _add_noise(self, std: float) -> np.ndarray:
        """Add random noise"""
        return np.random.normal(0, std, self.n_periods)
    
    def _add_crisis_events(self, data: np.ndarray, crisis_dates: List[int], impact: float, duration: int = 6) -> np.ndarray:
        """Add crisis event impacts to data"""
        result = data.copy()
        for crisis_date in crisis_dates:
            # Check if crisis date is valid
            if crisis_date < 0 or crisis_date >= len(result):
                continue
            for i in range(duration):
                if crisis_date + i < len(result):
                    # Exponential decay of crisis impact
                    decay = np.exp(-i / duration)
                    result[crisis_date + i] += impact * decay
        return result
    
    def generate_climate_data(self, region: str, crisis_years: List[int] = [2020, 2010]) -> pd.DataFrame:
        """Generate climate data for a region"""
        data = {'date': self.dates, 'region': region}
        
        # Temperature anomaly (increasing trend + seasonality + crisis spikes)
        temp_base = np.random.uniform(-0.5, 0.5)
        temp_trend = self._add_trend(temp_base, 0.01)  # Warming trend
        temp_seasonal = self._add_seasonality(0.3)
        temp_noise = self._add_noise(0.1)
        crisis_dates = [int((year - self.start_date.year) * 12) for year in crisis_years]
        data['temperature_anomaly'] = self._add_crisis_events(
            temp_trend + temp_seasonal + temp_noise, crisis_dates, 1.5
        )
        
        # Precipitation anomaly
        precip_base = np.random.uniform(-10, 10)
        precip_trend = self._add_trend(precip_base, -0.05)
        precip_seasonal = self._add_seasonality(15)
        precip_noise = self._add_noise(5)
        data['precipitation_anomaly'] = precip_trend + precip_seasonal + precip_noise
        
        # Extreme weather events (increasing)
        extreme_base = np.random.poisson(2, self.n_periods)
        data['extreme_weather_events'] = np.maximum(0, extreme_base + np.arange(self.n_periods) // 20)
        
        # Sea level rise (steady increase)
        data['sea_level_rise'] = self._add_trend(0, 0.3) + self._add_noise(0.1)
        
        # Carbon emissions (increasing with some decrease after 2020)
        carbon_trend = self._add_trend(400, 0.5)
        post_2020_idx = int((2020 - self.start_date.year) * 12)
        carbon_trend[post_2020_idx:] += -0.1 * np.arange(len(carbon_trend[post_2020_idx:]))
        data['carbon_emissions'] = carbon_trend + self._add_noise(5)
        
        return pd.DataFrame(data)
    
    def generate_health_data(self, region: str, crisis_years: List[int] = [2020, 2014]) -> pd.DataFrame:
        """Generate health data for a region"""
        data = {'date': self.dates, 'region': region}
        
        # Disease outbreaks (spikes during crises)
        outbreak_base = np.random.poisson(1, self.n_periods)
        crisis_dates = [int((year - self.start_date.year) * 12) for year in crisis_years]
        data['disease_outbreaks'] = self._add_crisis_events(
            outbreak_base.astype(float), crisis_dates, 10, duration=12
        ).astype(int)
        
        # Hospital capacity (slight increase over time, decreases during crises)
        capacity_trend = self._add_trend(75, 0.05)
        capacity_noise = self._add_noise(2)
        data['hospital_capacity'] = np.clip(
            self._add_crisis_events(capacity_trend + capacity_noise, crisis_dates, -20),
            0, 100
        )
        
        # Vaccination rates (increasing trend)
        vacc_trend = self._add_trend(60, 0.1)
        vacc_noise = self._add_noise(3)
        data['vaccination_rates'] = np.clip(vacc_trend + vacc_noise, 0, 100)
        
        # Mortality rates (spikes during crises)
        mortality_base = np.random.uniform(7, 9, self.n_periods)
        data['mortality_rates'] = self._add_crisis_events(mortality_base, crisis_dates, 3)
        
        # Pandemic preparedness index
        prep_trend = self._add_trend(50, 0.08)
        prep_noise = self._add_noise(5)
        data['pandemic_preparedness_index'] = np.clip(prep_trend + prep_noise, 0, 100)
        
        return pd.DataFrame(data)
    
    def generate_food_data(self, region: str, crisis_years: List[int] = [2011, 2008]) -> pd.DataFrame:
        """Generate food supply data for a region"""
        data = {'date': self.dates, 'region': region}
        
        # Crop yields (decreases during crises)
        yield_trend = self._add_trend(100, 0.05)
        yield_seasonal = self._add_seasonality(5)
        yield_noise = self._add_noise(3)
        crisis_dates = [int((year - self.start_date.year) * 12) for year in crisis_years]
        data['crop_yields'] = np.clip(
            self._add_crisis_events(yield_trend + yield_seasonal + yield_noise, crisis_dates, -25),
            50, 150
        )
        
        # Food prices (increases during crises)
        price_trend = self._add_trend(100, 0.1)
        price_noise = self._add_noise(5)
        data['food_prices'] = self._add_crisis_events(price_trend + price_noise, crisis_dates, 30)
        
        # Supply chain disruptions
        disruption_base = np.random.poisson(1, self.n_periods)
        data['supply_chain_disruptions'] = self._add_crisis_events(
            disruption_base.astype(float), crisis_dates, 8
        ).astype(int)
        
        # Grain reserves (decreases during crises)
        reserves_trend = self._add_trend(80, -0.02)
        reserves_noise = self._add_noise(5)
        data['grain_reserves'] = np.clip(
            self._add_crisis_events(reserves_trend + reserves_noise, crisis_dates, -20),
            20, 100
        )
        
        # Agricultural production
        prod_trend = self._add_trend(100, 0.08)
        prod_noise = self._add_noise(4)
        data['agricultural_production'] = np.clip(
            self._add_crisis_events(prod_trend + prod_noise, crisis_dates, -15),
            60, 140
        )
        
        return pd.DataFrame(data)
    
    def generate_economic_data(self, region: str, crisis_years: List[int] = [2008, 2020]) -> pd.DataFrame:
        """Generate economic data for a region"""
        data = {'date': self.dates, 'region': region}
        
        # GDP growth (negative during crises)
        gdp_base = np.random.uniform(2, 4, self.n_periods)
        crisis_dates = [int((year - self.start_date.year) * 12) for year in crisis_years]
        data['gdp_growth'] = self._add_crisis_events(gdp_base, crisis_dates, -8)
        
        # Unemployment rate (spikes during crises)
        unemp_trend = self._add_trend(5, 0.01)
        unemp_noise = self._add_noise(0.5)
        data['unemployment_rate'] = np.clip(
            self._add_crisis_events(unemp_trend + unemp_noise, crisis_dates, 6),
            1, 25
        )
        
        # Inflation rate
        inflation_trend = self._add_trend(2, 0.02)
        inflation_noise = self._add_noise(0.5)
        data['inflation_rate'] = self._add_crisis_events(
            inflation_trend + inflation_noise, crisis_dates, 3
        )
        
        # Trade balance
        trade_trend = self._add_trend(0, -0.01)
        trade_noise = self._add_noise(2)
        data['trade_balance'] = self._add_crisis_events(trade_trend + trade_noise, crisis_dates, -10)
        
        # Debt to GDP ratio
        debt_trend = self._add_trend(60, 0.2)
        debt_noise = self._add_noise(3)
        data['debt_to_gdp_ratio'] = self._add_crisis_events(debt_trend + debt_noise, crisis_dates, 15)
        
        # Stock market volatility (spikes during crises)
        volatility_base = np.random.uniform(10, 20, self.n_periods)
        data['stock_market_volatility'] = self._add_crisis_events(
            volatility_base, crisis_dates, 40, duration=9
        )
        
        return pd.DataFrame(data)
    
    def generate_all_data(self, regions: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate all data for all regions
        
        Args:
            regions: List of regions (defaults to all regions)
        
        Returns:
            Dictionary with dataframes for each domain
        """
        if regions is None:
            regions = REGIONS
        
        climate_data = []
        health_data = []
        food_data = []
        economic_data = []
        
        for region in regions:
            # Vary crisis years slightly by region
            climate_crisis = [2020, 2010] if region != "Africa" else [2011, 2010]
            health_crisis = [2020, 2014] if region != "Asia" else [2020, 2003]
            food_crisis = [2011, 2008] if region != "Asia" else [2008, 2021]
            economic_crisis = [2008, 2020]
            
            climate_data.append(self.generate_climate_data(region, climate_crisis))
            health_data.append(self.generate_health_data(region, health_crisis))
            food_data.append(self.generate_food_data(region, food_crisis))
            economic_data.append(self.generate_economic_data(region, economic_crisis))
        
        return {
            'climate': pd.concat(climate_data, ignore_index=True),
            'health': pd.concat(health_data, ignore_index=True),
            'food': pd.concat(food_data, ignore_index=True),
            'economic': pd.concat(economic_data, ignore_index=True)
        }

if __name__ == "__main__":
    # Test data generation
    simulator = DataSimulator()
    data = simulator.generate_all_data()
    
    print("Generated data shapes:")
    for domain, df in data.items():
        print(f"{domain}: {df.shape}")
        print(df.head())
        print("\n")
