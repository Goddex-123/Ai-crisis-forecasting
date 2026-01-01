"""
Global constants for the Crisis Forecasting System
"""
from enum import Enum

# Crisis Types
class CrisisType(Enum):
    PANDEMIC = "pandemic"
    FOOD_SHORTAGE = "food_shortage"
    CLIMATE_DISASTER = "climate_disaster"
    ECONOMIC_COLLAPSE = "economic_collapse"
    COMPOSITE = "composite"

# Alert Levels
class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Data Domains
class DataDomain(Enum):
    CLIMATE = "climate"
    HEALTH = "health"
    FOOD = "food"
    ECONOMIC = "economic"

# Regions
REGIONS = [
    "North America",
    "South America",
    "Europe",
    "Africa",
    "Asia",
    "Oceania",
    "Middle East"
]

# Countries
COUNTRIES = [
    "United States", "China", "India", "Brazil", "Russia",
    "Japan", "Germany", "United Kingdom", "France", "Italy",
    "Canada", "South Korea", "Spain", "Australia", "Mexico",
    "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", "Switzerland",
    # Add more as needed
]

# Time Constants
MONTHS_PER_YEAR = 12
DAYS_PER_MONTH = 30
FORECAST_HORIZON_MONTHS = 12

# Model Names
MODEL_ARIMA = "arima"
MODEL_PROPHET = "prophet"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_XGBOOST = "xgboost"
MODEL_LSTM = "lstm"
MODEL_ISOLATION_FOREST = "isolation_forest"
MODEL_ENSEMBLE = "ensemble"

ALL_MODELS = [
    MODEL_ARIMA,
    MODEL_PROPHET,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
    MODEL_LSTM,
    MODEL_ISOLATION_FOREST,
    MODEL_ENSEMBLE
]

# Feature Categories
CLIMATE_FEATURES = [
    'temperature_anomaly',
    'precipitation_anomaly',
    'extreme_weather_events',
    'sea_level_rise',
    'carbon_emissions'
]

HEALTH_FEATURES = [
    'disease_outbreaks',
    'hospital_capacity',
    'vaccination_rates',
    'mortality_rates',
    'pandemic_preparedness_index'
]

FOOD_FEATURES = [
    'crop_yields',
    'food_prices',
    'supply_chain_disruptions',
    'grain_reserves',
    'agricultural_production'
]

ECONOMIC_FEATURES = [
    'gdp_growth',
    'unemployment_rate',
    'inflation_rate',
    'trade_balance',
    'debt_to_gdp_ratio',
    'stock_market_volatility'
]

# Historical Crises for Validation
HISTORICAL_CRISES = {
    "COVID-19": {
        "type": CrisisType.PANDEMIC,
        "start_date": "2020-01",
        "end_date": "2022-12",
        "severity": 95,
        "regions_affected": ["Global"]
    },
    "2008 Financial Crisis": {
        "type": CrisisType.ECONOMIC_COLLAPSE,
        "start_date": "2008-09",
        "end_date": "2009-06",
        "severity": 90,
        "regions_affected": ["Global"]
    },
    "2011 East Africa Drought": {
        "type": CrisisType.FOOD_SHORTAGE,
        "start_date": "2011-07",
        "end_date": "2012-08",
        "severity": 85,
        "regions_affected": ["Africa"]
    },
    "2010 Pakistan Floods": {
        "type": CrisisType.CLIMATE_DISASTER,
        "start_date": "2010-07",
        "end_date": "2010-09",
        "severity": 80,
        "regions_affected": ["Asia"]
    },
    "2014-2016 Ebola Outbreak": {
        "type": CrisisType.PANDEMIC,
        "start_date": "2014-03",
        "end_date": "2016-01",
        "severity": 75,
        "regions_affected": ["Africa"]
    }
}

# Color Schemes for Visualization
RISK_COLORS = {
    AlertLevel.LOW: "#00FF00",
    AlertLevel.MEDIUM: "#FFFF00",
    AlertLevel.HIGH: "#FF8C00",
    AlertLevel.CRITICAL: "#FF0000"
}

DOMAIN_COLORS = {
    DataDomain.CLIMATE: "#1f77b4",
    DataDomain.HEALTH: "#ff7f0e",
    DataDomain.FOOD: "#2ca02c",
    DataDomain.ECONOMIC: "#d62728"
}

# Database Table Names
TABLE_CLIMATE_DATA = "climate_data"
TABLE_HEALTH_DATA = "health_data"
TABLE_FOOD_DATA = "food_data"
TABLE_ECONOMIC_DATA = "economic_data"
TABLE_PREDICTIONS = "predictions"
TABLE_ALERTS = "alerts"
TABLE_MODEL_PERFORMANCE = "model_performance"
