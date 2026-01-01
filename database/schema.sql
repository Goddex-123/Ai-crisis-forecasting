-- Crisis Forecasting System Database Schema

-- Climate Data Table
CREATE TABLE IF NOT EXISTS climate_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    temperature_anomaly REAL,
    precipitation_anomaly REAL,
    extreme_weather_events INTEGER,
    sea_level_rise REAL,
    carbon_emissions REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, region, country)
);

-- Health Data Table
CREATE TABLE IF NOT EXISTS health_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    disease_outbreaks INTEGER,
    hospital_capacity REAL,
    vaccination_rates REAL,
    mortality_rates REAL,
    pandemic_preparedness_index REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, region, country)
);

-- Food Data Table
CREATE TABLE IF NOT EXISTS food_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    crop_yields REAL,
    food_prices REAL,
    supply_chain_disruptions INTEGER,
    grain_reserves REAL,
    agricultural_production REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, region, country)
);

-- Economic Data Table
CREATE TABLE IF NOT EXISTS economic_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    gdp_growth REAL,
    unemployment_rate REAL,
    inflation_rate REAL,
    trade_balance REAL,
    debt_to_gdp_ratio REAL,
    stock_market_volatility REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, region, country)
);

-- Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    crisis_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    probability REAL NOT NULL,
    risk_score REAL NOT NULL,
    confidence_lower REAL,
    confidence_upper REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_date DATE NOT NULL,
    region VARCHAR(100) NOT NULL,
    country VARCHAR(100),
    crisis_type VARCHAR(50) NOT NULL,
    alert_level VARCHAR(20) NOT NULL,
    risk_score REAL NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Performance Table
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value REAL NOT NULL,
    dataset VARCHAR(20) NOT NULL,  -- train, validation, test
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_climate_date_region ON climate_data(date, region);
CREATE INDEX IF NOT EXISTS idx_health_date_region ON health_data(date, region);
CREATE INDEX IF NOT EXISTS idx_food_date_region ON food_data(date, region);
CREATE INDEX IF NOT EXISTS idx_economic_date_region ON economic_data(date, region);
CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON predictions(target_date, region);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(is_active, alert_date);
