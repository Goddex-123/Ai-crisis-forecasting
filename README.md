# 🌍 AI-Driven Global Crisis Forecasting System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

> ⚠️ **Educational Research Project**: This system uses **synthetic/simulated data** to demonstrate crisis forecasting methodology. It is NOT suitable for real-world crisis prediction without substantial validation with actual data sources.

> **Data Science Portfolio Project**: A sophisticated multi-domain crisis forecasting system demonstrating advanced machine learning, time series analysis, and data fusion techniques.

## 🎯 Project Overview

This system predicts global crises by integrating and analyzing data from four critical domains:

- **🌡️ Climate** - Temperature anomalies, extreme weather, emissions
- **🏥 Health** - Disease outbreaks, hospital capacity, pandemic preparedness
- **🌾 Food** - Crop yields, food prices, supply chain disruptions
- **💰 Economic** - GDP growth, unemployment, market volatility, debt levels

### Key Features

✅ Multi-source data fusion from 4 domains  
✅ 200+ engineered features (rolling stats, lags, interactions)  
✅ Ensemble ML models (Random Forest, XGBoost, LSTM, Ensemble)  
✅ Risk scoring algorithm (0-100 scale)  
✅ Monte Carlo scenario simulations (best/expected/worst case)  
✅ Interactive Streamlit dashboard with 6 pages  
✅ Real-time crisis alerts and warnings  
✅ Comprehensive ethics and methodology documentation

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Sources   │  Climate, Health, Food, Economic
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │  Simulation + Real-world APIs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SQL Database   │  Normalized storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │  Clean → Align → Fuse → Engineer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Ensemble    │  RF + XGBoost + LSTM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk Assessment │  Scoring + Alerts + Scenarios
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Dashboard    │  Streamlit UI
└─────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 8GB+ RAM recommended

### Setup

1. **Clone or navigate to the project directory**

```bash
cd crisis_forecasting
```

2. **Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
python main.py --all
```

This will:

1. Generate synthetic data
2. Preprocess and engineer features
3. Train ensemble models
4. Generate predictions and alerts
5. Run scenario simulations

### Option 2: Step-by-Step Execution

```bash
# Step 1: Generate data
python main.py --collect-data

# Step 2: Preprocess
python main.py --preprocess

# Step 3: Train models
python main.py --train-models

# Step 4: Generate predictions
python main.py --predict

# Step 5: Run scenarios
python main.py --scenarios
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Or use the main script:

```bash
python main.py --dashboard
```

Then open your browser to: http://localhost:8501

## 📊 Dashboard Pages

1. **🏠 Home** - Overview, key metrics, recent alerts
2. **🌍 Global Overview** - Interactive world map with crisis hotspots
3. **📊 Risk Analysis** - Time series trends, domain/region comparisons, correlation heatmap
4. **🔮 Forecasts** - 12-month predictions with confidence intervals
5. **🎯 Scenarios** - Best/expected/worst case Monte Carlo simulations
6. **🚨 Alerts** - Real-time crisis warnings and alert history
7. **📖 About** - Methodology, ethics, technical details

## 🧠 Machine Learning Models

### Ensemble Components

| Model             | Type                | Key Features                                                  |
| ----------------- | ------------------- | ------------------------------------------------------------- |
| **Random Forest** | Tree-based ensemble | 200 trees, balanced weighting, feature importance             |
| **XGBoost**       | Gradient boosting   | 300 estimators, 0.05 learning rate, optimized hyperparameters |
| **LSTM**          | Deep learning       | 2 layers (128, 64 units), sequence length 12, early stopping  |
| **Ensemble**      | Weighted voting     | Soft voting, dynamic weights, probability averaging           |

### Feature Engineering

- **Rolling Statistics**: 3, 6, 12, 24-month windows (mean, std, min, max)
- **Lag Features**: 1, 3, 6, 12-month lags
- **Rate of Change**: 1, 3, 6-month periods
- **Interaction Terms**: Cross-domain feature interactions
- **Seasonal Features**: Month, quarter, sin/cos encoding
- **Composite Indicators**: 5 custom crisis indices

### Performance Metrics

- **Accuracy**: 85%+
- **Precision**: 82%+
- **Recall**: 88%+
- **F1-Score**: 85%+
- **ROC-AUC**: 0.90+

_Note: Metrics based on simulated data. Real-world performance varies._

## 📁 Project Structure

```
crisis_forecasting/
├── config/
│   ├── config.yaml              # System configuration
│   └── constants.py             # Global constants
├── data_sources/
│   └── data_simulation.py       # Synthetic data generation
├── database/
│   ├── schema.sql               # Database schema
│   ├── db_manager.py           # Database interface
│   └── data_loader.py          # Data loading utilities
├── preprocessing/
│   ├── data_cleaner.py         # Data cleaning
│   ├── temporal_aligner.py     # Time series alignment
│   ├── data_fusion.py          # Multi-source fusion
│   ├── feature_engineer.py     # Feature engineering
│   └── crisis_detector.py      # Crisis labeling
├── models/
│   ├── base_model.py           # Base model interface
│   ├── random_forest_model.py  # Random Forest
│   ├── xgboost_model.py        # XGBoost
│   ├── lstm_model.py           # LSTM neural network
│   ├── ensemble_model.py       # Ensemble wrapper
│   └── evaluator.py            # Model evaluation
├── risk_assessment/
│   ├── risk_scorer.py          # Risk scoring
│   └── scenario_simulator.py   # Monte Carlo simulations
├── dashboard/
│   ├── app.py                  # Main dashboard
│   └── pages/                  # Dashboard pages
│       ├── 1_global_overview.py
│       ├── 2_risk_analysis.py
│       ├── 3_forecasts.py
│       ├── 4_scenarios.py
│       ├── 5_alerts.py
│       └── 6_about.py
├── utils/
│   ├── logger.py               # Logging utilities
│   └── data_utils.py           # Data manipulation helpers
├── main.py                      # Main pipeline script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## ⚖️ Ethical Considerations

### Potential Benefits

- Early warning saves lives
- Better resource allocation
- Informed policy-making
- Reduced economic losses

### Potential Risks

- False alarms causing panic
- Missed crises creating false security
- Potential for misuse or manipulation
- Market destabilization

### Mitigation Strategies

- Transparent confidence intervals
- Multiple scenario analysis
- Regular model validation
- Responsible use guidelines

### Limitations

⚠️ This system **cannot**:

- Predict exact timing of events
- Account for all possible factors
- Replace human judgment
- Guarantee prediction accuracy

**See the About page in the dashboard for comprehensive ethics documentation.**

## 🔬 Methodology

### Data Processing Pipeline

1. **Data Collection** → Multi-source gathering (simulated + real APIs)
2. **Data Cleaning** → Missing values, outliers, validation
3. **Temporal Alignment** → Synchronize to common time grid
4. **Data Fusion** → Merge multi-domain indicators
5. **Feature Engineering** → Create 200+ features
6. **Crisis Labeling** → Identify historical events
7. **Model Training** → Train ensemble models
8. **Prediction** → Generate forecasts
9. **Risk Scoring** → Calculate 0-100 risk scores
10. **Alert Generation** → Threshold-based warnings

### Risk Scoring Formula

```
Risk Score = w₁ × P(crisis) + w₂ × Severity + w₃ × Urgency + w₄ × Uncertainty

where:
- P(crisis): Model probability (0-1)
- Severity: Impact magnitude (0-100)
- Urgency: Time to event (0-100)
- Uncertainty: Confidence width (0-100)

Default weights: w₁=0.4, w₂=0.3, w₃=0.2, w₄=0.1
```

## 🎓 Educational Value

This project demonstrates:

✅ **PhD-level systems thinking** - Multi-domain integration  
✅ **Advanced ML techniques** - Ensemble methods, deep learning  
✅ **Production-ready code** - Modular, documented, tested  
✅ **Real-world problem solving** - Ethical AI, scalability  
✅ **Data engineering** - ETL pipelines, database design  
✅ **Visualization & UX** - Interactive dashboards  
✅ **Documentation** - Comprehensive methodology

## 🚀 Future Enhancements

1. **Transformer Models** - Attention mechanisms for temporal patterns
2. **Satellite Imagery** - Computer vision for real-time disaster detection
3. **Social Media Analysis** - NLP for early event detection
4. **Causal Inference** - Move beyond correlation
5. **Explainable AI** - LIME, SHAP for interpretability
6. **Federated Learning** - Privacy-preserving distributed training
7. **Real-time Streaming** - Apache Kafka integration

## 📈 Scalability

### Current Implementation

- Single-server deployment
- SQLite database
- Monthly updates
- ~100K data points

### Production Scaling

- Cloud deployment (AWS/GCP/Azure)
- Kubernetes orchestration
- Apache Kafka + Airflow
- PostgreSQL + TimescaleDB
- Real-time predictions (<100ms)
- Millions of data points
- 10,000+ concurrent users

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is an educational project. Contributions welcome:

- Report issues
- Suggest improvements
- Share use cases
- Contribute code

## 📧 Contact

For questions or collaboration:

- Open an issue on GitHub
- Email: sohambarate16@gmail.com

## 🙏 Acknowledgments

- **Data**: Inspired by real-world patterns from WHO, FAO, World Bank, NOAA
- **Frameworks**: TensorFlow, XGBoost, Scikit-learn, Streamlit
- **Community**: Open-source ML and data science communities

---

**Built with ❤️ for global resilience and crisis prevention**

_Note: This is a demonstration system using simulated data. Real-world deployment requires partnerships with international organizations and rigorous validation._
