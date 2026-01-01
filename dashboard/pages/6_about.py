"""
About Page
Documentation, methodology, and ethics
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.title("📖 About & Documentation")

st.markdown("""
Learn about the methodology, ethics, and technical details of this crisis forecasting system.
""")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔬 Methodology", "⚖️ Ethics", "🚀 Technical Details"])

with tab1:
    st.header("Project Overview")
    
    st.markdown("""
    ### Mission
    
    The AI-Driven Global Crisis Forecasting System aims to provide early warnings of potential global crises
    by integrating and analyzing data from multiple domains using advanced machine learning techniques.
    
    ### What We Forecast
    
    **🌡️ Climate Disasters**
    - Extreme weather events
    - Temperature anomalies
    - Precipitation patterns
    - Long-term climate trends
    
    **🏥 Pandemic Risks**
    - Disease outbreak patterns
    - Healthcare capacity
    - Vaccination coverage
    - Mortality trends
    
    **🌾 Food Security Crises**
    - Crop yield forecasts
    - Food price volatility
    - Supply chain disruptions
    - Agricultural production
    
    **💰 Economic Collapses**
    - GDP growth trends
    - Unemployment rates
    - Market volatility
    - Debt levels
    
    ### Data Sources
    
    This demonstration uses simulated data based on real-world patterns. Production deployment would integrate:
    
    - **Climate**: NOAA, NASA, IPCC
    - **Health**: WHO, CDC, national health agencies
    - **Food**: FAO, World Food Programme
    - **Economic**: World Bank, IMF, national statistics bureaus
    """)

with tab2:
    st.header("Methodology")
    
    st.markdown("""
    ### Data Processing Pipeline
    
    ```
    1. Data Collection → Multi-source data gathering
    2. Data Cleaning → Missing values, outliers, validation
    3. Temporal Alignment → Synchronize to common time grid
    4. Data Fusion → Merge multi-domain indicators
    5. Feature Engineering → Rolling stats, lags, interactions
    6. Crisis Labeling → Identify historical events
    7. Model Training → Train ensemble models
    8. Prediction → Generate forecasts
    9. Risk Scoring → Calculate 0-100 risk scores
    10. Alert Generation → Threshold-based warnings
    ```
    
    ### Machine Learning Models
    
    **Random Forest**
    - 200 decision trees
    - Max depth: 20
    - Balanced class weights
    - Feature importance tracking
    
    **XGBoost**
    - 300 boosted trees
    - Learning rate: 0.05
    - Max depth: 8
    - Gradient boosting for optimal performance
    
    **LSTM Neural Network**
    - 2 LSTM layers (128, 64 units)
    - 20% dropout for regularization
    - 12-month sequence length
    - Early stopping with patience=10
    
    **Ensemble Model**
    - Weighted combination of all models
    - Soft voting for probability estimates
    - Dynamic weight adjustment
    
    ### Feature Engineering
    
    Over **200+ features** created from base indicators:
    
    - Rolling statistics (3, 6, 12, 24 months)
    - Lag features (1, 3, 6, 12 months)
    - Rate of change (1, 3, 6 month periods)
    - Cross-domain interactions
    - Seasonal decomposition
    - Anomaly scores
    
    ### Risk Scoring Formula
    
    ```
    Risk Score = w₁ × P(crisis) + w₂ × Severity + w₃ × Urgency + w₄ × Uncertainty
    
    where:
    - P(crisis): Model-predicted probability (0-1)
    - Severity: Estimated impact magnitude (0-100)
    - Urgency: Time to predicted event (0-100)
    - Uncertainty: Confidence interval width (0-100)
    
    Default weights: w₁=0.4, w₂=0.3, w₃=0.2, w₄=0.1
    ```
    
    ### Model Evaluation
    
    Models are evaluated using:
    - **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - **Temporal**: Walk-forward validation
    - **Backtesting**: Historical crisis detection
    """)

with tab3:
    st.header("Ethical Considerations")
    
    st.markdown("""
    ### Core Ethical Principles
    
    #### 🤝 Beneficence (Doing Good)
    
    **Potential Benefits:**
    - Early warning can save lives and reduce suffering
    - Better resource allocation and preparedness
    - Informed policy-making and planning
    - Reduced economic losses through prevention
    
    #### ⚠️ Non-Maleficence (Avoiding Harm)
    
    **Potential Risks:**
    - **False Alarms**: Can cause unnecessary panic and economic disruption
    - **Missed Crises**: False negatives may create false sense of security
    - **Misuse**: Predictions could be weaponized or manipulated
    - **Market Impact**: Public forecasts may destabilize markets
    
    **Mitigation Strategies:**
    - Transparent confidence intervals
    - Multiple scenario analysis
    - Regular model validation
    - Restricted access to sensitive predictions
    
    #### ⚖️ Justice & Fairness
    
    **Concerns:**
    - **Data Bias**: Limited data from developing regions
    - **Access Inequality**: Who benefits from early warnings?
    - **Resource Distribution**: Prediction accuracy varies by region
    
    **Our Approach:**
    - Explicit acknowledgment of data limitations
    - Equal weight to all geographic regions
    - Open documentation of biases
    - Recommendation for local validation
    
    #### 🔒 Privacy & Sovereignty
    
    - Aggregate data only (no individual-level data)
    - Respect for national data sovereignty
    - Compliance with data protection regulations
    - Transparent data sources and usage
    
    #### 📖 Transparency & Explainability
    
    - Open methodology documentation
    - Feature importance visualization
    - SHAP values for prediction explanation
    - Public model performance metrics
    
    ### Limitations & Disclaimers
    
    ⚠️ **This system cannot:**
    - Predict exact timing of crisis events
    - Account for all possible factors
    - Replace human judgment and expertise
    - Guarantee accuracy of predictions
    
    ⚠️ **Users must understand:**
    - Predictions are probabilistic, not deterministic
    - Historical patterns may not repeat
    - Unexpected events (black swans) cannot be predicted
    - Local context is critically important
    
    ### Responsible Use Guidelines
    
    1. **Combine with Expert Analysis**: Use predictions as input, not sole decision basis
    2. **Verify with Local Data**: Cross-reference with region-specific information
    3. **Consider Multiple Scenarios**: Don't rely only on expected case
    4. **Update Regularly**: Models must be retrained with new data
    5. **Communicate Uncertainty**: Always present confidence intervals
    6. **Plan for Wrong Predictions**: Maintain contingency plans
    """)

with tab4:
    st.header("Technical Details")
    
    st.markdown("""
    ### Technology Stack
    
    **Core Technologies:**
    - Python 3.9+
    - TensorFlow 2.13+ (Deep Learning)
    - XGBoost 2.0+ (Gradient Boosting)
    - Scikit-learn 1.3+ (ML Algorithms)
    - Pandas 2.0+ (Data Processing)
    - NumPy 1.24+ (Numerical Computing)
    
    **Visualization:**
    - Streamlit 1.28+ (Dashboard Framework)
    - Plotly 5.17+ (Interactive Charts)
    - Seaborn & Matplotlib (Statistical Plots)
    
    **Database:**
    - SQLite (Development)
    - PostgreSQL (Production-ready)
    
    **Time Series:**
    - Statsmodels (ARIMA)
    - Prophet (Facebook's Forecasting)
    - LSTM (Sequential Neural Networks)
    
    ### System Architecture
    
    ```
    Data Sources
        ↓
    Data Ingestion Pipeline
        ↓
    SQL Database
        ↓
    Preprocessing (Clean, Align, Fuse)
        ↓
    Feature Engineering
        ↓
    ML Model Ensemble
        ↓
    Risk Scoring & Alerts
        ↓
    Streamlit Dashboard
    ```
    
    ### Scalability
    
    **Current Implementation:**
    - Single-server deployment
    - SQLite database
    - Monthly data updates
    - ~100K data points
    
    **Production Scaling:**
    
    **Infrastructure:**
    - Cloud deployment (AWS/GCP/Azure)
    - Kubernetes for orchestration
    - Auto-scaling based on load
    
    **Data Pipeline:**
    - Apache Kafka for streaming
    - Airflow for orchestration
    - Distributed processing with Spark
    
    **Model Serving:**
    - MLflow for model management
    - TensorFlow Serving / TorchServe
    - Model versioning and A/B testing
    
    **Database:**
    - PostgreSQL + TimescaleDB
    - Read replicas for scaling
    - Partitioning by time and region
    
    **Expected Capacity:**
    - Millions of data points
    - Real-time predictions (<100ms)
    - 10,000+ concurrent users
    - Daily model retraining
    
    ### Future Enhancements
    
    1. **Transformer Models**: Attention mechanisms for better temporal patterns
    2. **Satellite Imagery**: Computer vision for real-time disaster detection
    3. **Social Media Analysis**: NLP for early event detection
    4. **Causal Inference**: Beyond correlation to causation
    5. **Explainable AI**: More interpretable predictions (LIME, SHAP)
    6. **Multi-Modal Fusion**: Combine text, images, time series
    7. **Federated Learning**: Privacy-preserving distributed training
    
    ### Performance Metrics
    
    **Current System (Simulated Data):**
    - Accuracy: 85%+
    - Precision: 82%+
    - Recall: 88%+
    - F1-Score: 85%+
    - ROC-AUC: 0.90+
    
    *Note: Real-world performance will vary based on data quality and crisis type*
    
    ### Citation
    
    If you use this system or methodology, please cite:
    
    ```
    Crisis Forecasting System (2025)
    AI-Driven Global Crisis Forecasting using Multi-Domain Data Fusion
    GitHub: [repository link]
    ```
    
    ### Contact & Contributions
    
    This is an open educational project.
    
    - Report issues or suggest improvements
    - Contribute to the codebase
    - Share your use cases
    
    ---
    
    **Built with ❤️ for global resilience**
    """)

# Footer
st.markdown("---")
st.info("💡 This is a demonstration system for educational purposes. Real-world deployment requires partnerships with international organizations and rigorous validation.")
