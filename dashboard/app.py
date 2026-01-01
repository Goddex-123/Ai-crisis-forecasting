"""
AI-Driven Global Crisis Forecasting System
Main Streamlit Dashboard Application
"""
import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

# Page configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dashboard_config = config['dashboard']

st.set_page_config(
    page_title=dashboard_config['title'],
    page_icon=dashboard_config['page_icon'],
    layout=dashboard_config['layout'],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem;
        font-weight: 700;
    }
    h2 {
        color: #FF6B6B;
        font-size: 1.8rem;
    }
    h3 {
        color: #888;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
    .alert-critical {
        background-color: #8B0000;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
    }
    .alert-high {
        background-color: #FF8C00;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .alert-medium {
        background-color: #FFFF00;
        padding: 1rem;
        border-radius: 0.5rem;
        color: black;
    }
    .alert-low {
        background-color: #00FF00;
        padding: 1rem;
        border-radius: 0.5rem;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/FF4B4B/FFFFFF?text=Crisis+Forecasting", use_container_width=True)
    
    st.title("Navigation")
    st.markdown("---")
    
    st.markdown("""
    ### About This System
    
    This AI-driven system forecasts global crises by analyzing:
    
    🌡️ **Climate** - Temperature, extreme weather, emissions
    
    🏥 **Health** - Disease outbreaks, pandemic preparedness
    
    🌾 **Food** - Crop yields, prices, supply chains
    
    💰 **Economic** - GDP, unemployment, market volatility
    
    Using advanced ML models including:
    - Random Forest
    - XGBoost
    - LSTM Neural Networks
    - Ensemble Methods
    
    ---
    
    ### Quick Stats
    """)

# Main content
st.title("🌍 AI-Driven Global Crisis Forecasting System")

st.markdown("""
### Welcome to the Crisis Forecasting Dashboard

This system uses artificial intelligence to predict and assess global crisis risks across multiple domains.
Navigate through the pages using the sidebar to explore:

- **Global Overview** - Interactive world map with crisis hotspots
- **Risk Analysis** - Detailed risk breakdowns by domain and region
- **Forecasts** - Time series predictions with confidence intervals
- **Scenarios** - Best/worst case scenario simulations
- **Alerts** - Real-time crisis alerts and warnings
- **Model Insights** - Model performance and feature importance
- **About** - Documentation, methodology, and ethics

---
""")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🌡️ Climate Risk",
        value="MEDIUM",
        delta="+5% from last month"
    )

with col2:
    st.metric(
        label="🏥 Pandemic Risk",
        value="LOW",
        delta="-12% from last month"
    )

with col3:
    st.metric(
        label="🌾 Food Security Risk",
        value="HIGH",
        delta="+8% from last month"
    )

with col4:
    st.metric(
        label="💰 Economic Risk",
        value="MEDIUM",
        delta="+2% from last month"
    )

st.markdown("---")

# Recent alerts
st.subheader("🚨 Recent Critical Alerts")

alert_col1, alert_col2 = st.columns(2)

with alert_col1:
    st.markdown("""
    <div class="alert-high">
        <strong>HIGH ALERT</strong> - Food Crisis Risk in Africa<br>
        Risk Score: 78/100 | Probability: 65%
    </div>
    """, unsafe_allow_html=True)

with alert_col2:
    st.markdown("""
    <div class="alert-medium">
        <strong>MEDIUM ALERT</strong> - Economic Instability in South America<br>
        Risk Score: 58/100 | Probability: 45%
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting started
st.subheader("🚀 Getting Started")

st.markdown("""
1. **Explore the Global Overview** to see current crisis hotspots on the world map
2. **Review Risk Analysis** for detailed breakdowns by region and crisis type
3. **Check Forecasts** to see predicted trends for the next 12 months
4. **Run Scenario Simulations** to explore best/worst case outcomes
5. **Monitor Alerts** for real-time warnings and updates

**Note**: This is a demonstration system using simulated data for educational purposes.
Real-world deployment would require partnerships with international organizations.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>AI-Driven Global Crisis Forecasting System | PhD-Level Data Science Project</p>
    <p>Built with: Python • TensorFlow • XGBoost • Streamlit • Plotly</p>
</div>
""", unsafe_allow_html=True)
