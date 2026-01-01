"""
Forecasts Page
Time series predictions with confidence intervals
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.title("🔮 Crisis Forecasts")

st.markdown("""
12-month ahead predictions for crisis probabilities with confidence intervals.
""")

# Sidebar
st.sidebar.subheader("Forecast Settings")
selected_region = st.sidebar.selectbox(
    "Select Region",
    ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania', 'Middle East']
)

selected_crisis = st.sidebar.selectbox(
    "Crisis Type",
    ['Composite', 'Pandemic', 'Food Shortage', 'Climate Disaster', 'Economic Collapse']
)

forecast_months = st.sidebar.slider(
    "Forecast Horizon (months)",
    min_value=3,
    max_value=24,
    value=12
)

# Generate historical and forecast data
historical_dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='MS')
forecast_dates = pd.date_range(start='2025-02-01', periods=forecast_months, freq='MS')

# Historical data
np.random.seed(42)
base_prob = 0.3
historical_probs = []

for i in range(len(historical_dates)):
    prob = base_prob + 0.002 * i + np.random.normal(0, 0.05)
    prob = np.clip(prob, 0, 1)
    historical_probs.append(prob)

# Forecast data
forecast_probs = []
lower_bounds = []
upper_bounds = []

for i in range(forecast_months):
    prob = historical_probs[-1] + 0.01 * i + np.random.normal(0, 0.03)
    prob = np.clip(prob, 0, 1)
    
    uncertainty = 0.1 + 0.01 * i  # Increasing uncertainty
    lower = np.clip(prob - uncertainty, 0, 1)
    upper = np.clip(prob + uncertainty, 0, 1)
    
    forecast_probs.append(prob)
    lower_bounds.append(lower)
    upper_bounds.append(upper)

# Visualization
st.subheader(f"{selected_crisis} Crisis Forecast - {selected_region}")

fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=historical_dates,
    y=historical_probs,
    name='Historical',
    line=dict(color='blue', width=2),
    mode='lines'
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast_probs,
    name='Forecast',
    line=dict(color='red', width=2, dash='dash'),
    mode='lines'
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=list(forecast_dates) + list(forecast_dates[::-1]),
    y=upper_bounds + lower_bounds[::-1],
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% Confidence Interval',
    showlegend=True
))

# Risk threshold line
fig.add_hline(
    y=0.7,
    line_dash="dot",
    line_color="orange",
    annotation_text="High Risk Threshold",
    annotation_position="right"
)

fig.update_layout(
    title=f"{selected_crisis} Crisis Probability Over Time",
    xaxis_title="Date",
    yaxis_title="Crisis Probability",
    hovermode='x unified',
    height=500,
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig, use_container_width=True)

# Forecast statistics
st.subheader("Forecast Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Probability",
        f"{historical_probs[-1]:.1%}",
        ""
    )

with col2:
    st.metric(
        f"{forecast_months}-Month Forecast",
        f"{forecast_probs[-1]:.1%}",
        f"{(forecast_probs[-1] - historical_probs[-1]):.1%}"
    )

with col3:
    st.metric(
        "Peak Probability",
        f"{max(forecast_probs):.1%}",
        f"Month {forecast_probs.index(max(forecast_probs)) + 1}"
    )

with col4:
    high_risk_months = sum(1 for p in forecast_probs if p > 0.7)
    st.metric(
        "High Risk Months",
        high_risk_months,
        f"{(high_risk_months/forecast_months)*100:.0f}% of period"
    )

# Model breakdown
st.subheader("Model Predictions Breakdown")

models = ['Random Forest', 'XGBoost', 'LSTM', 'Ensemble']
model_probs = [np.random.uniform(0.3, 0.6) for _ in models]

model_df = pd.DataFrame({
    'Model': models,
    'Predicted Probability': model_probs,
    'Confidence': [np.random.uniform(0.7, 0.95) for _ in models]
})

col1, col2 = st.columns([2, 1])

with col1:
    import plotly.express as px
    fig_models = px.bar(
        model_df,
        x='Model',
        y='Predicted Probability',
        color='Confidence',
        color_continuous_scale='Blues',
        title="Individual Model Predictions"
    )
    fig_models.update_layout(height=400)
    st.plotly_chart(fig_models, use_container_width=True)

with col2:
    st.dataframe(
        model_df.style.format({
            'Predicted Probability': '{:.1%}',
            'Confidence': '{:.1%}'
        }),
        use_container_width=True
    )

# Forecast narrative
st.subheader("💡 Forecast Interpretation")

if forecast_probs[-1] > 0.7:
    st.error(f"""
    **HIGH RISK FORECAST**
    
    The model predicts a high probability ({forecast_probs[-1]:.1%}) of {selected_crisis.lower()} 
    crisis in {selected_region} within the next {forecast_months} months.
    
    **Recommended Actions**:
    - Activate early warning systems
    - Mobilize resources and preparedness measures
    - Increase monitoring frequency
    """)
elif forecast_probs[-1] > 0.5:
    st.warning(f"""
    **MODERATE RISK FORECAST**
    
    The model indicates moderate risk ({forecast_probs[-1]:.1%}) of crisis in the forecast period.
    Continue monitoring and maintain readiness protocols.
    """)
else:
    st.success(f"""
    **LOW RISK FORECAST**
    
    Current forecast indicates low risk ({forecast_probs[-1]:.1%}) of crisis.
    Maintain standard monitoring procedures.
    """)
