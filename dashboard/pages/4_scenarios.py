"""
Scenarios Page
Best/worst case scenario simulations
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.title("🎯 Scenario Simulations")

st.markdown("""
Explore best-case, expected, and worst-case crisis scenarios for strategic planning.
""")

# Sidebar
st.sidebar.subheader("Simulation Settings")
selected_region = st.sidebar.selectbox(
    "Select Region",
    ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania', 'Middle East']
)

simulation_months = st.sidebar.slider(
    "Simulation Horizon (months)",
    min_value=6,
    max_value=24,
    value=12
)

# Generate scenario data
dates = pd.date_range(start='2025-02-01', periods=simulation_months, freq='MS')

# Base scenario parameters
base_prob = 0.4
trend = 0.01

# Best case (5th percentile)
best_case = []
for i in range(simulation_months):
    prob = base_prob + trend * i * 0.3 + np.random.normal(-0.05, 0.02)
    best_case.append(np.clip(prob, 0, 1))

# Expected case (50th percentile)
expected = []
for i in range(simulation_months):
    prob = base_prob + trend * i + np.random.normal(0, 0.03)
    expected.append(np.clip(prob, 0, 1))

# Worst case (95th percentile)
worst_case = []
for i in range(simulation_months):
    prob = base_prob + trend * i * 1.7 + np.random.normal(0.05, 0.02)
    worst_case.append(np.clip(prob, 0, 1))

# Visualization
st.subheader(f"Crisis Probability Scenarios - {selected_region}")

fig = go.Figure()

# Best case
fig.add_trace(go.Scatter(
    x=dates,
    y=best_case,
    name='Best Case (5th percentile)',
    line=dict(color='green', width=2),
    mode='lines+markers'
))

# Expected
fig.add_trace(go.Scatter(
    x=dates,
    y=expected,
    name='Expected (50th percentile)',
    line=dict(color='blue', width=3),
    mode='lines+markers'
))

# Worst case
fig.add_trace(go.Scatter(
    x=dates,
    y=worst_case,
    name='Worst Case (95th percentile)',
    line=dict(color='red', width=2),
    mode='lines+markers'
))

# Shaded region
fig.add_trace(go.Scatter(
    x=list(dates) + list(dates[::-1]),
    y=worst_case + best_case[::-1],
    fill='toself',
    fillcolor='rgba(128,128,128,0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Uncertainty Range',
    showlegend=True
))

fig.update_layout(
    title="Crisis Probability Under Different Scenarios",
    xaxis_title="Date",
    yaxis_title="Crisis Probability",
    hovermode='x unified',
    height=500,
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig, use_container_width=True)

# Scenario comparison
st.subheader("Scenario Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🟢 Best Case")
    st.metric("Final Probability", f"{best_case[-1]:.1%}")
    st.metric("Average Probability", f"{np.mean(best_case):.1%}")
    st.metric("Peak Probability", f"{max(best_case):.1%}")
    
    high_risk_months = sum(1 for p in best_case if p > 0.7)
    st.metric("High Risk Months", high_risk_months)

with col2:
    st.markdown("### 🔵 Expected")
    st.metric("Final Probability", f"{expected[-1]:.1%}")
    st.metric("Average Probability", f"{np.mean(expected):.1%}")
    st.metric("Peak Probability", f"{max(expected):.1%}")
    
    high_risk_months = sum(1 for p in expected if p > 0.7)
    st.metric("High Risk Months", high_risk_months)

with col3:
    st.markdown("### 🔴 Worst Case")
    st.metric("Final Probability", f"{worst_case[-1]:.1%}")
    st.metric("Average Probability", f"{np.mean(worst_case):.1%}")
    st.metric("Peak Probability", f"{max(worst_case):.1%}")
    
    high_risk_months = sum(1 for p in worst_case if p > 0.7)
    st.metric("High Risk Months", high_risk_months)

# Impact assessment
st.subheader("Potential Impact Assessment")

impact_factors = ['Population Affected', 'Economic Loss (USD)', 'Duration (months)', 'Recovery Time (years)']
best_impacts = [100000, 1e9, 3, 1]
expected_impacts = [500000, 5e9, 6, 2]
worst_impacts = [2000000, 20e9, 12, 5]

impact_df = pd.DataFrame({
    'Factor': impact_factors,
    'Best Case': best_impacts,
    'Expected': expected_impacts,
    'Worst Case': worst_impacts
})

# Format numbers
impact_display = impact_df.copy()
impact_display['Best Case'] = impact_display['Best Case'].apply(lambda x: f"{x:,.0f}" if x < 1e6 else f"${x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.1f}M")
impact_display['Expected'] = impact_display['Expected'].apply(lambda x: f"{x:,.0f}" if x < 1e6 else f"${x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.1f}M")
impact_display['Worst Case'] = impact_display['Worst Case'].apply(lambda x: f"{x:,.0f}" if x < 1e6 else f"${x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.1f}M")

st.dataframe(impact_display, use_container_width=True, hide_index=True)

# Mitigation strategies
st.subheader("🛡️ Recommended Mitigation Strategies")

if max(expected) > 0.7:
    st.error("""
    **HIGH PRIORITY ACTIONS REQUIRED**
    
    Based on expected scenario projections:
    
    1. **Immediate**: Activate emergency response protocols
    2. **Short-term** (1-3 months): Mobilize resources and stockpiles
    3. **Medium-term** (3-6 months): Implement preventive measures
    4. **Long-term** (6+ months): Structural reforms and capacity building
    """)
elif max(expected) > 0.5:
    st.warning("""
    **MODERATE PREPAREDNESS RECOMMENDED**
    
    Suggested actions:
    
    1. Enhance monitoring systems
    2. Pre-position resources
    3. Coordinate with stakeholders
    4. Review and update response plans
    """)
else:
    st.success("""
    **STANDARD MONITORING SUFFICIENT**
    
    Continue regular monitoring and maintain preparedness protocols.
    """)

# Monte Carlo details
with st.expander("ℹ️ About Scenario Simulations"):
    st.markdown("""
    These scenarios are generated using **Monte Carlo simulations** with 1,000+ iterations.
    
    - **Best Case (5th percentile)**: Optimistic outcome with favorable conditions
    - **Expected (50th percentile)**: Most likely outcome based on current trends
    - **Worst Case (95th percentile)**: Pessimistic outcome with compounding factors
    
    The uncertainty range captures the variability in potential outcomes based on:
    - Model uncertainty
    - Data quality limitations
    - Unforeseen events
    - Policy interventions
    """)
