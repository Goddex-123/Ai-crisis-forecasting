"""
Risk Analysis Page
Detailed risk analysis by domain and region
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.title("📊 Risk Analysis")

st.markdown("""
In-depth analysis of crisis risks across different domains and regions.
""")

# Generate sample data
regions = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania', 'Middle East']
domains = ['Climate', 'Health', 'Food', 'Economic']

# Create time series data
dates = pd.date_range(start='2023-01-01', end='2025-01-01', freq='MS')
time_series_data = []

for region in regions:
    for domain in domains:
        base_risk = np.random.uniform(30, 70)
        trend = np.random.uniform(-0.5, 0.5)
        
        for i, date in enumerate(dates):
            risk = base_risk + trend * i + np.random.normal(0, 5)
            risk = np.clip(risk, 0, 100)
            
            time_series_data.append({
                'date': date,
                'region': region,
                'domain': domain,
                'risk_score': risk
            })

df_time = pd.DataFrame(time_series_data)

# Sidebar filters
st.sidebar.subheader("Filters")
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    regions,
    default=['North America', 'Europe', 'Asia']
)

selected_domains = st.sidebar.multiselect(
    "Select Domains",
    domains,
    default=domains
)

# Filter data
df_filtered = df_time[
    (df_time['region'].isin(selected_regions)) &
    (df_time['domain'].isin(selected_domains))
]

# Risk trends over time
st.subheader("Risk Trends Over Time")

fig_trend = px.line(
    df_filtered,
    x='date',
    y='risk_score',
    color='domain',
    line_dash='region',
    title="Crisis Risk Evolution by Domain and Region",
    labels={'risk_score': 'Risk Score (0-100)', 'date': 'Date'}
)

fig_trend.update_layout(
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig_trend, use_container_width=True)

# Domain comparison
st.subheader("Current Risk by Domain")

latest_data = df_filtered[df_filtered['date'] == df_filtered['date'].max()]

fig_domain = px.box(
    latest_data,
    x='domain',
    y='risk_score',
    color='domain',
    title="Risk Score Distribution by Domain",
    labels={'risk_score': 'Risk Score (0-100)', 'domain': 'Domain'}
)

fig_domain.update_layout(height=400)

st.plotly_chart(fig_domain, use_container_width=True)

# Regional comparison
st.subheader("Current Risk by Region")

fig_region = px.bar(
    latest_data.groupby('region')['risk_score'].mean().reset_index(),
    x='region',
    y='risk_score',
    color='risk_score',
    color_continuous_scale=['green', 'yellow', 'orange', 'red'],
    title="Average Risk Score by Region",
    labels={'risk_score': 'Average Risk Score', 'region': 'Region'}
)

fig_region.update_layout(height=400)

st.plotly_chart(fig_region, use_container_width=True)

# Correlation heatmap
st.subheader("Cross-Domain Risk Correlation")

# Pivot data for correlation
pivot_data = latest_data.pivot_table(
    values='risk_score',
    index='region',
    columns='domain'
)

correlation = pivot_data.corr()

fig_corr = px.imshow(
    correlation,
    text_auto='.2f',
    aspect='auto',
    color_continuous_scale='RdYlGn_r',
    title="Domain Risk Correlation Matrix"
)

fig_corr.update_layout(height=500)

st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("""
**Interpretation**: Higher correlation indicates that risks in these domains tend to move together.
For example, economic and food crises are often correlated.
""")

# Key insights
st.subheader("📌 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Highest Risk Domain**: {latest_data.groupby('domain')['risk_score'].mean().idxmax()}
    
    Average Score: {latest_data.groupby('domain')['risk_score'].mean().max():.1f}/100
    """)

with col2:
    st.warning(f"""
    **Highest Risk Region**: {latest_data.groupby('region')['risk_score'].mean().idxmax()}
    
    Average Score: {latest_data.groupby('region')['risk_score'].mean().max():.1f}/100
    """)
