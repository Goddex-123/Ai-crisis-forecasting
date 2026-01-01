"""
Alerts Page
Real-time crisis alerts and warnings
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.title("🚨 Crisis Alerts")

st.markdown("""
Real-time monitoring of crisis warnings and risk alerts across all regions.
""")

# Generate sample alerts
alert_types = ['Critical', 'High', 'Medium']
regions = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania', 'Middle East']
crisis_types = ['Pandemic', 'Food Shortage', 'Climate Disaster', 'Economic Collapse']

alerts_data = []
for i in range(15):
    alert_date = datetime.now() - timedelta(days=np.random.randint(0, 30))
    alert_type = np.random.choice(alert_types, p=[0.2, 0.3, 0.5])
    
    alerts_data.append({
        'date': alert_date,
        'level': alert_type,
        'region': np.random.choice(regions),
        'crisis_type': np.random.choice(crisis_types),
        'risk_score': np.random.randint(50, 95),
        'probability': np.random.uniform(0.5, 0.95),
        'is_active': np.random.choice([True, False], p=[0.7, 0.3])
    })

df_alerts = pd.DataFrame(alerts_data)
df_alerts = df_alerts.sort_values('date', ascending=False)

# Filters
st.sidebar.subheader("Filters")

filter_level = st.sidebar.multiselect(
    "Alert Level",
    alert_types,
    default=alert_types
)

filter_region = st.sidebar.multiselect(
    "Region",
    regions,
    default=regions
)

filter_active = st.sidebar.checkbox("Show only active alerts", value=True)

# Apply filters
df_filtered = df_alerts[
    (df_alerts['level'].isin(filter_level)) &
    (df_alerts['region'].isin(filter_region))
]

if filter_active:
    df_filtered = df_filtered[df_filtered['is_active'] == True]

# Alert summary
st.subheader("Alert Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    critical_count = len(df_filtered[df_filtered['level'] == 'Critical'])
    st.metric(
        "🔴 Critical Alerts",
        critical_count,
        ""
    )

with col2:
    high_count = len(df_filtered[df_filtered['level'] == 'High'])
    st.metric(
        "🟠 High Alerts",
        high_count,
        ""
    )

with col3:
    medium_count = len(df_filtered[df_filtered['level'] == 'Medium'])
    st.metric(
        "🟡 Medium Alerts",
        medium_count,
        ""
    )

with col4:
    active_count = len(df_filtered[df_filtered['is_active'] == True])
    st.metric(
        "✅ Active Alerts",
        active_count,
        f"{(active_count/len(df_filtered)*100):.0f}%" if len(df_filtered) > 0 else "0%"
    )

st.markdown("---")

# Alert feed
st.subheader("Alert Feed")

for idx, alert in df_filtered.iterrows():
    # Color coding
    if alert['level'] == 'Critical':
        color_class = 'alert-critical'
        icon = '🔴'
    elif alert['level'] == 'High':
        color_class = 'alert-high'
        icon = '🟠'
    else:
        color_class = 'alert-medium'
        icon = '🟡'
    
    status = "🟢 ACTIVE" if alert['is_active'] else "⚪ RESOLVED"
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="{color_class}">
                <strong>{icon} {alert['level'].upper()} ALERT</strong> - {alert['crisis_type']} in {alert['region']}<br>
                Risk Score: {alert['risk_score']:.0f}/100 | Probability: {alert['probability']:.0%}<br>
                <small>Issued: {alert['date'].strftime('%Y-%m-%d %H:%M')} | Status: {status}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button(f"Details", key=f"btn_{idx}"):
                st.info(f"""
                **Alert Details**
                
                - Region: {alert['region']}
                - Crisis Type: {alert['crisis_type']}
                - Risk Score: {alert['risk_score']}/100
                - Probability: {alert['probability']:.1%}
                - Issued: {alert['date'].strftime('%Y-%m-%d %H:%M')}
                - Status: {'Active' if alert['is_active'] else 'Resolved'}
                """)
        
        st.markdown("<br>", unsafe_allow_html=True)

# Alert timeline
st.subheader("Alert Timeline")

fig = px.scatter(
    df_filtered,
    x='date',
    y='risk_score',
    color='level',
    size='probability',
    hover_data=['region', 'crisis_type'],
    color_discrete_map={
        'Critical': '#8B0000',
        'High': '#FF8C00',
        'Medium': '#FFFF00'
    },
    title="Alert History and Risk Scores"
)

fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Alert distribution
st.subheader("Alert Distribution")

col1, col2 = st.columns(2)

with col1:
    # By region
    region_counts = df_filtered['region'].value_counts()
    fig_region = px.bar(
        x=region_counts.index,
        y=region_counts.values,
        labels={'x': 'Region', 'y': 'Number of Alerts'},
        title="Alerts by Region",
        color=region_counts.values,
        color_continuous_scale='Reds'
    )
    fig_region.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    # By crisis type
    crisis_counts = df_filtered['crisis_type'].value_counts()
    fig_crisis = px.pie(
        values=crisis_counts.values,
        names=crisis_counts.index,
        title="Alerts by Crisis Type"
    )
    fig_crisis.update_layout(height=350)
    st.plotly_chart(fig_crisis, use_container_width=True)

# Alert notification settings
with st.expander("⚙️ Alert Settings"):
    st.markdown("### Email Notification Preferences")
    
    st.checkbox("Receive critical alerts", value=True)
    st.checkbox("Receive high priority alerts", value=True)
    st.checkbox("Receive medium priority alerts", value=False)
    
    st.selectbox(
        "Notification Frequency",
        ["Real-time", "Hourly digest", "Daily digest"]
    )
    
    st.button("Save Settings")
