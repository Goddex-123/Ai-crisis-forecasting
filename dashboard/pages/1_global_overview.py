"""
Global Overview Page
Interactive world map with crisis hotspots
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.title("🌍 Global Crisis Overview")

st.markdown("""
Real-time view of crisis risk levels across all regions and domains.
""")

# Region mapping for visualization
region_coords = {
    'North America': {'lat': 54.5260, 'lon': -105.2551, 'country': 'USA'},
    'South America': {'lat': -8.7832, 'lon': -55.4915, 'country': 'Brazil'},
    'Europe': {'lat': 54.5260, 'lon': 15.2551, 'country': 'Germany'},
    'Africa': {'lat': -8.7832, 'lon': 34.5085, 'country': 'Kenya'},
    'Asia': {'lat': 34.0479, 'lon': 100.6197, 'country': 'China'},
    'Oceania': {'lat': -25.2744, 'lon': 133.7751, 'country': 'Australia'},
    'Middle East': {'lat': 29.3117, 'lon': 47.4818, 'country': 'Saudi Arabia'}
}

# Simulated current risk data
risk_data = []
for region, coords in region_coords.items():
    import random
    risk_data.append({
        'region': region,
        'lat': coords['lat'],
        'lon': coords['lon'],
        'overall_risk': random.randint(20, 85),
        'climate_risk': random.randint(15, 90),
        'health_risk': random.randint(10, 70),
        'food_risk': random.randint(20, 95),
        'economic_risk': random.randint(25, 80)
    })

df = pd.DataFrame(risk_data)

# Filters
st.sidebar.subheader("Filters")
selected_domain = st.sidebar.selectbox(
    "Risk Domain",
    ["Overall", "Climate", "Health", "Food", "Economic"]
)

# Map column selection
risk_column_map = {
    "Overall": "overall_risk",
    "Climate": "climate_risk",
    "Health": "health_risk",
    "Food": "food_risk",
    "Economic": "economic_risk"
}

risk_col = risk_column_map[selected_domain]

# Interactive Map
st.subheader(f"{selected_domain} Risk Map")

fig = px.scatter_geo(
    df,
    lat='lat',
    lon='lon',
    size=risk_col,
    color=risk_col,
    hover_name='region',
    hover_data={
        'lat': False,
        'lon': False,
        risk_col: ':.1f'
    },
    color_continuous_scale=['green', 'yellow', 'orange', 'red', 'darkred'],
    size_max=50,
    title=f"Global {selected_domain} Crisis Risk Distribution"
)

fig.update_layout(
    geo=dict(
        showland=True,
        landcolor='rgb(243, 243, 243)',
        coastlinecolor='rgb(204, 204, 204)',
        projection_type='natural earth'
    ),
    height=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True)

# Risk table
st.subheader("Regional Risk Breakdown")

display_df = df[['region', 'overall_risk', 'climate_risk', 'health_risk', 'food_risk', 'economic_risk']].copy()
display_df.columns = ['Region', 'Overall', 'Climate', 'Health', 'Food', 'Economic']
display_df = display_df.sort_values('Overall', ascending=False)

# Color code the table
def color_risk(val):
    if val >= 75:
        color = '#8B0000'
    elif val >= 50:
        color = '#FF8C00'
    elif val >= 25:
        color = '#FFFF00'
    else:
        color = '#00FF00'
    return f'background-color: {color}; color: {"white" if val >= 50 else "black"}'

styled_df = display_df.style.applymap(color_risk, subset=['Overall', 'Climate', 'Health', 'Food', 'Economic'])
st.dataframe(styled_df, use_container_width=True)

# Summary statistics
st.subheader("Global Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Highest Risk Region",
        display_df.iloc[0]['Region'],
        f"Score: {display_df.iloc[0]['Overall']:.1f}"
    )

with col2:
    st.metric(
        "Average Global Risk",
        f"{df['overall_risk'].mean():.1f}",
        f"Std: {df['overall_risk'].std():.1f}"
    )

with col3:
    critical_regions = (df['overall_risk'] >= 75).sum()
    st.metric(
        "Critical Risk Regions",
        critical_regions,
        f"{(critical_regions/len(df)*100):.0f}% of total"
    )
