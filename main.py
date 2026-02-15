import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# =====================================
# Read the data 
# =====================================

df = pd.read_csv('can_data.csv')

# Test if data is in the dataframe
print(f'Loaded {len(df)} rows and {len(df.columns)} columns')
print(df.head())
print('\n')
# print(df.dtypes)
# missing_data = df.isnull().sum()
# print(missing_data[missing_data > 0]) 

# =====================================
# Clean the data
# =====================================
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
print(f"   First timestamp: {df['datetime'].iloc[0]}")
print(f"   Last timestamp: {df['datetime'].iloc[-1]}")

df['time_elapsed'] = df['timestamp'] - df['timestamp'].min()
print(f"   Total session duration: {df['time_elapsed'].max():.1f} seconds")

# i want to predict engine analysis, so just drop any rows that don't have values
critical_columns = ['RPM', 'TPS', 'MAP', 'Lambda']
df_clean = df.dropna(subset=critical_columns)
df_clean = df_clean[df_clean['RPM'] > 0]
# Forward fill for missing values
df_clean = df_clean.fillna(method='ffill')

# =====================================
# Insights
# =====================================

# 1. How well is the engine breathing 
df_clean['volumetric_efficiency'] = (df_clean['MAP'] / df_clean['Barometer']) * 100
print(f"   Average efficiency: {df_clean['volumetric_efficiency'].mean():.2f}%")
print(f"   Peak efficiency: {df_clean['volumetric_efficiency'].max():.2f}%")

# 2. Acceleration Detection
df_clean['rpm_change'] = df_clean['RPM'].diff() 
df_clean['acceleration_rate'] = df_clean['rpm_change'] / df_clean['time_elapsed'].diff()
df_clean['is_accelerating'] = (df_clean['rpm_change'] > 100) & (df_clean['TPS'] > 50)
num_accel_events = df_clean['is_accelerating'].sum()
print(f"Found {num_accel_events} acceleration data points")
print(f"   Accelerated for {(num_accel_events/len(df_clean)*100):.1f}% of the session")

# 3. Gear shift detection
shift_threshold = -500
potential_shifts = df_clean[
    (df_clean['rpm_change'] < shift_threshold) & 
    (df_clean['TPS'] > 30)  
].copy()

print(f"Detected {len(potential_shifts)} gear shifts")
print(f"Average RPM drop: {potential_shifts['rpm_change'].mean():.0f} RPM")
print(f"Shift RPM range: {potential_shifts['RPM'].min():.0f} - {potential_shifts['RPM'].max():.0f}")

# Shift quality
potential_shifts['shift_rpm_before'] = potential_shifts['RPM'] - potential_shifts['rpm_change']
avg_shift_rpm = potential_shifts['shift_rpm_before'].mean()
print(f"   Average shift point: {avg_shift_rpm:.0f} RPM")

# Find the best shift point
df_clean['power_index'] = (df_clean['RPM'] / 1000) * (df_clean['MAP'] / 10) * (df_clean['TPS'] / 100)
optimal_shift_rpm = df_clean[df_clean['power_index'] > df_clean['power_index'].quantile(0.9)]['RPM'].min()
print(f"Recommended shift point: {optimal_shift_rpm:.0f} RPM (for max power)")

if avg_shift_rpm < optimal_shift_rpm - 500:
    print(f"Driver is shifting {optimal_shift_rpm - avg_shift_rpm:.0f} RPM too early!")
    print(f"Shifting later could improve acceleration")

# 4. Driver Aggression Score
# CALCULATE THE REQUIRED COLUMNS FIRST:
df_clean['tps_change'] = df_clean['TPS'].diff()
df_clean['map_change_rate'] = df_clean['MAP'].diff() / df_clean['time_elapsed'].diff()

# NOW calculate aggression score:
df_clean['aggression_score'] = (
    (df_clean['TPS'] / 100) * 0.3 +                  
    (df_clean['tps_change'].abs() / 10).clip(0, 1) * 0.2 +  
    (df_clean['RPM'] / df_clean['RPM'].max()) * 0.3 +  
    (df_clean['map_change_rate'].abs() / 5).clip(0, 1) * 0.2  
)

print(f"Driving aggression analyzed")
print(f"Average aggression: {df_clean['aggression_score'].mean():.3f}")
print(f"Peak aggression: {df_clean['aggression_score'].max():.3f}")

# Driving style analysis
aggressive_threshold = 0.5
aggressive_moments = df_clean[df_clean['aggression_score'] > aggressive_threshold]
smooth_moments = df_clean[df_clean['aggression_score'] < 0.3]

print(f"\n   Driving style breakdown:")
print(f"   - Smooth driving: {len(smooth_moments)/len(df_clean)*100:.1f}%")
print(f"   - Moderate: {(len(df_clean)-len(aggressive_moments)-len(smooth_moments))/len(df_clean)*100:.1f}%")
print(f"   - Aggressive: {len(aggressive_moments)/len(df_clean)*100:.1f}%")

# Overall analysis
avg_aggression = df_clean['aggression_score'].mean()
if avg_aggression < 0.3:
    print(f"\nAssessment: CONSERVATIVE driver - good")
elif avg_aggression < 0.5:
    print(f"\nAssessment: BALANCED driver - good")
else:
    print(f"\nAssessment: AGGRESSIVE driver - pushing a bit hard")

# =====================================
# Graphs/Charts
# =====================================


# 1. Gear shift analysis graph
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df_clean['time_elapsed'],
    y=df_clean['RPM'],
    mode='lines',
    name='Engine RPM',
    line=dict(color='#3498db', width=2),
    hovertemplate='Time: %{x:.2f}s<br>RPM: %{y:.0f}<extra></extra>'
))

fig1.add_trace(go.Scatter(
    x=potential_shifts['time_elapsed'],
    y=potential_shifts['RPM'],
    mode='markers',
    name='Detected Shifts',
    marker=dict(
        color='red', 
        size=12, 
        symbol='x',
        line=dict(width=2)
    ),
    hovertemplate='Shift at %{x:.2f}s<br>RPM: %{y:.0f}<extra></extra>'
))

fig1.add_hline(
    y=optimal_shift_rpm, 
    line_dash="dash", 
    line_color="green",
    line_width=2,
    annotation_text=f"Optimal Shift Point: {optimal_shift_rpm:.0f} RPM",
    annotation_position="right"
)

fig1.add_hline(
    y=avg_shift_rpm, 
    line_dash="dot", 
    line_color="orange",
    line_width=2,
    annotation_text=f"Actual Avg Shift: {avg_shift_rpm:.0f} RPM",
    annotation_position="left"
)

fig1.update_layout(
    title={
        'text': 'Automatic Gear Shift Detection & Optimization',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': '#2c3e50'}
    },
    xaxis_title='Time Elapsed (seconds)',
    yaxis_title='Engine RPM',
    height=600,
    template='plotly_white',
    hovermode='closest',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
)

# Add annotation explaining the insight
fig1.add_annotation(
    text=f"Detected {len(potential_shifts)} shifts | Shifting {abs(optimal_shift_rpm - avg_shift_rpm):.0f} RPM too early!",
    xref="paper", yref="paper",
    x=0.5, y=-0.15,
    showarrow=False,
    font=dict(size=12, color='red'),
    xanchor='center'
)

fig1.write_html('graph1_shift_detection.html')

df_sample = df_clean.iloc[::5].copy()

fig2 = go.Figure()

# Create scatter plot with color based on aggression score
fig2.add_trace(go.Scatter(
    x=df_sample['time_elapsed'],
    y=df_sample['RPM'],
    mode='markers',
    marker=dict(
        size=6,
        color=df_sample['aggression_score'],
        colorscale='Reds',  # White to Red gradient
        showscale=True,
        colorbar=dict(
            title="Aggression<br>Score",
            thickness=20,
            len=0.7,
            x=1.02
        ),
        cmin=0,
        cmax=1,
        line=dict(width=0.5, color='rgba(0,0,0,0.3)')
    ),
    text=df_sample['aggression_score'].round(3),
    hovertemplate='Time: %{x:.2f}s<br>RPM: %{y:.0f}<br>Aggression: %{text}<extra></extra>'
))

# Add threshold lines
fig2.add_hline(
    y=df_clean['RPM'].mean(), 
    line_dash="dash", 
    line_color="gray",
    annotation_text=f"Average RPM: {df_clean['RPM'].mean():.0f}",
    annotation_position="right"
)

# Add zones with colored rectangles
fig2.add_hrect(
    y0=0, y1=3000,
    fillcolor="green", opacity=0.1,
    line_width=0,
    annotation_text="Cruise Zone", annotation_position="top left"
)

fig2.add_hrect(
    y0=6000, y1=df_clean['RPM'].max(),
    fillcolor="red", opacity=0.1,
    line_width=0,
    annotation_text="Performance Zone", annotation_position="top left"
)

fig2.update_layout(
    title={
        'text': 'Driving Aggression Analysis: Session Timeline',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': '#2c3e50'}
    },
    xaxis_title='Time Elapsed (seconds)',
    yaxis_title='Engine RPM',
    height=600,
    template='plotly_white',
    hovermode='closest'
)

# Add summary annotation
fig2.add_annotation(
    text=f"Aggressive Driving: {len(aggressive_moments)/len(df_clean)*100:.1f}% | Average Score: {avg_aggression:.3f}",
    xref="paper", yref="paper",
    x=0.5, y=-0.12,
    showarrow=False,
    font=dict(size=12, color='#e74c3c'),
    xanchor='center'
)

fig2.write_html('graph2_aggression_timeline.html')
print("Saved: graph2_aggression_timeline.html")

# =====================================
# GRAPH 3: RPM vs Throttle Position (Performance Map)
# =====================================
print("\nCreating Graph 3: RPM vs Throttle Performance Map...")

fig3 = px.scatter(
    df_clean,
    x='RPM',
    y='TPS',
    color='aggression_score',
    color_continuous_scale='RdYlGn_r',  # Red (aggressive) to Green (smooth)
    title='Engine Performance Map: RPM vs Throttle Position',
    labels={
        'RPM': 'Engine RPM',
        'TPS': 'Throttle Position (%)',
        'aggression_score': 'Aggression Score'
    },
    hover_data={
        'RPM': ':.0f',
        'TPS': ':.1f',
        'aggression_score': ':.3f',
        'MAP': ':.2f',
        'Lambda': ':.3f'
    }
)

# Add shift points as annotations
for idx, shift in potential_shifts.head(10).iterrows():  # Show first 10 shifts
    fig3.add_annotation(
        x=shift['RPM'],
        y=shift['TPS'],
        text="⬆",
        showarrow=False,
        font=dict(size=20, color='blue'),
        opacity=0.6
    )

fig3.update_layout(
    title_font_size=20,
    xaxis_title_font_size=14,
    yaxis_title_font_size=14,
    height=600,
    template='plotly_white'
)

# Add explanation
fig3.add_annotation(
    text="Red = Aggressive | Green = Smooth | Blue arrows = Gear shifts",
    xref="paper", yref="paper",
    x=0.5, y=-0.15,
    showarrow=False,
    font=dict(size=11)
)

fig3.write_html('graph3_performance_map.html')
print("Saved: graph3_performance_map.html")

# =====================================
# GRAPH 4: Multi-Panel Dashboard
# =====================================
print("\nCreating Graph 4: Performance Dashboard...")

fig4 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Shift Quality: Actual vs Optimal',
        'Aggression Distribution',
        'Volumetric Efficiency Over Time',
        'Performance Metrics Summary'
    ),
    specs=[
        [{'type': 'scatter'}, {'type': 'histogram'}],
        [{'type': 'scatter'}, {'type': 'bar'}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.15
)

# Subplot 1: Shift Quality (scatter showing shifts vs optimal line)
if len(potential_shifts) > 0:
    fig4.add_trace(
        go.Scatter(
            x=list(range(len(potential_shifts))),
            y=potential_shifts['shift_rpm_before'],
            mode='markers',
            name='Actual Shifts',
            marker=dict(color='red', size=10, symbol='circle')
        ),
        row=1, col=1
    )
    
    # Add optimal line
    fig4.add_hline(
        y=optimal_shift_rpm,
        line_dash="dash",
        line_color="green",
        row=1, col=1
    )
    
    fig4.add_annotation(
        text=f"Optimal: {optimal_shift_rpm:.0f}",
        xref="x", yref="y",
        x=len(potential_shifts)/2, y=optimal_shift_rpm,
        showarrow=False,
        font=dict(size=10, color='green'),
        row=1, col=1
    )

# Subplot 2: Aggression Distribution (histogram)
fig4.add_trace(
    go.Histogram(
        x=df_clean['aggression_score'],
        nbinsx=30,
        name='Aggression',
        marker_color='rgba(231, 76, 60, 0.7)',
        showlegend=False
    ),
    row=1, col=2
)

# Add vertical line for threshold
fig4.add_vline(
    x=aggressive_threshold,
    line_dash="dash",
    line_color="red",
    row=1, col=2
)

# Subplot 3: Volumetric Efficiency over time
fig4.add_trace(
    go.Scatter(
        x=df_clean['time_elapsed'].iloc[::10],
        y=df_clean['volumetric_efficiency'].iloc[::10],
        mode='lines',
        name='Vol. Efficiency',
        line=dict(color='#2ecc71', width=2),
        showlegend=False
    ),
    row=2, col=1
)

# Add average line
avg_vol_eff = df_clean['volumetric_efficiency'].mean()
fig4.add_hline(
    y=avg_vol_eff,
    line_dash="dot",
    line_color="gray",
    row=2, col=1
)

# Subplot 4: Summary Bar Chart
metrics = {
    'Avg Shift RPM (÷100)': avg_shift_rpm / 100,
    'Peak RPM (÷100)': df_clean['RPM'].max() / 100,
    'Avg Vol. Eff. %': avg_vol_eff,
    'Aggressive % (×10)': (len(aggressive_moments)/len(df_clean)*100) * 10
}

fig4.add_trace(
    go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker_color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
        text=[f"{v:.0f}" for v in metrics.values()],
        textposition='auto',
        showlegend=False
    ),
    row=2, col=2
)

# Update axes labels
fig4.update_xaxes(title_text="Shift Number", row=1, col=1)
fig4.update_yaxes(title_text="Shift RPM", row=1, col=1)

fig4.update_xaxes(title_text="Aggression Score", row=1, col=2)
fig4.update_yaxes(title_text="Frequency", row=1, col=2)

fig4.update_xaxes(title_text="Time (seconds)", row=2, col=1)
fig4.update_yaxes(title_text="Efficiency %", row=2, col=1)

fig4.update_xaxes(title_text="Metric", row=2, col=2)
fig4.update_yaxes(title_text="Value", row=2, col=2)

# Overall layout
fig4.update_layout(
    title_text="Performance Analysis Dashboard",
    title_font_size=20,
    height=800,
    showlegend=False,
    template='plotly_white'
)

fig4.write_html('graph4_dashboard.html')
print("Saved: graph4_dashboard.html")