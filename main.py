import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# =====================================
# STEP 1: Load the Data
# =====================================
print("Loading data...")
df = pd.read_csv('can_data.csv')
print(f"Loaded {len(df)} rows and {len(df.columns)} columns\n")

# =====================================
# STEP 2: Clean the Data
# =====================================
print("Cleaning data...")

# Convert timestamps
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['time_elapsed'] = df['timestamp'] - df['timestamp'].min()

print(f"   First timestamp: {df['datetime'].iloc[0]}")
print(f"   Last timestamp: {df['datetime'].iloc[-1]}")
print(f"   Session duration: {df['time_elapsed'].max():.1f} seconds")

# Remove rows with missing critical data
critical_columns = ['RPM', 'TPS', 'MAP', 'Lambda']
df_clean = df.dropna(subset=critical_columns)
df_clean = df_clean[df_clean['RPM'] > 0]
df_clean = df_clean.ffill()

print(f"   Cleaned dataset: {len(df_clean)} rows")
print(f"   Removed: {len(df) - len(df_clean)} rows\n")

# =====================================
# STEP 3: Derive Insights
# =====================================
print("Deriving insights...\n")

# Insight 1: Volumetric Efficiency
df_clean['volumetric_efficiency'] = (df_clean['MAP'] / df_clean['Barometer']) * 100
print(f"1. Volumetric Efficiency:")
print(f"   Average: {df_clean['volumetric_efficiency'].mean():.2f}%")
print(f"   Peak: {df_clean['volumetric_efficiency'].max():.2f}%\n")

# Insight 2: Acceleration Detection
df_clean['rpm_change'] = df_clean['RPM'].diff()
df_clean['acceleration_rate'] = df_clean['rpm_change'] / df_clean['time_elapsed'].diff()
df_clean['is_accelerating'] = (df_clean['rpm_change'] > 100) & (df_clean['TPS'] > 50)
num_accel_events = df_clean['is_accelerating'].sum()
print(f"2. Acceleration Detection:")
print(f"   Acceleration events: {num_accel_events}")
print(f"   Time accelerating: {(num_accel_events/len(df_clean)*100):.1f}%\n")

# Insight 3: Gear Shift Detection
shift_threshold = -500
potential_shifts = df_clean[
    (df_clean['rpm_change'] < shift_threshold) & 
    (df_clean['TPS'] > 30)
].copy()

print(f"3. Gear Shift Detection:")
print(f"   Detected: {len(potential_shifts)} shifts")
print(f"   Average RPM drop: {potential_shifts['rpm_change'].mean():.0f} RPM")
print(f"   Shift range: {potential_shifts['RPM'].min():.0f} - {potential_shifts['RPM'].max():.0f} RPM")

# Calculate shift quality
potential_shifts['shift_rpm_before'] = potential_shifts['RPM'] - potential_shifts['rpm_change']
avg_shift_rpm = potential_shifts['shift_rpm_before'].mean()
print(f"   Average shift point: {avg_shift_rpm:.0f} RPM")

# Find optimal shift point
df_clean['power_index'] = (df_clean['RPM'] / 1000) * (df_clean['MAP'] / 10) * (df_clean['TPS'] / 100)
optimal_shift_rpm = df_clean[df_clean['power_index'] > df_clean['power_index'].quantile(0.9)]['RPM'].min()
print(f"   Optimal shift point: {optimal_shift_rpm:.0f} RPM")

if avg_shift_rpm < optimal_shift_rpm - 500:
    print(f"Shifting {optimal_shift_rpm - avg_shift_rpm:.0f} RPM too early!\n")
else:
    print(f"Shift timing is good!\n")

# Insight 4: Driving Aggression Score
df_clean['tps_change'] = df_clean['TPS'].diff()
df_clean['map_change_rate'] = df_clean['MAP'].diff() / df_clean['time_elapsed'].diff()
df_clean['aggression_score'] = (
    (df_clean['TPS'] / 100) * 0.3 +
    (df_clean['tps_change'].abs() / 10).clip(0, 1) * 0.2 +
    (df_clean['RPM'] / df_clean['RPM'].max()) * 0.3 +
    (df_clean['map_change_rate'].abs() / 5).clip(0, 1) * 0.2
)

print(f"4. Driving Aggression Score:")
print(f"   Average: {df_clean['aggression_score'].mean():.3f}")
print(f"   Peak: {df_clean['aggression_score'].max():.3f}")

aggressive_threshold = 0.5
aggressive_moments = df_clean[df_clean['aggression_score'] > aggressive_threshold]
smooth_moments = df_clean[df_clean['aggression_score'] < 0.3]

print(f"   Driving style breakdown:")
print(f"   - Smooth: {len(smooth_moments)/len(df_clean)*100:.1f}%")
print(f"   - Moderate: {(len(df_clean)-len(aggressive_moments)-len(smooth_moments))/len(df_clean)*100:.1f}%")
print(f"   - Aggressive: {len(aggressive_moments)/len(df_clean)*100:.1f}%")

avg_aggression = df_clean['aggression_score'].mean()
if avg_aggression < 0.3:
    print(f"   Assessment: CONSERVATIVE driver\n")
elif avg_aggression < 0.5:
    print(f"   Assessment: BALANCED driver\n")
else:
    print(f"   Assessment: AGGRESSIVE driver\n")

# =====================================
# STEP 4: Create Graphs
# =====================================


# Graph 1: Shift Detection
print("\n1. Creating shift detection graph...")
plt.figure(figsize=(12, 6))

plt.plot(df_clean['time_elapsed'], df_clean['RPM'], 
         color='steelblue', linewidth=1.5, label='Engine RPM')

plt.scatter(potential_shifts['time_elapsed'], potential_shifts['RPM'], 
           color='red', s=100, marker='x', linewidth=3, 
           label=f'Shifts ({len(potential_shifts)} detected)', zorder=5)

plt.axhline(y=optimal_shift_rpm, color='green', linestyle='--', 
           linewidth=2, label=f'Optimal: {optimal_shift_rpm:.0f} RPM')
plt.axhline(y=avg_shift_rpm, color='orange', linestyle=':', 
           linewidth=2, label=f'Average: {avg_shift_rpm:.0f} RPM')

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Engine RPM', fontsize=12)
plt.title('Gear Shift Detection & Analysis', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph1_shift_detection.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: graph1_shift_detection.png")

# Graph 2: RPM Distribution
print("\n2. Creating RPM distribution graph...")
plt.figure(figsize=(12, 6))

plt.hist(df_clean['RPM'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')

plt.axvline(x=3000, color='orange', linestyle='--', linewidth=2, label='Normal (3000)')
plt.axvline(x=6000, color='red', linestyle='--', linewidth=2, label='Performance (6000)')
plt.axvline(x=9000, color='darkred', linestyle='--', linewidth=2, label='High Perf (9000)')

plt.xlabel('Engine RPM', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('RPM Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graph2_rpm_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: graph2_rpm_distribution.png")

# Graph 3: Performance Map
print("\n3. Creating performance map...")
plt.figure(figsize=(12, 6))

scatter = plt.scatter(df_clean['RPM'], df_clean['TPS'], 
                     c=df_clean['aggression_score'], 
                     cmap='YlOrRd', s=10, alpha=0.6)

cbar = plt.colorbar(scatter)
cbar.set_label('Aggression Score', fontsize=11)

plt.scatter(potential_shifts['RPM'], potential_shifts['TPS'], 
           color='blue', s=50, marker='^', edgecolor='black',
           label='Gear Shifts', zorder=5)

plt.xlabel('Engine RPM', fontsize=12)
plt.ylabel('Throttle Position (%)', fontsize=12)
plt.title('Performance Map: RPM vs Throttle', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph3_performance_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: graph3_performance_map.png")

# Graph 4: Dashboard
print("\n4. Creating summary dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Shift Quality
ax1 = axes[0, 0]
shift_numbers = list(range(len(potential_shifts)))
ax1.scatter(shift_numbers, potential_shifts['shift_rpm_before'], 
           color='red', s=80, alpha=0.7)
ax1.axhline(y=optimal_shift_rpm, color='green', linestyle='--', 
           linewidth=2, label=f'Optimal: {optimal_shift_rpm:.0f}')
ax1.set_xlabel('Shift Number')
ax1.set_ylabel('Shift RPM')
ax1.set_title('Shift Quality', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Aggression Distribution
ax2 = axes[0, 1]
ax2.hist(df_clean['aggression_score'], bins=30, 
        color='coral', alpha=0.7, edgecolor='black')
ax2.axvline(x=aggressive_threshold, color='red', linestyle='--', 
           linewidth=2, label=f'Threshold: {aggressive_threshold}')
ax2.set_xlabel('Aggression Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Aggression Distribution', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Subplot 3: Volumetric Efficiency
ax3 = axes[1, 0]
ax3.plot(df_clean['time_elapsed'][::10], 
        df_clean['volumetric_efficiency'][::10],
        color='green', linewidth=2)
ax3.axhline(y=df_clean['volumetric_efficiency'].mean(), 
           color='gray', linestyle='--', linewidth=2,
           label=f"Avg: {df_clean['volumetric_efficiency'].mean():.1f}%")
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Efficiency (%)')
ax3.set_title('Volumetric Efficiency', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Summary Metrics
ax4 = axes[1, 1]
metrics = ['Avg Shift\n÷100', 'Peak RPM\n÷100', 'Avg Eff.\n%', 'Aggressive\n%']
values = [
    avg_shift_rpm / 100,
    df_clean['RPM'].max() / 100,
    df_clean['volumetric_efficiency'].mean(),
    (len(aggressive_moments) / len(df_clean) * 100)
]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

ax4.set_ylabel('Value')
ax4.set_title('Key Metrics', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graph4_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: graph4_dashboard.png")

