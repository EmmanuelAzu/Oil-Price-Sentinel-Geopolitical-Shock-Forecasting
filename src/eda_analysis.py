import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD THE ALIGNED DATASET
# ==========================================
print("Loading aligned dataset...")
df = pd.read_csv("aligned_energy_sentinels_dataset.csv", index_col='Date', parse_dates=True)

# ==========================================
# 2. SENTIMENT CORRELATION MATRIX
# ==========================================
print("Generating Correlation Matrix...")
# Select only the sentiment columns and the target % change columns
cols_to_correlate = [col for col in df.columns if 'Sentiment' in col] + ['Brent_Pct_Change', 'WTI_Pct_Change']
corr_matrix = df[cols_to_correlate].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.5, vmax=0.5)
plt.title("Correlation Matrix: Geopolitical Sentiment vs. Oil Price Volatility")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# ==========================================
# 3. SHOCK IDENTIFICATION: 2022 INVASION
# ==========================================
print("Generating 2022 Shock Visualization...")
# Zoom in on the specific period of the 2022 supply disruptions (Jan 2022 - June 2022)
shock_df = df.loc['2022-01-01':'2022-06-30']

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the Oil Price on the primary Y-axis
color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Brent Crude Close Price (USD)', color=color)
ax1.plot(shock_df.index, shock_df['Brent_Close'], color=color, linewidth=2, label='Brent Price')
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary Y-axis for Sentiment
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Russia GDELT Sentiment Score', color=color)  
# We use a rolling average (e.g., 5 days) to smooth out the noise and see the trend
ax2.plot(shock_df.index, shock_df['Russia_Sentiment'].rolling(window=5).mean(), 
         color=color, linestyle='--', label='Russia Sentiment (5-Day Rolling)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Historical Shock Identification: 2022 Supply Disruptions")
fig.tight_layout()
plt.savefig("shock_2022.png")
plt.show()

print("EDA Complete! Check your folder for the saved graphs.")