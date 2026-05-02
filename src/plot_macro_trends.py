import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE DATA
# ==========================================
print("Loading aligned dataset...")
df = pd.read_csv("aligned_energy_sentinels_dataset.csv", index_col='Date', parse_dates=True)

# ==========================================
# 2. PLOT THE MACRO TRENDS
# ==========================================
print("Generating comprehensive visualization...")
fig, ax1 = plt.subplots(figsize=(16, 8))

# --- Primary Y-Axis: Brent Crude Price ---
ax1.set_xlabel('Date', fontweight='bold')
ax1.set_ylabel('Brent Crude Price (USD)', color='black', fontweight='bold')
# We plot Brent Close as a thick black line so it stands out against the sentiment colors
ax1.plot(df.index, df['Brent_Close'], color='black', linewidth=2.5, label='Brent Price')
ax1.tick_params(axis='y', labelcolor='black')

# --- Secondary Y-Axis: Country Sentiments ---
ax2 = ax1.twinx()
ax2.set_ylabel('GDELT Sentiment Score (30-Day Rolling Avg)', color='gray', fontweight='bold')

# Find all the sentiment columns automatically
sentiment_cols = [col for col in df.columns if 'Sentiment' in col]

# Plot each country's 30-day rolling average
for col in sentiment_cols:
    country_name = col.replace('_Sentiment', '')
    # Using rolling(30).mean() to smooth out the daily noise
    ax2.plot(df.index, df[col].rolling(window=30).mean(), label=country_name, alpha=0.8, linewidth=1.5)

plt.title('Global Oil Prices vs. Macro Geopolitical Sentiment Trends (2020 - 2026)', fontsize=14, fontweight='bold')

# Combine legends from both axes and place it outside the plot so it doesn't cover data
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', bbox_to_anchor=(1.05, 0.5))

# Adjust layout to fit the legend and save
plt.tight_layout()
plt.savefig("macro_trends_overview.png")
plt.show()

print("Graph successfully saved as macro_trends_overview.png!")