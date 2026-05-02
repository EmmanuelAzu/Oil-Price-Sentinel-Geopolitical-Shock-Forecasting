import yfinance as yf
import pandas as pd
import glob

# ==========================================
# 1. FETCH YAHOO FINANCE PRICES (IN-MEMORY)
# ==========================================
print("Pulling Yahoo Finance prices into memory...")
tickers = ['BZ=F', 'CL=F']
oil_data = yf.download(tickers, start="2020-01-01", end="2026-04-30")

brent_close = oil_data['Close']['BZ=F'].dropna().rename("Brent_Close")
wti_close = oil_data['Close']['CL=F'].dropna().rename("WTI_Close")

finance_df = pd.concat([brent_close, wti_close], axis=1)
finance_df.index = pd.to_datetime(finance_df.index).tz_localize(None)

# ==========================================
# 2. LOAD YOUR REAL GDELT CSV
# ==========================================
print("Loading real GDELT Sentiment Data...")
# Find the exact filename since it has numbers at the end
gdelt_filename = glob.glob("gdelt_sentiment_2020*.csv")[0] 
gdelt_df = pd.read_csv(gdelt_filename)

gdelt_df['Date'] = pd.to_datetime(gdelt_df['Date'])
sentiment_df = gdelt_df.pivot(index='Date', columns='Country', values='Sentiment')
sentiment_df.columns = [f"{col}_Sentiment" for col in sentiment_df.columns]

# ==========================================
# 3. MERGE AND OVERWRITE DUMMY FILES
# ==========================================
print("Aligning real sentiment with oil prices...")
master_df = finance_df.join(sentiment_df, how='inner')
master_df['Brent_Pct_Change'] = master_df['Brent_Close'].pct_change()
master_df['WTI_Pct_Change'] = master_df['WTI_Close'].pct_change()
master_df = master_df.dropna()

print("New train/val/test splits...")
total_samples = len(master_df)
train_end = int(total_samples * 0.70)
val_end = int(total_samples * 0.85)

master_df.iloc[:train_end].to_csv("train_data_merged.csv")
master_df.iloc[train_end:val_end].to_csv("val_data_merged.csv")
master_df.iloc[val_end:].to_csv("test_data_merged.csv")
master_df.to_csv("aligned_energy_sentinels_dataset.csv")

print("Success! Your files are now populated with REAL geopolitical data.")