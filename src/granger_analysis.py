import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

# Suppress warnings to keep the console output clean
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
print("Loading aligned dataset...")
df = pd.read_csv("aligned_energy_sentinels_dataset.csv", index_col='Date', parse_dates=True)

# Define our targets and predictors
countries = ['USA', 'Saudi Arabia', 'Russia', 'Canada', 'Iraq', 'China', 'UAE']
targets = ['Brent_Pct_Change', 'WTI_Pct_Change']
max_lags = 3  # Testing 1, 2, and 3 days out

# ==========================================
# 2. RUN GRANGER CAUSALITY TESTS
# ==========================================
print("\n--- GRANGER CAUSALITY TEST RESULTS ---")
print("Testing if Country Sentiment predicts Oil Price Volatility\n")

# statsmodels expects a 2D array: [Target (Y), Predictor (X)]
for target in targets:
    print(f"\n================ TARGET: {target} ================")
    for country in countries:
        sentiment_col = f"{country}_Sentiment"
        
        # Check if the column exists to avoid errors (e.g., if space replaced by underscore)
        if sentiment_col in df.columns and target in df.columns:
            # Drop any lingering NaNs just for the specific columns we are testing
            test_data = df[[target, sentiment_col]].dropna()
            
            print(f"\n--- {sentiment_col} ---")
            try:
                # Run the test silently (verbose=False) so we can format the output ourselves
                gc_res = grangercausalitytests(test_data, maxlag=max_lags, verbose=False)
                
                # Extract and print the p-values for each lag
                for lag in range(1, max_lags + 1):
                    # We look at the 'SSR based F test' p-value, which is standard
                    p_value = gc_res[lag][0]['ssr_ftest'][1]
                    
                    # If p-value is less than 0.05, it is statistically significant!
                    if p_value < 0.05:
                        print(f"  Lag {lag} Day(s): p-value = {p_value:.4f}  <--- *** SIGNIFICANT ***")
                    else:
                        print(f"  Lag {lag} Day(s): p-value = {p_value:.4f}")
            except Exception as e:
                print(f"  Error calculating: {e}")

print("\nLag Analysis Complete!")