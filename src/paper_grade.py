import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error

# ==========================================
# 1. ENHANCED FEATURE ENGINEERING (Per Paper Logic)
# ==========================================
def oil_specific_features(df):
    # Sentiment Indicators: Mean, Volatility, and Momentum
    sent_cols = [c for c in df.columns if 'Sentiment' in c]
    df['Sent_Momentum'] = df[sent_cols].diff().mean(axis=1) # Sentiment change rate
    df['Sent_Vol'] = df[sent_cols].std(axis=1)             # Sentiment disagreement
    
    # Financial Features: Returns and Volatility Clustering
    df['Return_Lag1'] = df['Brent_Pct_Change'].shift(1)
    df['Realized_Vol'] = df['Brent_Pct_Change'].rolling(window=5).std()
    
    return df.dropna()

train_feat = oil_specific_features(pd.read_csv("train_data_merged.csv", index_col='Date', parse_dates=True))
test_feat = oil_specific_features(pd.read_csv("test_data_merged.csv", index_col='Date', parse_dates=True))

# Target: The paper looks at fluctuations. We will predict the raw return.
target = 'Brent_Pct_Change'
features = ['Sent_Momentum', 'Sent_Vol', 'Return_Lag1', 'Realized_Vol'] + [c for c in train_feat.columns if 'Sentiment' in c]

# ==========================================
# 2. THE NOVEL FORECASTING ALGORITHM (XGBoost w/ Huber Loss)
# ==========================================
# Huber Loss behaves like MSE for small errors and MAE for large ones,
# making it less "cowardly" during market shocks.
model = xgb.XGBRegressor(
    objective='reg:pseudohubererror', 
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    alpha=10,  # L1 regularization to ignore noise
    lambda_=1  # L2 regularization for stability
)

model.fit(train_feat[features], train_feat[target])

# ==========================================
# 3. INTERPRETATION & EVALUATION
# ==========================================
preds = model.predict(test_feat[features])

# NEW INDICATOR: Absolute Directional Error (ADE)
# Does the model predict a large move when a large move actually happens?
large_move_threshold = 0.03
mask = np.abs(test_feat[target]) > large_move_threshold
big_move_acc = accuracy_score(np.sign(test_feat[target][mask]), np.sign(preds[mask]))

print(f"Overall Directional Accuracy: {accuracy_score(np.sign(test_feat[target]), np.sign(preds)):.2%}")
print(f"Accuracy on Extreme Shocks (>3%): {big_move_acc:.2%}")