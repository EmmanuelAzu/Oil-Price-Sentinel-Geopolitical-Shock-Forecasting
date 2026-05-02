import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA AND RE-CREATE SCALERS (Must match model_builder.py)
train_df = pd.read_csv("train_data_merged.csv", index_col='Date', parse_dates=True)
test_df = pd.read_csv("test_data_merged.csv", index_col='Date', parse_dates=True)

price_cols = ['Brent_Close', 'WTI_Close']
sent_cols = [col for col in train_df.columns if 'Sentiment' in col]
target_cols = ['Brent_Pct_Change', 'WTI_Pct_Change']

scaler_price = StandardScaler().fit(train_df[price_cols])
scaler_sent = StandardScaler().fit(train_df[sent_cols])

# 2. PREPARE TEST SEQUENCES
LOOK_BACK = 30
test_prices_scaled = scaler_price.transform(test_df[price_cols])
test_sent_scaled = scaler_sent.transform(test_df[sent_cols])

X_test_p, X_test_s, y_test = [], [], []
for i in range(len(test_df) - LOOK_BACK):
    X_test_p.append(test_prices_scaled[i:(i + LOOK_BACK)])
    X_test_s.append(test_sent_scaled[i:(i + LOOK_BACK)])
    y_test.append(test_df[target_cols].iloc[i + LOOK_BACK].values)

X_test_p, X_test_s, y_test = np.array(X_test_p), np.array(X_test_s), np.array(y_test)

# 3. LOAD MODEL AND PREDICT
print("Loading best model...")
model = tf.keras.models.load_model("best_hybrid_model.keras")
predictions = model.predict([X_test_p, X_test_s])

# 4. PLOT ACTUAL VS PREDICTED (Brent)
plt.figure(figsize=(15, 7))
plt.plot(test_df.index[LOOK_BACK:], y_test[:, 0], label='Actual % Change (Brent)', alpha=0.5, color='gray')
plt.plot(test_df.index[LOOK_BACK:], predictions[:, 0], label='Predicted % Change', color='red', linewidth=2)

plt.title('Test Set Performance: Brent Crude Oil Price Volatility', fontsize=14, fontweight='bold')
plt.ylabel('Daily % Change')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('test_results_comparison.png')
plt.show()

# Calculate Accuracy Metric
mae = np.mean(np.abs(y_test - predictions))
print(f"Mean Absolute Error on Test Set: {mae:.4f}")