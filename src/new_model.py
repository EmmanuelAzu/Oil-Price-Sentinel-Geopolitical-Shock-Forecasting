import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================
train_df = pd.read_csv("train_data_merged.csv", index_col='Date', parse_dates=True)
test_df = pd.read_csv("test_data_merged.csv", index_col='Date', parse_dates=True)

target_col = 'Brent_Pct_Change' # Predicting next-day return

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    # a) Lagged Returns: r(t-1), r(t-2)
    df['lag_1'] = df[target_col].shift(1)
    df['lag_2'] = df[target_col].shift(2)
    
    # b) Rolling Statistics (5-day momentum)
    df['rolling_mean_5'] = df[target_col].rolling(window=5).mean()
    df['rolling_std_5'] = df[target_col].rolling(window=5).std()
    
    # c) Sentiment Volatility
    sent_cols = [c for c in df.columns if 'Sentiment' in c]
    df['sent_std'] = df[sent_cols].std(axis=1)
    df['sent_mean'] = df[sent_cols].mean(axis=1)
    
    # d) Time Features
    df['day_of_week'] = df.index.dayofweek
    
    return df.dropna()

train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

# Define X (features) and y (target)
features = ['lag_1', 'lag_2', 'rolling_mean_5', 'rolling_std_5', 'sent_std', 'sent_mean', 'day_of_week'] + \
           [c for c in train_df.columns if 'Sentiment' in c]

X_train, y_train = train_feat[features], train_feat[target_col]
X_test, y_test = test_feat[features], test_feat[target_col]

# ==========================================
# 3. TRAINING (XGBoost)
# ==========================================
print("Training Production-Grade XGBoost Model...")

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    objective='reg:absoluteerror' # Robust to outliers
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ==========================================
# 4. EVALUATION
# ==========================================
preds = model.predict(X_test)

# Directional Accuracy
actual_dir = np.sign(y_test)
pred_dir = np.sign(preds)
dir_acc = accuracy_score(actual_dir, pred_dir)

print(f"Directional Accuracy: {dir_acc:.2%}")
print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")

# ==========================================
# 5. VISUALIZE RESULTS
# ==========================================
plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label='Actual Return', alpha=0.4, color='gray')
plt.plot(y_test.index, preds, label='XGBoost Prediction', color='blue', linewidth=1.5)
plt.title('XGBoost Strategy: Brent Return Predictions', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig('xgboost_results.png')
plt.show()

# Feature Importance Plot
xgb.plot_importance(model, max_num_features=10)
plt.title('Which Sentiment/Market signals actually matter?')
plt.show()