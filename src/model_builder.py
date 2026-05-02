import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD & SCALE THE DATA
# ==========================================
print("Loading split datasets...")
train_df = pd.read_csv("train_data_merged.csv", index_col='Date', parse_dates=True)
val_df = pd.read_csv("val_data_merged.csv", index_col='Date', parse_dates=True)
test_df = pd.read_csv("test_data_merged.csv", index_col='Date', parse_dates=True)

# Separate features
price_cols = ['Brent_Close', 'WTI_Close']
sent_cols = [col for col in train_df.columns if 'Sentiment' in col]
target_cols = ['Brent_Pct_Change', 'WTI_Pct_Change']

# Scaling inputs
scaler_price = StandardScaler()
scaler_sent = StandardScaler()

train_prices_scaled = scaler_price.fit_transform(train_df[price_cols])
train_sent_scaled = scaler_sent.fit_transform(train_df[sent_cols])

val_prices_scaled = scaler_price.transform(val_df[price_cols])
val_sent_scaled = scaler_sent.transform(val_df[sent_cols])

test_prices_scaled = scaler_price.transform(test_df[price_cols])
test_sent_scaled = scaler_sent.transform(test_df[sent_cols])

# ==========================================
# 2. SEQUENCE GENERATION (30-Day Look-Back)
# ==========================================
LOOK_BACK = 30 

def create_sequences(prices, sentiments, targets, look_back):
    X_price, X_sent, y = [], [], []
    for i in range(len(prices) - look_back):
        X_price.append(prices[i:(i + look_back)])
        X_sent.append(sentiments[i:(i + look_back)])
        y.append(targets.iloc[i + look_back].values)
    return np.array(X_price), np.array(X_sent), np.array(y)

print(f"Generating {LOOK_BACK}-day sequences...")
X_train_p, X_train_s, y_train = create_sequences(train_prices_scaled, train_sent_scaled, train_df[target_cols], LOOK_BACK)
X_val_p, X_val_s, y_val = create_sequences(val_prices_scaled, val_sent_scaled, val_df[target_cols], LOOK_BACK)
X_test_p, X_test_s, y_test = create_sequences(test_prices_scaled, test_sent_scaled, test_df[target_cols], LOOK_BACK)

# ==========================================
# 3. BUILD THE MULTI-MODAL HYBRID MODEL
# ==========================================
print("Building the Neural Network Architecture...")

# Branch A: Market Momentum (Price)
price_input = Input(shape=(LOOK_BACK, len(price_cols)), name="Historical_Prices")
lstm_out = LSTM(64, return_sequences=True, name="Price_LSTM")(price_input)

# Branch B: Geopolitical Shocks (Sentiment)
sent_input = Input(shape=(LOOK_BACK, len(sent_cols)), name="Geopolitical_Sentiment")
cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="News_CNN")(sent_input)

# Fusion: Attention Mechanism
attention_out = MultiHeadAttention(num_heads=4, key_dim=64, name="Contextual_Attention")(
    query=lstm_out, 
    value=cnn_out, 
    key=cnn_out
)

# Pooling
pooled_price = GlobalAveragePooling1D()(lstm_out)
pooled_attention = GlobalAveragePooling1D()(attention_out)

# Concatenate
merged = Concatenate(name="Fusion_Concatenation")([pooled_price, pooled_attention])

# Dense Layers
dense1 = Dense(64, activation='relu')(merged)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu')(dropout1)
final_output = Dense(2, activation='linear', name="Target_Predictions")(dense2)

# Compile
model = Model(inputs=[price_input, sent_input], outputs=final_output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

model.summary()

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
print("\n--- INITIATING TRAINING PHASE ---")

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_hybrid_model.keras", monitor='val_loss', save_best_only=True)

history = model.fit(
    x=[X_train_p, X_train_s],
    y=y_train,
    validation_data=([X_val_p, X_val_s], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

print("\nTraining Complete! Best model saved as 'best_hybrid_model.keras'.")

# ==========================================
# 5. PLOT LOSS CURVES
# ==========================================
print("Generating Loss Curves...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Training History', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('training_loss_curve.png')
plt.show()

print("Process finished. You are ready to evaluate the test set next.")