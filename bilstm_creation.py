# =============================================================================
# Step 1: Import Libraries and Set Seeds for Reproducibility (Unchanged)
# =============================================================================

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import yfinance as yf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Dense, LSTM, Bidirectional, Dropout, Input, 
                                     Embedding, Flatten, Concatenate, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# Step 2: Collect all data regarding Bitcoin
# =============================================================================


ticker = "BTC-USD"

# Define date range
start_date = "2023-01-01"
end_date = "2023-12-31"
time_steps = 60

# Download 'Close' prices
data = yf.download(ticker, start=start_date, end=end_date)['Close']
data = data.reset_index()

# Drops data with missing values (needed for clarity and accuracy)
df = data.copy()
df.dropna(inplace=True)


# =============================================================================
# Step 3: Scale prices for the Bitcoin data
# =============================================================================

# Scale prices separately for each ticker to [0,1]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])
df['Scaled'] = scaled_data

# =============================================================================
# Step 4: Create Universal Bitcoin Sequences (With Scaled Prices only)
# =============================================================================

def create_universal_sequences(df, time_steps=60):
    X_seq, y = [], []
    scaled_prices = df['Scaled'].values

    for i in range(len(scaled_prices) - time_steps):
        X_seq.append(scaled_prices[i:i+time_steps])
        y.append(scaled_prices[i+time_steps])
            
    return np.array(X_seq), np.array(y)

time_steps = 60
X, y = create_universal_sequences(df, time_steps)
X = X.reshape(X.shape[0], time_steps, 1)

# =============================================================================
# Step 5: Build Universal Models: One with LSTM and One with BiLSTM
# =============================================================================
def build_universal_model_lstm(time_steps):

    # Input for metadata and sequencing of prices
    price_input = Input(shape=(time_steps, 1), name='price_input')
    
    # LSTM section (ESSENTIAL)
    x = LSTM(128, return_sequences=True)(price_input)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)

    # LSTM Output variable
    output = Dense(1)(x)
    
    model = Model(inputs=[price_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_universal_model_bilstm(time_steps):

    # Inputs for metadata and sequence
    price_input = Input(shape=(time_steps, 1), name='price_input')
    
    # Bidirectional LSTM (BiLSTM) Section (EVEN MORE ESSENTIAL)
    x = Bidirectional(LSTM(128, return_sequences=True))(price_input)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)
    
    # Combine the BiLSTM output with Dense Layers and Embedding  
    output = Dense(1)(x)
    
    model = Model(inputs=[price_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

universal_model_lstm = build_universal_model_lstm(time_steps)
universal_model_bilstm = build_universal_model_bilstm(time_steps)

print("-----------------------------------------------------------------------------")
print("Universal LSTM Model Summary:")
universal_model_lstm.summary()

print("-----------------------------------------------------------------------------")
print("\nUniversal BiLSTM Model Summary:")
universal_model_bilstm.summary()

# =============================================================================
# Step 6: Split Data into Training and Testing Sets - This is where I currently am at the moment
# =============================================================================

split_idx = int(0.8 * len(X))
X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

# =============================================================================
# Step 7: Set Up Callbacks
# =============================================================================

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

checkpoint_lstm_cb = ModelCheckpoint(os.path.join(model_dir, 'universal_lstm_best.keras'),
                                     monitor='val_loss', save_best_only=True, verbose=1)
checkpoint_bilstm_cb = ModelCheckpoint(os.path.join(model_dir, 'universal_bilstm_best.keras'),
                                       monitor='val_loss', save_best_only=True, verbose=1)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# =============================================================================
# Step 8: Train Both Models
# =============================================================================

print("-----------------------------------------------------------------------------")
print("\nTraining Universal LSTM Model...")
history_lstm = universal_model_lstm.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, checkpoint_lstm_cb, tensorboard_callback],
    verbose=1
)

print("-----------------------------------------------------------------------------")
print("\nTraining Universal BiLSTM Model...")
history_bilstm = universal_model_bilstm.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, checkpoint_bilstm_cb, tensorboard_callback],
    verbose=1
)

# =============================================================================
# Step 9: Load Best Models and Evaluate on Test Set
# =============================================================================

universal_model_lstm = load_model(os.path.join(model_dir, 'universal_lstm_best.keras'))
universal_model_bilstm = load_model(os.path.join(model_dir, 'universal_bilstm_best.keras'))

# Predictions from each model
preds_lstm = universal_model_lstm.predict(X_test)
preds_bilstm = universal_model_bilstm.predict(X_test)

# Ensemble: average predictions from both models
ensemble_preds = (preds_lstm + preds_bilstm) / 2.0

# Inverse scale predictions per sample using the corresponding scaler (per ticker)
def inverse_scale_predictions(preds, y_true, scaler):
    # Inverse transform both prediction and ground truth
    preds_inv = scaler.inverse_transform(preds)
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    
    # Flatten for plotting or RMSE calc
    return preds_inv.flatten(), y_true_inv.flatten()

# For each model, do the inverse transformation
lstm_preds_final, y_test_final = inverse_scale_predictions(preds_lstm, y_test, scaler)
bilstm_preds_final, _ = inverse_scale_predictions(preds_bilstm, y_test, scaler)
ensemble_preds_final, _ = inverse_scale_predictions(ensemble_preds, y_test, scaler)

def calculate_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

rmse_lstm = calculate_rmse(y_test_final, lstm_preds_final)
rmse_bilstm = calculate_rmse(y_test_final, bilstm_preds_final)
rmse_ensemble = calculate_rmse(y_test_final, ensemble_preds_final)

print("-----------------------------------------------------------------------------")
print(f"\nUniversal LSTM Model RMSE: {rmse_lstm:.4f}")
print(f"Universal BiLSTM Model RMSE: {rmse_bilstm:.4f}")
print(f"Ensemble Model RMSE: {rmse_ensemble:.4f}")
print("-----------------------------------------------------------------------------")

# =============================================================================
# Step 10: Visualization - Plot Predictions per Ticker
# =============================================================================

# Get unique tickers from test set stock IDs
unique_stock_ids = np.unique(X_test[1])
categories = df['Ticker'].astype('category').cat.categories

plt.figure(figsize=(18, 12))
plot_idx = 1
for stock_id in unique_stock_ids:
    ticker_name = categories[stock_id]
    # Find indices in test set corresponding to this ticker
    mask = (X_test[1] == stock_id)
    if np.sum(mask) == 0:
        continue
    actual = y_test_final[mask]
    pred_lstm = lstm_preds_final[mask]
    pred_bilstm = bilstm_preds_final[mask]
    pred_ensemble = ensemble_preds_final[mask]
    
    plt.subplot(3, 3, plot_idx)
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(pred_lstm, label='LSTM', linestyle='--', color='red')
    plt.plot(pred_bilstm, label='BiLSTM', linestyle='--', color='green')
    plt.plot(pred_ensemble, label='Ensemble', linestyle='-.', color='purple')
    plt.title(f'{ticker_name} Price Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plot_idx += 1

plt.tight_layout()
plt.show()



