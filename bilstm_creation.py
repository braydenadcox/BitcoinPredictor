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

# Download 'Close' prices for all tickers in one call
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
X = X.reshape(X.shape[0], time_steps, 1)  # add feature dimension

# =============================================================================
# Step 5: Build Universal Models: One with LSTM and One with BiLSTM
# =============================================================================

def build_universal_model_lstm(time_steps, n_stocks, n_sectors):
    # Inputs for metadata and sequence
    stock_input = Input(shape=(1,), name='stock_input')
    sector_input = Input(shape=(1,), name='sector_input')
    price_input = Input(shape=(time_steps, 1), name='price_input')
    
    # Embedding layers for metadata
    stock_embed = Embedding(input_dim=n_stocks, output_dim=8)(stock_input)
    sector_embed = Embedding(input_dim=n_sectors, output_dim=4)(sector_input)
    stock_embed = Reshape((8,))(stock_embed)
    sector_embed = Reshape((4,))(sector_embed)
    
    # LSTM branch
    x = LSTM(128, return_sequences=True)(price_input)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    
    # Combine LSTM output with embeddings
    combined = Concatenate()([x, stock_embed, sector_embed])
    combined = Dense(32, activation='relu')(combined)
    output = Dense(1)(combined)
    
    model = Model(inputs=[price_input, stock_input, sector_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_universal_model_bilstm(time_steps, n_stocks, n_sectors):
    # Inputs for metadata and sequence
    stock_input = Input(shape=(1,), name='stock_input')
    sector_input = Input(shape=(1,), name='sector_input')
    price_input = Input(shape=(time_steps, 1), name='price_input')
    
    # Embedding layers for metadata
    stock_embed = Embedding(input_dim=n_stocks, output_dim=8)(stock_input)
    sector_embed = Embedding(input_dim=n_sectors, output_dim=4)(sector_input)
    stock_embed = Reshape((8,))(stock_embed)
    sector_embed = Reshape((4,))(sector_embed)
    
    # Bidirectional LSTM branch
    x = Bidirectional(LSTM(128, return_sequences=True))(price_input)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)
    
    # Combine BiLSTM output with embeddings
    combined = Concatenate()([x, stock_embed, sector_embed])
    combined = Dense(32, activation='relu')(combined)
    output = Dense(1)(combined)
    
    model = Model(inputs=[price_input, stock_input, sector_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

n_stocks = df['StockID'].nunique()
n_sectors = df['SectorID'].nunique()

universal_model_lstm = build_universal_model_lstm(time_steps, n_stocks, n_sectors)
universal_model_bilstm = build_universal_model_bilstm(time_steps, n_stocks, n_sectors)

print("-----------------------------------------------------------------------------")
print("Universal LSTM Model Summary:")
universal_model_lstm.summary()

print("-----------------------------------------------------------------------------")
print("\nUniversal BiLSTM Model Summary:")
universal_model_bilstm.summary()

# =============================================================================
# Step 6: Split Data into Training and Testing Sets
# =============================================================================

split_idx = int(0.8 * len(X))
X_train = [X[:split_idx], stock_ids[:split_idx], sector_ids[:split_idx]]
y_train = y[:split_idx]
X_test = [X[split_idx:], stock_ids[split_idx:], sector_ids[split_idx:]]
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
def inverse_scale_predictions(preds, X_stock_ids, y_true):
    final_preds = []
    final_y = []
    for i, (pred, stock_id_val) in enumerate(zip(preds, X_stock_ids)):
        # Get the ticker name from the categorical mapping
        ticker_name = df['Ticker'].astype('category').cat.categories[stock_id_val]
        scaler = scalers[ticker_name]
        # Inverse transform a single prediction
        pred_inv = scaler.inverse_transform(np.array([[pred[0]]]))
        final_preds.append(pred_inv[0][0])
        
        # Also inverse transform the true value
        y_inv = scaler.inverse_transform(np.array([[y_true[i]]]))
        final_y.append(y_inv[0][0])
    return np.array(final_preds), np.array(final_y)

# For each model, do the inverse transformation
lstm_preds_final, y_test_final = inverse_scale_predictions(preds_lstm, X_test[1], y_test)
bilstm_preds_final, _ = inverse_scale_predictions(preds_bilstm, X_test[1], y_test)
ensemble_preds_final, _ = inverse_scale_predictions(ensemble_preds, X_test[1], y_test)

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



