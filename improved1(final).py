# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:49:19 2026

@author: fatemeh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')

TRAIN_START = "2024-08-01"
TRAIN_END   = "2024-08-08"
TEST_START  = "2024-08-08"
TEST_END    = "2024-08-15"
SYMBOL      = "BTC-USD"
Z_THRESHOLD = 2.5
SAVE_DIR = r"C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\anomaly_in_markets"
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_data(start, end, symbol=SYMBOL, interval="1h"):
    df = yf.download(symbol, start=start, end=end, interval=interval)
    df = df[['Close']].copy()
    df.columns = ['close']
    df.index = pd.to_datetime(df.index)
    return df

train_csv = os.path.join(SAVE_DIR, "btc_train_aug1_7.csv")
test_csv  = os.path.join(SAVE_DIR, "btc_test_aug8_14.csv")

if os.path.exists(train_csv):
    train_df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
else:
    train_df = fetch_data(TRAIN_START, TRAIN_END)
    train_df.to_csv(train_csv)

if os.path.exists(test_csv):
    test_df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
else:
    test_df = fetch_data(TEST_START, TEST_END)
    test_df.to_csv(test_csv)

print(f"Training data: {train_df.shape[0]} hours")
print(f"Test data:     {test_df.shape[0]} hours")

# ------------------------------ ANOMALY DETECTION ------------------------------ #
mu = train_df['close'].mean()
sigma = train_df['close'].std()
train_df['z_score'] = (train_df['close'] - mu) / sigma
anomalies = train_df[np.abs(train_df['z_score']) > Z_THRESHOLD]
print(f"Anomalies detected (|Z| > {Z_THRESHOLD}): {len(anomalies)}")
if len(anomalies) > 0:
    print(anomalies[['close', 'z_score']])
else:
    print("No anomaly at 2.5 – lowering threshold to 2.0 to capture the crash.")
    Z_THRESHOLD = 2.0
    anomalies = train_df[np.abs(train_df['z_score']) > Z_THRESHOLD]
    print(f"Now anomalies detected: {len(anomalies)}")
    print(anomalies[['close', 'z_score']])

train_clean = train_df[np.abs(train_df['z_score']) <= Z_THRESHOLD].copy()

plt.figure(figsize=(10,4))
plt.plot(train_df.index, train_df['close'], label='Training Close', color='grey', alpha=0.6)
plt.scatter(anomalies.index, anomalies['close'], color='red', s=50, label=f'Anomalies (Z>{Z_THRESHOLD})')
plt.title('Training Data (Aug 1‑7, 2024) – Anomalies Highlighted')
plt.xlabel('Date')
plt.ylabel('BTC Close Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_anomalies.png"), dpi=150)
plt.close()

def add_features(df):
    df = df.copy()
    # Lag features: price 1h, 2h, 3h, 6h, 12h, 24h ago
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'lag_{lag}'] = df['close'].shift(lag)
    # Moving averages
    df['ma_6'] = df['close'].rolling(6).mean()
    df['ma_12'] = df['close'].rolling(12).mean()
    # Hour of day
    df['hour_of_day'] = df.index.hour
    # Drop rows with NaN (due to shifts/rolling)
    df.dropna(inplace=True)
    return df

train_dirty_feat = add_features(train_df)
train_clean_feat = add_features(train_clean)
test_feat = add_features(test_df)

# Target
y_train_dirty = train_dirty_feat['close']
y_train_clean = train_clean_feat['close']
y_test = test_feat['close']

# Feature columns (exclude close and z_score if present)
feature_cols = [col for col in train_dirty_feat.columns if col not in ['close', 'z_score']]
X_train_dirty = train_dirty_feat[feature_cols]
X_train_clean = train_clean_feat[feature_cols]
X_test = test_feat[feature_cols]

print(f"Training features (dirty): {X_train_dirty.shape}")
print(f"Training features (clean): {X_train_clean.shape}")
print(f"Test features: {X_test.shape}")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_dirty_scaled = scaler_X.fit_transform(X_train_dirty)
y_train_dirty_scaled = scaler_y.fit_transform(y_train_dirty.values.reshape(-1,1)).ravel()
X_train_clean_scaled = scaler_X.transform(X_train_clean)
y_train_clean_scaled = scaler_y.transform(y_train_clean.values.reshape(-1,1)).ravel()
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).ravel()

results = {}

def train_evaluate(name, model, X_train, y_train, X_test, y_test, is_scaled=False):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if is_scaled:
        preds_orig = scaler_y.inverse_transform(preds.reshape(-1,1)).ravel()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
    else:
        preds_orig = preds
        y_test_orig = y_test

    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mae = mean_absolute_error(y_test_orig, preds_orig)
    mape = np.mean(np.abs((y_test_orig - preds_orig) / y_test_orig)) * 100
    r2 = r2_score(y_test_orig, preds_orig)
    return rmse, mae, mape, r2, preds_orig

# ---- 1. Linear Regression ----
print("\n===== Linear Regression =====")
rmse_d_lr, mae_d_lr, mape_d_lr, r2_d_lr, pred_d_lr = train_evaluate(
    "LR dirty", LinearRegression(), X_train_dirty, y_train_dirty, X_test, y_test
)
rmse_c_lr, mae_c_lr, mape_c_lr, r2_c_lr, pred_c_lr = train_evaluate(
    "LR clean", LinearRegression(), X_train_clean, y_train_clean, X_test, y_test
)
improve_rmse_lr = (rmse_d_lr - rmse_c_lr) / rmse_d_lr * 100
improve_mae_lr = (mae_d_lr - mae_c_lr) / mae_d_lr * 100
results['Linear Regression'] = {
    'dirty': {'RMSE': rmse_d_lr, 'MAE': mae_d_lr, 'MAPE': mape_d_lr, 'R2': r2_d_lr},
    'clean': {'RMSE': rmse_c_lr, 'MAE': mae_c_lr, 'MAPE': mape_c_lr, 'R2': r2_c_lr},
    'improve_RMSE': improve_rmse_lr,
    'improve_MAE': improve_mae_lr,
    'pred_dirty': pred_d_lr,
    'pred_clean': pred_c_lr
}

# ---- 2. XGBoost ----
print("\n===== XGBoost =====")
xgb_params = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
rmse_d_xgb, mae_d_xgb, mape_d_xgb, r2_d_xgb, pred_d_xgb = train_evaluate(
    "XGB dirty", xgb.XGBRegressor(**xgb_params),
    X_train_dirty, y_train_dirty, X_test, y_test
)
rmse_c_xgb, mae_c_xgb, mape_c_xgb, r2_c_xgb, pred_c_xgb = train_evaluate(
    "XGB clean", xgb.XGBRegressor(**xgb_params),
    X_train_clean, y_train_clean, X_test, y_test
)
improve_rmse_xgb = (rmse_d_xgb - rmse_c_xgb) / rmse_d_xgb * 100
improve_mae_xgb = (mae_d_xgb - mae_c_xgb) / mae_d_xgb * 100
results['XGBoost'] = {
    'dirty': {'RMSE': rmse_d_xgb, 'MAE': mae_d_xgb, 'MAPE': mape_d_xgb, 'R2': r2_d_xgb},
    'clean': {'RMSE': rmse_c_xgb, 'MAE': mae_c_xgb, 'MAPE': mape_c_xgb, 'R2': r2_c_xgb},
    'improve_RMSE': improve_rmse_xgb,
    'improve_MAE': improve_mae_xgb,
    'pred_dirty': pred_d_xgb,
    'pred_clean': pred_c_xgb
}

# ---- 3. MLPRegressor ----
print("\n===== MLP Neural Network =====")
mlp_params = dict(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=1000, random_state=42)
rmse_d_mlp, mae_d_mlp, mape_d_mlp, r2_d_mlp, pred_d_mlp = train_evaluate(
    "MLP dirty", MLPRegressor(**mlp_params),
    X_train_dirty_scaled, y_train_dirty_scaled, X_test_scaled, y_test_scaled, is_scaled=True
)
rmse_c_mlp, mae_c_mlp, mape_c_mlp, r2_c_mlp, pred_c_mlp = train_evaluate(
    "MLP clean", MLPRegressor(**mlp_params),
    X_train_clean_scaled, y_train_clean_scaled, X_test_scaled, y_test_scaled, is_scaled=True
)
improve_rmse_mlp = (rmse_d_mlp - rmse_c_mlp) / rmse_d_mlp * 100
improve_mae_mlp = (mae_d_mlp - mae_c_mlp) / mae_d_mlp * 100
results['MLP Neural Net'] = {
    'dirty': {'RMSE': rmse_d_mlp, 'MAE': mae_d_mlp, 'MAPE': mape_d_mlp, 'R2': r2_d_mlp},
    'clean': {'RMSE': rmse_c_mlp, 'MAE': mae_c_mlp, 'MAPE': mape_c_mlp, 'R2': r2_c_mlp},
    'improve_RMSE': improve_rmse_mlp,
    'improve_MAE': improve_mae_mlp,
    'pred_dirty': pred_d_mlp,
    'pred_clean': pred_c_mlp
}

# ------------------------------ SUMMARY TABLE ------------------------------ #
print("\n" + "="*120)
print(f"{'Model':<22} {'Dirty RMSE':>10} {'Clean RMSE':>10} {'RMSE Impr':>8}  {'Dirty MAE':>10} {'Clean MAE':>10} {'MAE Impr':>8}  {'MAPE Dirty':>9} {'MAPE Clean':>9}  {'R2 Dirty':>7} {'R2 Clean':>7}")
print("="*120)
for model, res in results.items():
    d = res['dirty']
    c = res['clean']
    imp_rmse = res['improve_RMSE']
    imp_mae = res['improve_MAE']
    print(f"{model:<22} ${d['RMSE']:>9,.0f} ${c['RMSE']:>9,.0f} {imp_rmse:>7.1f}%  ${d['MAE']:>9,.0f} ${c['MAE']:>9,.0f} {imp_mae:>7.1f}%  {d['MAPE']:>8.2f}% {c['MAPE']:>8.2f}%  {d['R2']:>6.3f}  {c['R2']:>6.3f}")
print("="*120)

# ------------------------------ PLOTS ------------------------------ #
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
test_times = test_feat.index

# Linear Regression
axes[0,0].plot(test_times, y_test, color='green', label='Actual')
axes[0,0].plot(test_times, results['Linear Regression']['pred_dirty'], '--', color='red', label='Dirty')
axes[0,0].set_title(f"Linear Regression (Dirty)\nRMSE=${rmse_d_lr:,.0f}, MAE=${mae_d_lr:,.0f}, R²={r2_d_lr:.3f}")
axes[0,0].legend()
axes[0,1].plot(test_times, y_test, color='green', label='Actual')
axes[0,1].plot(test_times, results['Linear Regression']['pred_clean'], '--', color='blue', label='Clean')
axes[0,1].set_title(f"Linear Regression (Clean)\nRMSE=${rmse_c_lr:,.0f}, MAE=${mae_c_lr:,.0f}, R²={r2_c_lr:.3f}")
axes[0,1].legend()

# XGBoost
axes[1,0].plot(test_times, y_test, color='green', label='Actual')
axes[1,0].plot(test_times, results['XGBoost']['pred_dirty'], '--', color='red', label='Dirty')
axes[1,0].set_title(f"XGBoost (Dirty)\nRMSE=${rmse_d_xgb:,.0f}, MAE=${mae_d_xgb:,.0f}, R²={r2_d_xgb:.3f}")
axes[1,0].legend()
axes[1,1].plot(test_times, y_test, color='green', label='Actual')
axes[1,1].plot(test_times, results['XGBoost']['pred_clean'], '--', color='blue', label='Clean')
axes[1,1].set_title(f"XGBoost (Clean)\nRMSE=${rmse_c_xgb:,.0f}, MAE=${mae_c_xgb:,.0f}, R²={r2_c_xgb:.3f}")
axes[1,1].legend()

# MLP
axes[2,0].plot(test_times, y_test, color='green', label='Actual')
axes[2,0].plot(test_times, results['MLP Neural Net']['pred_dirty'], '--', color='red', label='Dirty')
axes[2,0].set_title(f"MLP Neural Net (Dirty)\nRMSE=${rmse_d_mlp:,.0f}, MAE=${mae_d_mlp:,.0f}, R²={r2_d_mlp:.3f}")
axes[2,0].legend()
axes[2,1].plot(test_times, y_test, color='green', label='Actual')
axes[2,1].plot(test_times, results['MLP Neural Net']['pred_clean'], '--', color='blue', label='Clean')
axes[2,1].set_title(f"MLP Neural Net (Clean)\nRMSE=${rmse_c_mlp:,.0f}, MAE=${mae_c_mlp:,.0f}, R²={r2_c_mlp:.3f}")
axes[2,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_comparison.png"), dpi=150)
plt.show()
print("Charts saved in", SAVE_DIR)