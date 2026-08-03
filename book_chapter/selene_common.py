"""Reproducible, non-interactive rebuild of the published SELENE ANN.

Refactored from ``train_model.py`` so the book-chapter experiments can import a
single source of truth for data loading, the model architecture, training, and
metrics. Deterministic (fixed seeds) so reported numbers are stable.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)
tf.random.set_seed(42)

CSV = os.path.join(os.path.dirname(__file__), "..", "aligned_averaged_by_crater.csv")

# The 18 top-correlated ACE SIS/CRIS features used by the published SELENE model
# (identical to train_model.py).
FEATURES = [
    "sis_cnt_O_1", "sis_cnt_O_2", "sis_cnt_C_2",
    "ace_cnt_Fe_4", "ace_cnt_Ni_2", "ace_cnt_Cr_4",
    "ace_cnt_S_3", "ace_cnt_Ti_4", "ace_cnt_Fe_7",
    "ace_cnt_Ni_1", "ace_cnt_F_3", "ace_cnt_Ca_5",
    "ace_cnt_S_4", "ace_cnt_Fe_5", "ace_cnt_Ar_5",
    "ace_cnt_Ne_2", "ace_cnt_Si_6", "ace_cnt_B_5",
]


def load_data():
    """Return (X_scaled, y, scaler_X) from the daily-aggregated dataset.

    Drops rows with missing dose target and rows with any negative value in a
    feature or the target, matching the published preprocessing.
    """
    df = pd.read_csv(CSV).dropna(subset=["dose_D1"])
    cols = FEATURES + ["dose_D1"]
    df = df[(df[cols] >= 0).all(axis=1)]
    y = df["dose_D1"].to_numpy().reshape(-1, 1)
    X = df[FEATURES].astype(float).to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


def split(X, y):
    """Deterministic 90/10 shuffle split (random_state=42)."""
    Xs, ys = shuffle(X, y, random_state=42)
    i = int(len(Xs) * 0.9)
    return Xs[:i], Xs[i:], ys[:i], ys[i:]


def build_selene(input_dim, dropout1=0.4, dropout2=0.3):
    """The published SELENE architecture: 64 -> 32 -> 1 with BN + LeakyReLU."""
    return Sequential([
        Dense(64, kernel_regularizer=l2(1e-2), input_dim=input_dim),
        BatchNormalization(), LeakyReLU(alpha=0.01), Dropout(dropout1),
        Dense(32, kernel_regularizer=l2(1e-2)),
        BatchNormalization(), LeakyReLU(alpha=0.01), Dropout(dropout2),
        Dense(1),
    ])


def train_selene(X_train, y_train, X_val, y_val, lr=1e-3, epochs=300):
    """Train SELENE with early stopping + LR reduction. Returns (model, history)."""
    m = build_selene(X_train.shape[1])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae",
              metrics=["mae", "mape", "mse"])
    cb = [EarlyStopping(patience=25, restore_best_weights=True),
          ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-10, verbose=0)]
    h = m.fit(X_train, y_train, epochs=epochs, batch_size=16,
              validation_data=(X_val, y_val), callbacks=cb, verbose=0)
    return m, h.history


def metrics(y_true, y_pred):
    """Return dict with mse, mae, r2, mape (percent)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) /
                       np.clip(np.abs(y_true), 1e-12, None))) * 100),
    }
