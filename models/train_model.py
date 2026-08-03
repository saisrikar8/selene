import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, ReLU, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish
import numpy as np
import pickle, os
import matplotlib.pyplot as plt

import losses
import activations

# Constants
PROCESSED_CSV_FILE = "../aligned_averaged_by_crater.csv"
# Based on your top correlators
FEATURE_PREFIXES = [
    "sis_cnt_O_1", "sis_cnt_O_2", "sis_cnt_C_2",
    "ace_cnt_Fe_4", "ace_cnt_Ni_2", "ace_cnt_Cr_4",
    "ace_cnt_S_3", "ace_cnt_Ti_4", "ace_cnt_Fe_7",
    "ace_cnt_Ni_1", "ace_cnt_F_3", "ace_cnt_Ca_5",
    "ace_cnt_S_4", "ace_cnt_Fe_5", "ace_cnt_Ar_5",
    "ace_cnt_Ne_2", "ace_cnt_Si_6", "ace_cnt_B_5"
]

MODEL_FILE = "huber_lunar_radiation_model.keras"

# Load or process data
if os.path.exists("../X.npy") and os.path.exists("../y.npy") and os.path.exists("../scaler_X.pkl"):
    X = np.load("../X.npy")
    y = np.load("../y.npy")
    with open("../scaler_X.pkl", "rb") as f: scaler_X = pickle.load(f)
else:
    # Load the CSV
    df = pd.read_csv(PROCESSED_CSV_FILE)

    # Drop rows with missing target
    df = df.dropna(subset=["dose_D1"])

    # Ensure all required feature columns are present
    missing = [c for c in FEATURE_PREFIXES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Drop rows with negative values in any feature or in dose_D1
    columns_to_check = FEATURE_PREFIXES + ["dose_D1"]
    df = df[(df[columns_to_check] >= 0).all(axis=1)]

    # Extract and scale target
    y = df["dose_D1"].to_numpy().reshape(-1, 1)

    # Extract and scale features
    X = df[FEATURE_PREFIXES].astype(float).to_numpy()
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    # Save outputs
    np.save("../X.npy", X)
    np.save("../y.npy", y)
    with open("../scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

print(f"X shape: {X.shape}, y shape: {y.shape}")

X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

# Split
split_idx = int(len(X_shuffled) * 0.9)
X_train, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
y_train, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]

np.save("../X_val.npy", X_val)
np.save("../y_val.npy", y_val)

# Build model

model = Sequential([
    Dense(64, kernel_regularizer=l2(1e-2), input_dim=X.shape[1]),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    Dropout(0.4),

    Dense(32, kernel_regularizer=l2(1e-2)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    Dropout(0.3),

    Dense(1)
])


lr = float(input("Learning rate: "))
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss= 'mae',
              metrics=["mae", 'mape', 'mse'])

cb = []
if input("Use callbacks? (y/n): ").lower()=="y":
    cb = [
        EarlyStopping(patience=25, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-10, verbose=1)
    ]

# Train
history = model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_val, y_val), callbacks=cb)

model.save(os.path.join("..", MODEL_FILE))
model.summary()

# Plot
hist = history.history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist["loss"], label="train")
plt.plot(hist.get("val_loss",[]), label="val")
plt.legend()
plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(hist["mae"], label="train")
plt.plot(hist.get("val_mae",[]), label="val")
plt.legend()
plt.title("MAE")
plt.show()
