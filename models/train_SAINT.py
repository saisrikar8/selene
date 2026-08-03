import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle, os
import matplotlib.pyplot as plt

# Constants
PROCESSED_CSV_FILE = "../aligned_averaged_by_crater.csv"
FEATURE_PREFIXES = [
    "sis_cnt_O_1", "sis_cnt_O_2", "sis_cnt_C_2",
    "ace_cnt_Fe_4", "ace_cnt_Ni_2", "ace_cnt_Cr_4",
    "ace_cnt_S_3", "ace_cnt_Ti_4", "ace_cnt_Fe_7",
    "ace_cnt_Ni_1", "ace_cnt_F_3", "ace_cnt_Ca_5",
]
MODEL_FILE = "SAINT_model.keras"
BEST_WEIGHTS_FILE = "../best_saint.weights.h5"

# Load or process data
if os.path.exists("../X.npy") and os.path.exists("../y.npy") and os.path.exists("../scaler_X.pkl"):
    X = np.load("../X.npy")
    y = np.load("../y.npy")
    with open("../scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
else:
    df = pd.read_csv(PROCESSED_CSV_FILE)
    df = df.dropna(subset=["dose_D1"])
    missing = [c for c in FEATURE_PREFIXES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    columns_to_check = FEATURE_PREFIXES + ["dose_D1"]
    df = df[(df[columns_to_check] >= 0).all(axis=1)]
    y = df["dose_D1"].to_numpy().reshape(-1, 1)
    X = df[FEATURE_PREFIXES].astype(float).to_numpy()
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    np.save("../X.npy", X)
    np.save("../y.npy", y)
    with open("../scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# --- SAINT Transformer model definition ---
class SAINTBlock(tf.keras.Model):
    def __init__(self, feature_dim, num_heads, dropout=0.1):
        super().__init__()
        self.col_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)
        self.col_ffn = tf.keras.Sequential([
            layers.Dense(feature_dim * 4, activation='relu'),
            layers.Dense(feature_dim),
        ])
        self.col_norm1 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)
        self.row_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)
        self.row_ffn = tf.keras.Sequential([
            layers.Dense(feature_dim * 4, activation='relu'),
            layers.Dense(feature_dim),
        ])
        self.row_norm1 = layers.LayerNormalization()
        self.row_norm2 = layers.LayerNormalization()

    def call(self, x, training=False):
        x_col = tf.transpose(x, perm=[0, 2, 1])
        attn_out = self.col_attn(x_col, x_col, training=training)
        x_col = self.col_norm1(x_col + self.dropout(attn_out, training=training))
        x_col_for_ffn = tf.transpose(x_col, perm=[0, 2, 1])
        ffn_out = self.col_ffn(x_col_for_ffn, training=training)
        x_col = tf.transpose(x_col_for_ffn + self.dropout(ffn_out, training=training), perm=[0, 2, 1])
        x = tf.transpose(x_col, perm=[0, 2, 1])
        x_row = tf.transpose(x, perm=[1, 0, 2])
        attn_out = self.row_attn(x_row, x_row, training=training)
        x_row = self.row_norm1(x_row + self.dropout(attn_out, training=training))
        ffn_out = self.row_ffn(x_row, training=training)
        x_row = self.row_norm2(x_row + self.dropout(ffn_out, training=training))
        return tf.transpose(x_row, perm=[1, 0, 2])

class SAINTModel(tf.keras.Model):
    def __init__(self, num_features, feature_dim=32, num_heads=4, depth=4, dropout=0.1):
        super().__init__()
        self.input_proj = layers.Dense(feature_dim)
        self.blocks = [SAINTBlock(feature_dim, num_heads, dropout) for _ in range(depth)]
        self.norm = layers.LayerNormalization()
        self.head = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(1)
        ])

    def call(self, x, training=False):
        x = self.input_proj(x[..., tf.newaxis])
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        return self.head(x)

# Build model
model = SAINTModel(num_features=X.shape[1], feature_dim=32, num_heads=4, depth=4, dropout=0.1)

# --- Training setup ---
lr = float(input("Learning rate: "))
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss='mae',
    metrics=["mae", 'mape', 'mse']
)

use_callbacks = input("Use callbacks? (y/n): ").strip().lower() == "y"
cb = []
if use_callbacks:
    cb = [
        EarlyStopping(patience=25, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-10, verbose=1),
        ModelCheckpoint(
            BEST_WEIGHTS_FILE,
            monitor="val_mae",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

# --- Train ---
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=cb
)

# --- Optional: Load best weights if they exist ---
if os.path.exists(BEST_WEIGHTS_FILE):
    print("Loading best weights...")
    model.load_weights(BEST_WEIGHTS_FILE)

# Save final model
model.save(os.path.join("..", MODEL_FILE))
model.summary()

# --- Plot training curves ---
hist = history.history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist["loss"], label="train")
plt.plot(hist.get("val_loss", []), label="val")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(hist["mae"], label="train")
plt.plot(hist.get("val_mae", []), label="val")
plt.title("MAE")
plt.legend()
plt.show()

# --- Plot distribution of target values ---
plt.hist(y_train, bins=50, alpha=0.7, label="Train")
plt.hist(y_val, bins=50, alpha=0.7, label="Validation")
plt.xlabel("Dose (cGy/day)")
plt.ylabel("Frequency")
plt.legend()
plt.title("Target Value Distribution")
plt.show()
