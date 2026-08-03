import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.saving import register_keras_serializable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pickle
from tensorflow.keras import layers

@register_keras_serializable()
class SAINTBlock(tf.keras.layers.Layer):
    def __init__(self, feature_dim=32, num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.col_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)
        self.col_ffn = tf.keras.Sequential([
            layers.Dense(feature_dim * 4, activation='relu'),
            layers.Dense(feature_dim),
        ])
        self.col_norm1 = layers.LayerNormalization()
        self.col_norm2 = layers.LayerNormalization()
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class SAINTModel(tf.keras.Model):
    def __init__(self, num_features=18, feature_dim=32, num_heads=4, depth=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.depth = depth
        self.dropout_rate = dropout

        self.input_proj = layers.Dense(feature_dim)
        self.blocks = [
            SAINTBlock(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ]
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "feature_dim": self.feature_dim,
            "num_heads": self.num_heads,
            "depth": self.depth,
            "dropout": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# Load data and scalers
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Load model architecture
model = SAINTModel(num_features=X_val.shape[1], feature_dim=32, num_heads=4, depth=4, dropout=0.1)

# Build the model (important)
model.build(input_shape=(None, X_val.shape[1]))

# Load weights from .h5 file
model.load_weights("./SAINT_model/model.weights.h5")  # <-- change this to your weights file path

# Predict
y_pred = model.predict(X_val)

# Metrics (same as before)
mae = mean_absolute_error(y_val, y_pred)
mape = mean_absolute_percentage_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Save results, plot, etc... (no change)
df = pd.DataFrame({
    "true": y_val.flatten(),
    "pred": y_pred.flatten(),
    "error": (y_pred - y_val).flatten(),
    "abs_error": np.abs(y_pred - y_val).flatten()
})
df["is_outlier"] = df["abs_error"] > (2 * mae)
df.to_csv("saint_predictions.csv", index=False)
print(f"\nSaved results to saint_predictions.csv")

plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred, alpha=0.5, label="Predictions")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label="Ideal")
plt.xlabel("True Dose (cGy/day)")
plt.ylabel("Predicted Dose (cGy/day)")
plt.title("Predicted vs True")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["abs_error"], bins=40, alpha=0.7)
plt.axvline(mae, color='red', linestyle='--', label=f"MAE = {mae:.3f}")
plt.xlabel("Absolute Error (cGy/day)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.legend()
plt.tight_layout()
plt.show()

print("\nTop 10 outliers by absolute error:")
print(df.sort_values("abs_error", ascending=False).head(10))

