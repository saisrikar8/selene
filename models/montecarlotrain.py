import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
import numpy as np
import pickle, os
import matplotlib.pyplot as plt

# === MC Dropout layer: active at train & inference ===
class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

# === Constants ===
PROCESSED_CSV_FILE = "../aligned_averaged_by_crater.csv"
FEATURE_PREFIXES = [
    "sis_cnt_O_1", "sis_cnt_O_2", "sis_cnt_C_2",
    "ace_cnt_Fe_4", "ace_cnt_Ni_2", "ace_cnt_Cr_4",
    "ace_cnt_S_3", "ace_cnt_Ti_4", "ace_cnt_Fe_7",
    "ace_cnt_Ni_1", "ace_cnt_F_3", "ace_cnt_Ca_5",
    "ace_cnt_S_4", "ace_cnt_Fe_5", "ace_cnt_Ar_5",
    "ace_cnt_Ne_2", "ace_cnt_Si_6", "ace_cnt_B_5"
]
MODEL_FILE = "huber_lunar_radiation_model_heteroscedastic_mc_dropout.keras"

# === Load or preprocess data ===
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

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

np.save("../X_val.npy", X_val)
np.save("../y_val.npy", y_val)

# === Define heteroscedastic loss ===
def heteroscedastic_loss(y_true, y_pred):
    mean = y_pred[:, 0]
    log_var = y_pred[:, 1]
    log_var = tf.clip_by_value(log_var, -10.0, 10.0)  # clamp to avoid numeric issues
    precision = tf.exp(-log_var) + 1e-6               # add epsilon for stability
    sq_error = tf.square(y_true[:, 0] - mean)
    loss = 0.5 * (precision * sq_error + log_var)
    return tf.reduce_mean(loss)

# === Build model with MC Dropout ===
inputs = Input(shape=(X.shape[1],))
x = Dense(64, kernel_regularizer=l2(1e-1))(inputs)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.01)(x)
x = MCDropout(0.4)(x)

x = Dense(32, kernel_regularizer=l2(1e-1))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.01)(x)
x = MCDropout(0.3)(x)

outputs = Dense(2)(x)  # mean and log variance
model = Model(inputs=inputs, outputs=outputs)

lr = float(input("Learning rate: "))
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=heteroscedastic_loss,
              metrics=[])

cb = []
if input("Use callbacks? (y/n): ").lower() == "y":
    cb = [
        EarlyStopping(patience=25, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-10, verbose=1)
    ]

# === Train ===
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=cb
)

model.save(os.path.join("..", MODEL_FILE))
model.summary()

# === Plot training loss ===
hist = history.history
plt.figure(figsize=(8,6))
plt.plot(hist["loss"], label="train loss")
plt.plot(hist.get("val_loss", []), label="val loss")
plt.legend()
plt.title("Heteroscedastic Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
