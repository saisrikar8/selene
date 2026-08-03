import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

X = np.load("../X.npy")
y = np.load("../y.npy")

eval_results = {}

X_shuffled, y_shuffled = X, y

split_idx = int(len(X_shuffled) * 0.9)
X_train, X_val = X_shuffled[:split_idx], np.load("../X_val.npy")
y_train, y_val = y_shuffled[:split_idx], np.load("../y_val.npy")
eval_result = {}

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    loss='mae'
)

# Dictionary to store evaluation results
eval_result = {}

# Train the model and store evaluation history
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# After fitting, access evaluation results from the Booster
evals_result = xgb_model.evals_result()
print(xgb_model.evals_result())

y_pred = xgb_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / (np.abs(y_val))))

print(eval_result)

print(f"Mean Squared Error (MSE): {mse:.8f}")
print(f"Mean Absolute Error (MAE): {mae:.8f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
print(np.mean(y_val))
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.5, edgecolors='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel("True Dose Rate")
plt.ylabel("Predicted Dose Rate")
plt.title("XGBoost Predictions vs True Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("../xgb_vs_true.png")
plt.show()

rmse = evals_result['validation_0']['rmse']
epochs = range(1, len(rmse) + 1)

# Plotting
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, rmse, label="Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Loss (RMSE) vs Epochs")
plt.grid(True)

plt.tight_layout()
plt.savefig("../xgb_training_metrics.png")
plt.show()
