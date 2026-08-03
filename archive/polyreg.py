from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

degree = 1
n_splits = 1000
n_jobs = -1  # Use all available cores

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\n🔎 Performing Polynomial Regression with Degree {degree}...")


# === Load scalers ===
# === Load scaled input/target arrays ===
X_all = np.load("X.npy")
y_all_scaled = np.load("y.npy")

#X_train_selected = X_all[:int(0.8*len(X_all))]
#y_train = y_all_scaled[:int(0.8*len(X_all))]

#X_val_selected = X_all[int(0.8*len(X_all)):]
#y_val = y_all_scaled[int(0.8*len(X_all)):]


X_train_selected, X_val_selected, y_train, y_val = X_all[:int(0.9*len(X_all))], X_all[int(0.9*len(X_all)):], y_all_scaled[:int(0.9*len(X_all))], y_all_scaled[int(0.9*len(X_all)):]

def train_and_score(train_idx, test_idx):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_val_selected)

    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / np.maximum(np.abs(y_val), 1e-8))) * 100
    return r2, mae, mse, mape

# Run CV folds in parallel
scores = Parallel(n_jobs=n_jobs)(
    delayed(train_and_score)(train_idx, test_idx)
    for train_idx, test_idx in tqdm(kf.split(X_train_selected), total=n_splits, desc=f"Degree {degree}", leave=False)
)

# Aggregate metrics
r2s, maes, mses, mapes = zip(*scores)
avg_r2, avg_mae, avg_mse, avg_mape = np.mean(r2s), np.mean(maes), np.mean(mses), np.mean(mapes)

print(f"\n📈 Polynomial Regression (Degree {degree}) Results:")
print(f"   - R²   = {avg_r2:.4f}")
print(f"   - MAE  = {avg_mae:.4f}")
print(f"   - MSE  = {avg_mse:.4f}")
print(f"   - MAPE = {avg_mape:.2f}%")

# Train final model
final_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
final_model.fit(X_train_selected, y_train)
y_pred_poly = final_model.predict(X_val_selected)

# Plot predicted vs true values
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_poly, alpha=0.5, edgecolor='k', label='Polynomial Degree 3 Predictions')
lims = [min(y_val.min(), y_pred_poly.min()), max(y_val.max(), y_pred_poly.max())]
plt.plot(lims, lims, 'r--', label='Perfect Prediction (y = x)')
plt.xlabel('True Dose Rates')
plt.ylabel('Predicted Dose Rates')
plt.title('Polynomial Regression (Degree 3): True vs Predicted Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
