from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from tqdm import tqdm

# Suppose X_train_selected, y_train, X_val_selected, y_val are prepared


# === Load scalers ===
# === Load scaled input/target arrays ===
X_all = np.load("../X.npy")
y_all_scaled = np.load("../y.npy")

#X_train_selected = X_all[:int(0.8*len(X_all))]
#y_train = y_all_scaled[:int(0.8*len(X_all))]

#X_val_selected = X_all[int(0.8*len(X_all)):]
#y_val = y_all_scaled[int(0.8*len(X_all)):]


X_train_selected, X_val_selected, y_train, y_val = X_all[:int(0.9*len(X_all))], X_all[int(0.9*len(X_all)):], y_all_scaled[:int(0.9*len(X_all))], y_all_scaled[int(0.9*len(X_all)):]


# Ridge Regression example
ridge = Ridge(alpha=1e-2)
ridge.fit(X_train_selected, y_train)
y_pred_lr = ridge.predict(X_val_selected)

mse = mean_squared_error(y_val, y_pred_lr)
r2 = r2_score(y_val, y_pred_lr)
mean_abs_error = np.mean(np.abs(y_val - y_pred_lr))
mean_abs_percentage_error = (mean_abs_error / np.mean(y_val))*100
print(f"Linear Regression MSE: {mse:.6f}, R2: {r2:.4f}")
print(f"Linear Regression MAE: {mean_abs_error:.6f}, MAPE: {mean_abs_percentage_error:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred_lr, alpha=0.4, s=20, label='Linear Regression Predictions')
lims = [min(y_val.min(), y_pred_lr.min()), max(y_val.max(), y_pred_lr.max())]
plt.plot(lims, lims, 'r--', label='Perfect Prediction (y = x)')
plt.xlabel('True Dose Rates')
plt.ylabel('Predicted Dose Rates')
plt.title('Linear Regression: True vs Predicted Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1)
svr_rbf.fit(X_train_selected, y_train)

y_pred = svr_rbf.predict(X_val_selected)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
mean_abs_error = np.mean(np.abs(y_val - y_pred))
mean_abs_percentage_error = (mean_abs_error / np.mean(y_val))*100
print(f"SVR MSE: {mse:.6f}, R2: {r2:.4f}")
print(f"SVR MAE: {mean_abs_error:.6f}, MAPE: {mean_abs_percentage_error:.4f}")

# Plot predicted vs true
plt.scatter(y_val, y_pred, color='teal', edgecolor='k', alpha=0.7)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)  # Ideal diagonal
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Predicted vs. True')
plt.grid(True)
plt.show()

max_degree = 5
n_splits = 1000
n_jobs = -1  # Use all available cores

best_degree = None
best_r2 = -np.inf
best_model = None
metrics_by_degree = {}

print("🚀 Polynomial Regression with Parallel CV and Metrics...\n")

for degree in range(1, max_degree + 1):
    print(f"\n🔎 Degree {degree} CV Progress:")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def train_and_score(train_idx, test_idx):

        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_val_selected)

        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / np.maximum(np.abs(y_val), 1e-8))) * 100  # Avoid div/0
        return r2, mae, mse, mape

    # Run CV folds in parallel
    scores = Parallel(n_jobs=n_jobs)(
        delayed(train_and_score)(train_idx, test_idx)
        for train_idx, test_idx in tqdm(kf.split(X_train_selected), total=n_splits, desc=f"Degree {degree}", leave=False)
    )

    # Aggregate metrics
    r2s, maes, mses, mapes = zip(*scores)
    avg_r2, avg_mae, avg_mse, avg_mape = np.mean(r2s), np.mean(maes), np.mean(mses), np.mean(mapes)

    print(f"📈 Degree {degree} Results:")
    print(f"   - R²   = {avg_r2:.4f}")
    print(f"   - MAE  = {avg_mae:.4f}")
    print(f"   - MSE  = {avg_mse:.4f}")
    print(f"   - MAPE = {avg_mape:.2f}%")

    metrics_by_degree[degree] = (avg_r2, avg_mae, avg_mse, avg_mape)

    if avg_r2 > best_r2:
        best_r2 = avg_r2
        best_degree = degree
        best_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        best_model.fit(X_train_selected, y_train)

# Final model summary
print(f"\n✅ Best Degree: {best_degree}")
print(f"   - R²   = {metrics_by_degree[best_degree][0]:.4f}")
print(f"   - MAE  = {metrics_by_degree[best_degree][1]:.4f}")
print(f"   - MSE  = {metrics_by_degree[best_degree][2]:.4f}")
print(f"   - MAPE = {metrics_by_degree[best_degree][3]:.2f}%")