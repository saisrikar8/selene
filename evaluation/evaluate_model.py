import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from losses import quantile_loss
from activations import Mish
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import joblib


def percentile_of_value(array, value):
    return np.sum(array < value) / len(array) * 100

def mc_dropout_predict(model, X_input, T=1000000):
    """
    Perform T stochastic forward passes with dropout active (training=True).
    Returns mean prediction and variance (uncertainty).
    """
    preds = []
    for _ in range(T):
        pred = model(X_input, training=True).numpy()  # dropout active
        preds.append(pred)
    preds = np.array(preds)  # shape (T, batch_size, output_dim)
    pred_mean = preds.mean(axis=0)
    pred_var = preds.var(axis=0)
    return pred_mean, pred_var


# === Load trained model ===
model = load_model("../huber_lunar_radiation_model.keras")
print("✅ Model loaded.")

# === Load scalers ===
# === Load scaled input/target arrays ===
X_all = np.load("../X_val.npy")
y_all_scaled = np.load("../y_val.npy")

# === Split last 10% as test set ===
split_idx = int(0*len(X_all))
X_test_scaled = X_all[split_idx:]
y_test_scaled = y_all_scaled[split_idx:]

# === Predict ===
print("🔍 Predicting on test data with MC Dropout (T=100)...")
T = 100
y_pred_scaled, y_pred_var = mc_dropout_predict(model, X_test_scaled, T)

print(f'Mean variance: {np.mean(y_pred_var)}')

# === Inverse transform to original scale ===
# scaler_y expects 2D input, so reshape if needed
y_test_scaled_2d = y_test_scaled.reshape(-1, 1)
y_pred_scaled_2d = y_pred_scaled.reshape(-1, 1)

y_test =  y_test_scaled_2d.flatten()
y_pred = y_pred_scaled_2d.flatten()

# === Compute evaluation metrics ===
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"  🔹 Mean Squared Error (MSE): {mse:.8f}")
print(f"  🔹 Mean Absolute Error (MAE): {mae:.8f}")
print(f"  🔹 R² Score: {r2:.4f}")

print("Target mean:", y_test.mean())
print("Target std:", y_test.std())

# === Identify outliers (top 5% largest errors) ===
errors = np.abs(y_test - y_pred)
error_threshold = np.percentile(errors, 95)
outlier_indices = np.where(errors > error_threshold)[0]

print(f"\n🚨 Number of outliers (>{error_threshold:.6f} error): {len(outlier_indices)}")

# Safe MAPE calculation (avoid division by zero)
mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test)))) * 100
print(f"MAPE: {mape:.2f}%")

# === Save outliers to CSV ===
with open("../outliers.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "True Value", "Predicted Value", "Absolute Error"])
    for idx in outlier_indices:
        writer.writerow([idx, y_test[idx], y_pred[idx], errors[idx]])
print("💾 Outliers saved to '../outliers.csv'")

# === Plot: Predictions vs True values with outliers highlighted ===
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.2, s=10, label="All Predictions")
plt.scatter(y_test[outlier_indices], y_pred[outlier_indices],
            color='red', alpha=0.6, s=20, label="Outliers")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'k--', linewidth=1, label="Perfect Prediction")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values (Outliers Highlighted)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Histogram of errors ===
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=100, color='skyblue', edgecolor='black')
plt.axvline(error_threshold, color='red', linestyle='--', label=f'Outlier Threshold: {error_threshold:.6f}')
plt.title("Histogram of Absolute Errors")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Distribution of Target Values ===
plt.figure(figsize=(8, 4))
plt.hist(y_test, bins=100, edgecolor='black')
plt.title("Distribution of Target Values (y)")
plt.xlabel("Target Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Zoom-in on top 100 worst outliers ===
top_outliers_idx = np.argsort(errors)[-100:]

plt.figure(figsize=(10, 6))
plt.scatter(y_test[top_outliers_idx], y_pred[top_outliers_idx], color='red', s=20)
plt.plot(lims, lims, 'k--', linewidth=1)
plt.xlabel("True Values (Top 100 Worst Outliers)")
plt.ylabel("Predicted Values")
plt.title("Zoomed-In View: Top 100 Worst Outliers")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Histogram of Predicted Values ===
plt.figure(figsize=(10, 6))
plt.hist(y_pred, bins=100, color='skyblue', edgecolor='black')
plt.title("Histogram of Predicted Values")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Predicted vs True Values (again) ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, s=10)
plt.plot(lims, lims, 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print top 10 worst outliers in console ===
print("\n🔎 Top 10 Worst Outliers:")
worst_10 = np.argsort(errors)[-10:][::-1]
for idx in worst_10:
    print(f"Index {idx}: True={y_test[idx]:.6f}, Predicted={y_pred[idx]:.6f}, Error={errors[idx]:.6f}")

# === Plot: Training History ===
try:
    with open("../history.pkl", "rb") as f:
        history_dict = pickle.load(f)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['mae'], label='Training MAE', color='green')
    if 'val_mae' in history_dict:
        plt.plot(history_dict['val_mae'], label='Validation MAE', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("MAE Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print("⚠️ Could not load training history:", e)

scaler_X = joblib.load("../scaler_X.pkl")
X_test_unscaled = scaler_X.inverse_transform(X_test_scaled)

# === Find worst underapproximation ===
under_errors = y_test - y_pred  # positive = underpredict
under_idx_candidates = np.where(under_errors > 0)[0]

if len(under_idx_candidates) == 0:
    print("No underprediction outliers found.")
    worst_under_idx = None
else:
    # Get the underprediction error values only
    under_deltas = under_errors[under_idx_candidates]  # only positive values
    worst_under_idx = under_idx_candidates[np.argmax(under_deltas)]  # max underprediction

if worst_under_idx is not None:
    print(f"Worst underapproximation index: {worst_under_idx}")
    print(f"Monte Carlo Uncertainty: {y_pred_var[worst_under_idx]}")
    print(f"Monte Carlo Uncertainty z-score: {(y_pred_var[worst_under_idx] - np.mean(y_pred_var)) / (np.std(y_pred_var))}")
    print(f"Monte Carlo Uncertainty percentile: {percentile_of_value(y_pred_var, y_pred_var[worst_under_idx])}%")

    # Set window size
    window = 5
    start = max(0, worst_under_idx - window)
    end = min(len(y_test), worst_under_idx + window + 1)

    x_vals = np.arange(start, end)

    # Extract true/predicted dose segments
    true_segment = y_test[start:end]
    pred_segment = y_pred[start:end]

    # Extract SEP flux from unscaled features
    sep_flux_index = 0  # update to your SEP flux feature index
    sep_flux_segment = X_test_unscaled[start:end, sep_flux_index]

    # Plotting underapproximation
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_vals, true_segment, label="True Dose", color="blue", linewidth=2, alpha = 0.3)
    ax1.plot(x_vals, pred_segment, label="Predicted Dose", color="orange", linestyle="--", linewidth=2)
    ax1.axvline(x=worst_under_idx, color="red", linestyle=":", label="Worst Underapproximation")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Dose Rate (Gy/day)", color="black")

    # Add secondary axis for SEP flux
    ax2 = ax1.twinx()
    ax2.plot(x_vals, sep_flux_segment, label="SEP Count of Oxygen(10.0–13.1 MeV energy band)", color="red", linestyle="--", alpha=0.4)
    ax2.set_ylabel("SEP Count", color="gray")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Outlier Example: SEP Event and Dose Rate Underapproximation")
    plt.tight_layout()
    plt.savefig("../outlier_underapproximation_example.png", dpi=300)
    plt.show()
else:
    print("No underapproximation outlier to plot.")

#Find worst overapproximation
over_errors = y_pred - y_test  # positive = overpredict
over_idx_candidates = np.where(over_errors > 0)[0]

if len(over_idx_candidates) == 0:
    print("No overprediction outliers found.")
    worst_over_idx = None
else:
    # Get the overprediction error values only
    over_deltas = over_errors[over_idx_candidates]  # only positive values
    worst_over_idx = over_idx_candidates[np.argmax(over_deltas)]  # max overprediction

if worst_over_idx is not None:
    print(f"Worst overapproximation index: {worst_over_idx}")
    print(f"Monte Carlo Uncertainty: {y_pred_var[worst_over_idx]}")
    print(f"Monte Carlo Uncertainty z-score: {(y_pred_var[worst_over_idx] - np.mean(y_pred_var))/(np.std(y_pred_var))}")
    print(f"Monte Carlo Uncertainty percentile: {percentile_of_value(y_pred_var, y_pred_var[worst_over_idx])}%")

    # Set window size
    window = 5
    start = max(0, worst_over_idx - window)
    end = min(len(y_test), worst_over_idx + window + 1)

    x_vals = np.arange(start, end)

    # Extract true/predicted dose segments
    true_segment = y_test[start:end]
    pred_segment = y_pred[start:end]

    # Extract SEP flux from unscaled features
    sep_flux_index = 0  # update to your SEP flux feature index
    sep_flux_segment = X_test_unscaled[start:end, sep_flux_index]

    # Plotting overapproximation
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_vals, true_segment, label="True Dose", color="blue", linewidth=2, alpha = 0.6)
    ax1.plot(x_vals, pred_segment, label="Predicted Dose", color="orange", linestyle="--", linewidth=2)
    ax1.axvline(x=worst_over_idx, color="red", linestyle=":", label="Worst Overapproximation")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Dose Rate (Gy/day)", color="black")

    # Add secondary axis for SEP flux
    ax2 = ax1.twinx()
    ax2.plot(x_vals, sep_flux_segment, label="SEP Count of Oxygen on the 10.0–13.1 MeV energy band", color="red", linestyle="--", alpha=0.4)
    ax2.set_ylabel("SEP Count", color="gray")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Outlier Example: SEP Event and Dose Rate Overapproximation")
    plt.tight_layout()
    plt.savefig("../outlier_overapproximation_example.png", dpi=300)
    plt.show()
else:
    print("No overapproximation outlier to plot.")



uncertainty = y_pred_var.flatten()
errors = np.abs(y_test - y_pred)

# === Pearson & Spearman Correlation ===
pearson_r, pearson_p = pearsonr(uncertainty, errors)
spearman_r, spearman_p = spearmanr(uncertainty, errors)

print(f"\n📈 Pearson r = {pearson_r:.4f} (p = {pearson_p:.4e})")
print(f"📉 Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.4e})")

# === Scatter Plot with Linear Regression Line ===
plt.figure(figsize=(8, 5))
plt.scatter(uncertainty, errors, alpha=0.3, label="Data")

# Fit regression line
X_lin = uncertainty.reshape(-1, 1)
y_lin = errors
model_lin = LinearRegression()
model_lin.fit(X_lin, y_lin)
y_pred_line = model_lin.predict(X_lin)
plt.plot(uncertainty, y_pred_line, color="red", label="Linear Fit")

plt.xlabel("Predictive Variance (Uncertainty)")
plt.ylabel("Absolute Error")
plt.title(f"Uncertainty vs Prediction Error\nPearson r = {pearson_r:.4f}, Spearman ρ = {spearman_r:.4f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Set a threshold for "high uncertainty" — e.g., top 5% most uncertain predictions
uncertainty_threshold = np.percentile(uncertainty, 90)

# Flag indices where predictive variance exceeds this threshold
high_uncertainty_indices = np.where(uncertainty > uncertainty_threshold)[0]

print(f"\n🚨 Number of high-uncertainty predictions (>{uncertainty_threshold:.6f}): {len(high_uncertainty_indices)}")

# Compare how many of those are also among the actual top-5% worst errors
error_threshold = np.percentile(errors, 90)
actual_outlier_indices = np.where(errors > error_threshold)[0]

# Intersection
true_positives = np.intersect1d(high_uncertainty_indices, actual_outlier_indices)
precision = len(true_positives) / len(high_uncertainty_indices) if len(high_uncertainty_indices) > 0 else 0
recall = len(true_positives) / len(actual_outlier_indices) if len(actual_outlier_indices) > 0 else 0

print(f"✅ Outlier Thresholding Performance:")
print(f"  Precision (Correctly flagged / Flagged): {precision:.3f}")
print(f"  Recall (Correctly flagged / Actual outliers): {recall:.3f}")
actual_outliers = np.where(errors > error_threshold)[0]

thresholds = np.linspace(80, 99, 40)  # uncertainty percentile thresholds
precisions = []
recalls = []
f1_scores = []

for t in thresholds:
    uncert_thresh = np.percentile(uncertainty, t)
    high_uncertainty_idx = np.where(uncertainty > uncert_thresh)[0]

    tp = np.intersect1d(high_uncertainty_idx, actual_outliers)
    precision = len(tp) / len(high_uncertainty_idx) if len(high_uncertainty_idx) > 0 else 0
    recall = len(tp) / len(actual_outliers) if len(actual_outliers) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    if precision + recall > 0:
        f1_scores.append(2 * (precision * recall) / (precision + recall))
    else:
        f1_scores.append(0)

# Find best threshold by max F1
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', marker='o')
plt.plot(thresholds, recalls, label='Recall', marker='s')
plt.plot(thresholds, f1_scores, label='F1 Score', marker='^', linewidth=2, color='green')
plt.xlabel('Uncertainty Threshold Percentile')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score vs Uncertainty Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"🎯 Best F1 score of {best_f1:.3f} at uncertainty threshold percentile {best_threshold:.2f}")



# === Binned Average Error vs Uncertainty ===
df = pd.DataFrame({'uncertainty': uncertainty, 'error': errors})
df['uncert_bin'] = pd.qcut(df['uncertainty'], q=10, duplicates='drop')  # 10 quantile bins

bin_means = df.groupby('uncert_bin').mean()

plt.figure(figsize=(8, 5))
plt.plot(bin_means['uncertainty'], bin_means['error'], marker='o', linestyle='-', color='purple')
plt.xlabel("Mean Predictive Variance (per bin)")
plt.ylabel("Mean Absolute Error (per bin)")
plt.title("Binned Uncertainty vs Mean Absolute Error")
plt.grid(True)
plt.tight_layout()
plt.show()
