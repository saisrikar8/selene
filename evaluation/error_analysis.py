import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import re

PROCESSED_CSV_FILE = "../deleted_rows.csv"
PARTICLES = ['He', 'C', 'O', 'Fe', 'Si']
EPSILON = 1e-6

df = pd.read_csv(PROCESSED_CSV_FILE)
df = df[~df['dose_D1'].isna()]

feature_arrays = []
target_array = []
feature_names = []

# Helper function to find and sort columns by number suffix for a given prefix and element
def find_ordered_cols(df_cols, prefix, element):
    pattern = re.compile(rf"^{re.escape(prefix)}{element}_(\d+)$")
    matched = []
    for col in df_cols:
        m = pattern.match(col)
        if m:
            idx = int(m.group(1))
            matched.append((idx, col))
    matched.sort(key=lambda x: x[0])
    return [col for _, col in matched]

# === Collect SIS columns ===
sis_columns_ordered = []
for p in PARTICLES:
    # find all flux and count columns with numeric suffixes
    flux_cols = find_ordered_cols(df.columns, 'sis_flux_', p)
    cnt_cols = find_ordered_cols(df.columns, 'sis_cnt_', p)
    sis_columns_ordered.extend(flux_cols)
    sis_columns_ordered.extend(cnt_cols)

# === Collect ACE columns ===
ace_elements = ['C', 'O', 'Fe', 'Si']
ace_columns_ordered = []
for p in ace_elements:
    flux_cols = find_ordered_cols(df.columns, 'ace_flux_', p)
    cnt_cols = find_ordered_cols(df.columns, 'ace_cnt_', p)
    ace_columns_ordered.extend(flux_cols)
    ace_columns_ordered.extend(cnt_cols)

feature_names = sis_columns_ordered + ace_columns_ordered

for idx, row in df.iterrows():
    row_values = []
    skip_row = False

    # Extract SIS columns values
    for col in sis_columns_ordered:
        if col not in df.columns:
            print(f"Missing column: {col}, skipping row {idx}")
            skip_row = True
            break
        val = row[col]
        if pd.isna(val):
            print(f"NaN value in {col} at row {idx}, skipping")
            skip_row = True
            break
        try:
            val_float = float(val)
            if val_float < 0:
                print(f"Negative value in {col} at row {idx}")
                skip_row = True
                break
            row_values.append(val_float)
        except Exception as e:
            print(f"Error parsing {col} in row {idx}: {e}")
            skip_row = True
            break
    if skip_row:
        continue

    # Extract ACE columns values
    for col in ace_columns_ordered:
        if col not in df.columns:
            print(f"Missing column: {col}, skipping row {idx}")
            skip_row = True
            break
        val = row[col]
        if pd.isna(val):
            print(f"NaN value in {col} at row {idx}, skipping")
            skip_row = True
            break
        try:
            val_float = float(val)
            if val_float < 0:
                print(f"Negative value in {col} at row {idx}")
                skip_row = True
                break
            row_values.append(val_float)
        except Exception as e:
            print(f"Error parsing {col} in row {idx}: {e}")
            skip_row = True
            break
    if skip_row:
        continue

    feature_arrays.append(row_values)
    target_array.append(float(row['dose_D1']))

X = np.array(feature_arrays, dtype=np.float32)
y = np.array(target_array, dtype=np.float32)

print(f"Loaded features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("Minimum value in X before transform:", X.min())
print("Any negatives?", np.any(X < 0))

# Apply transformations
X_log = np.log1p(X + EPSILON)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Plot histograms of raw, log1p, and standardized features
for i in range(min(6, X.shape[1])):
    plt.figure(figsize=(18, 5))

    # Original
    plt.subplot(1, 3, 1)
    sns.histplot(X[:, i], bins=100, color='blue')
    plt.title(f"Original: {feature_names[i]}")
    plt.xlabel(feature_names[i])
    plt.grid(True)

    # Log1p
    plt.subplot(1, 3, 2)
    sns.histplot(X_log[:, i], bins=100, color='orange')
    plt.title(f"log1p: {feature_names[i]}")
    plt.xlabel(f"{feature_names[i]}_log1p")
    plt.grid(True)

    # Standard Scaled
    plt.subplot(1, 3, 3)
    sns.histplot(X_scaled[:, i], bins=100, color='green')
    plt.title(f"Standard Scaled: {feature_names[i]}")
    plt.xlabel(f"{feature_names[i]}_scaled")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
