import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    logging.info("Loading data for correlation analysis...")

    # Load relevant columns: dose + ace + sis cnt and flux columns
    df = pd.read_csv('../aligned_and_expanded_data.csv', usecols=lambda c: (
        c == 'dose_D1' or
        c.startswith('ace_cnt_') or
        c.startswith('ace_flux_') or
        c.startswith('sis_cnt_') or
        c.startswith('sis_flux_')
    ))

    logging.info(f"Data shape: {df.shape}")

    # Compute correlation of all features with dose_D1
    dose_corr = df.corr()['dose_D1'].drop('dose_D1')  # exclude self

    # Print all correlations with dose_D1, sorted by absolute value
    logging.info("Full list of correlations with dose_D1 (sorted by |value|):")
    for feature, corr_value in dose_corr.abs().sort_values(ascending=False).items():
        logging.info(f"{feature}: {dose_corr[feature]:.4f}")

    # Get top 25 by absolute correlation
    top_25 = dose_corr.abs().sort_values(ascending=False).head(25)
    top_25_features = top_25.index.tolist()

    logging.info("Top 5 features most correlated with dose_D1:")
    logging.info(dose_corr.loc[top_25.head(5).index])

    # Prepare 1D heatmap data: correlation values reshaped as a column (25 rows, 1 col)
    corr_values = dose_corr.loc[top_25_features].values.reshape(-1, 1)

    plt.figure(figsize=(4, 12))  # tall & narrow figure for vertical features

    ax = sns.heatmap(
        corr_values,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        vmin=-1,  # force full colormap range
        vmax=1,
        cbar=True,
        yticklabels=top_25_features,
        xticklabels=['dose_D1'],
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.title("Correlation of Top 25 Features")
    plt.tight_layout()

    output_file = "../correlation_with_dose_top25_1d_heatmap_vertical.png"
    plt.savefig(output_file)
    logging.info(f"Vertical 1D correlation heatmap saved as '{output_file}'")

    plt.show()


if __name__ == "__main__":
    main()
