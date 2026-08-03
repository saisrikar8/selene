import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    logging.info("Loading preprocessed dataset 'your_data.csv'")
    df = pd.read_csv('../aligned_and_expanded_data.csv')

    # Drop eband label columns
    eband_cols = [col for col in df.columns if col.startswith('sis_label_ebands')]
    logging.info(f"Dropping {len(eband_cols)} eband label columns.")
    df = df.drop(columns=eband_cols)

    # Drop timestamp columns (anything with 'epoch' or 'time' in name)
    timestamp_cols = [col for col in df.columns if 'epoch' in col.lower() or 'time' in col.lower()]
    logging.info(f"Dropping {len(timestamp_cols)} timestamp columns.")
    df = df.drop(columns=timestamp_cols)

    # Identify crater columns for grouping (e.g. dose_D1, dose_D2 if exists)
    crater_cols = [col for col in df.columns if col.startswith('dose_')]
    logging.info(f"Using crater columns for grouping: {crater_cols}")

    # Group by crater columns and average all other numeric columns
    logging.info("Grouping by crater columns and averaging other data...")
    grouped_df = df.groupby(crater_cols).mean().reset_index()

    logging.info(f"Grouped data shape: {grouped_df.shape}")

    # Save to CSV
    output_file = '../aligned_averaged_by_crater.csv'
    grouped_df.to_csv(output_file, index=False)
    logging.info(f"Saved grouped and averaged data to '{output_file}'")

if __name__ == "__main__":
    main()
