import pandas as pd
import os
import shutil
from tqdm import tqdm  # prettier in Colab

def load_dataset(drive_path, chunksize=500_000):
    try:
        with open(drive_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        total_chunks = total_lines // chunksize + 1

        # Step 4: Load using chunks + tqdm
        columns_to_delete = [
            "ace_fp_doy","ace_fp_year","ace_livetime"
        ]

        chunks = []
        print("📊 Loading CSV with progress bar:")
        for chunk in tqdm(pd.read_csv(drive_path, chunksize=chunksize, low_memory=False), total=total_chunks):
            chunk = chunk.drop(columns=columns_to_delete, errors='ignore')
            chunks.append(chunk)

        # Step 5: Concatenate all chunks
        data = pd.concat(chunks, ignore_index=True)
        print(f"✅ Dataset loaded successfully with shape: {data.shape}")
        data.to_csv("deleted_rows.csv", index=False)
        return data

    except Exception as e:
        print(f"❌ Load unsuccessful: {e}")
        return None

if __name__ == "__main__":
    # Make sure to define CSV_FILE before this line
    df = load_dataset("./deleted_rows.csv")
