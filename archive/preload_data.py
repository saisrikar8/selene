import pandas as pd
import numpy as np
import ast
import re

from tqdm import tqdm


def process_cell(cell):
    # Check if cell is NaN or not a list-like object
    if pd.isna(cell) or not isinstance(cell, str):
        return [cell]

    try:
        # Try to parse using literal_eval (works for lists of strings like format 2)
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, list) or isinstance(parsed, np.ndarray):
            # Check if it's format 1: list of numbers
            if all(isinstance(x, (int, float)) or re.match(r'^-?\d+(\.\d+)?$', str(x)) for x in parsed):
                return [float(x) for x in parsed]

            # Format 2: list of strings with range expressions
            midpoints = []
            for item in parsed:
                match = re.search(r'(\d+(\.\d+)?)-(\d+(\.\d+)?)', item)
                if match:
                    low = float(match.group(1))
                    high = float(match.group(3))
                    midpoint = (low + high) / 2
                    midpoints.append(midpoint)
            return midpoints
    except (ValueError, SyntaxError):
        pass

    # If not parseable, return as single-element list
    return [cell]


def expand_dataframe(df):
    new_rows = []
    for _, row in tqdm(df.iterrows(), desc="expanding dataframe"):
        new_row = []
        for cell in tqdm(row, desc= f'processing row{_ + 1}'):
            expanded = process_cell(cell)
            new_row.extend(expanded)
        new_rows.append(new_row)

    # Create column names based on maximum row length
    max_len = max(len(row) for row in new_rows)
    col_names = [f'col_{i}' for i in range(max_len)]
    return pd.DataFrame(new_rows, columns=col_names)


# Replace 'input.csv' with your actual filename
input_file = 'deleted_rows.csv'
output_file = 'processed_output.csv'

print("Reading csv file")
df = pd.read_csv(input_file, dtype=str)  # Read everything as string for uniform
print("Processing cells")
# processing
processed_df = expand_dataframe(df)
processed_df.to_csv(output_file, index=False)

print(f"Processed dataframe saved to {output_file}")