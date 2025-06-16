# src/ingest.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data

import pandas as pd
from src.preprocessing import preprocess_data  # adjust if your structure differs

def main():
    raw_path = "data/diabetic_data.csv"
    processed_path = "data/processed.csv"

    df = pd.read_csv(raw_path)
    X, y = preprocess_data(df)

    df_processed = X.copy()
    df_processed['readmitted'] = y

    df_processed.to_csv(processed_path, index=False)
    print(f"[✓] Processed data saved to {processed_path}")

if __name__ == "__main__":
    main()

