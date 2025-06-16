# generate_sample.py
import pandas as pd

# Load the full dataset
df = pd.read_csv("data/processed.csv")

# Take a small sample (e.g., 100 rows)
sample_df = df.sample(n=100, random_state=42)

# Save it
sample_df.to_csv("data/processed_sample.csv", index=False)
