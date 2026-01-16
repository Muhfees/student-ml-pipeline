import pandas as pd
from pathlib import Path

# Define paths
RAW_DATA_PATH = Path("data/raw/student_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/student_data.csv")

def ingest_data():
    print("Reading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Basic data info:")
    print(df.info())

    print("Saving processed data...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Data ingestion completed successfully.")

if __name__ == "__main__":
    ingest_data()
