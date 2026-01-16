import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = Path("data/processed/student_data.csv")
OUTPUT_DIR = Path("data/processed")

TARGET_COL = "result"

def main():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. Run: python src/ingestion.py"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Basic: separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # Save splits
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("✅ Preprocessing completed.")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test  shape: X={X_test.shape}, y={y_test.shape}")

if __name__ == "__main__":
    main()
