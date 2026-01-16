import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed/student_data.csv")

# Define the schema we expect
EXPECTED_COLUMNS = [
    "attendance",
    "study_hours",
    "previous_marks",
    "assignments",
    "result"
]

NUMERIC_COLUMNS = ["attendance", "study_hours", "previous_marks", "assignments"]
TARGET_COLUMN = "result"


def validate_data(df: pd.DataFrame) -> None:
    # 1) File not empty
    if df.empty:
        raise ValueError("Validation failed: Dataset is empty.")

    # 2) Column check (exact match in order is strict; you can relax if needed)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_cols:
        raise ValueError(f"Validation failed: Missing columns: {missing_cols}")
    if extra_cols:
        raise ValueError(f"Validation failed: Unexpected columns: {extra_cols}")

    # 3) Null check
    null_counts = df[EXPECTED_COLUMNS].isna().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Validation failed: Null values found: {bad}")

    # 4) Type check (must be numeric)
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Validation failed: Column '{col}' must be numeric, got {df[col].dtype}")

    # 5) Range checks (basic sanity rules)
    if not df["attendance"].between(0, 100).all():
        raise ValueError("Validation failed: 'attendance' must be between 0 and 100.")

    if (df["study_hours"] < 0).any():
        raise ValueError("Validation failed: 'study_hours' must be >= 0.")

    if not df["previous_marks"].between(0, 100).all():
        raise ValueError("Validation failed: 'previous_marks' must be between 0 and 100.")

    if not df["assignments"].between(0, 100).all():
        raise ValueError("Validation failed: 'assignments' must be between 0 and 100.")

    # 6) Target check (must be 0 or 1)
    valid_targets = set(df[TARGET_COLUMN].unique())
    if not valid_targets.issubset({0, 1}):
        raise ValueError(f"Validation failed: 'result' must contain only 0/1. Found: {sorted(valid_targets)}")


def main():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. "
            f"Run: python src/ingestion.py"
        )

    print("Reading processed data for validation...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Running validation checks...")
    validate_data(df)

    print("✅ Validation passed. Data is clean and matches expected schema.")


if __name__ == "__main__":
    main()
