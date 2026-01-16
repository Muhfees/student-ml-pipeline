import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    # Load data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("📊 Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = MODEL_DIR / "logistic_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")

if __name__ == "__main__":
    main()
