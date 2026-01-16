from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Student Performance Predictor")

MODEL_PATH = Path("models/logistic_model.joblib")

class StudentInput(BaseModel):
    attendance: float
    study_hours: float
    previous_marks: float
    assignments: float

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python src/train.py"
        )
    model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"status": "ok", "message": "Student ML API is running"}

@app.post("/predict")
def predict(payload: StudentInput):
    # Keep feature order same as training data
    X = pd.DataFrame([{
        "attendance": payload.attendance,
        "study_hours": payload.study_hours,
        "previous_marks": payload.previous_marks,
        "assignments": payload.assignments
    }])

    pred = int(model.predict(X)[0])
    label = "PASS" if pred == 1 else "FAIL"
    return {"prediction": pred, "label": label}
