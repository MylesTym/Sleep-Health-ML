from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI()

class InputData(BaseModel):
    Age: int
    SleepDuration: float
    StressLevel: int
    PhysicalActivityLevel: int

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.Age, data.SleepDuration, data.StressLevel, data.PhysicalActivityLevel]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    label = label_encoder.inverse_transform(prediction)[0]
    return {"prediction": label}