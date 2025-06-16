# api/main.py

from fastapi import FastAPI
from api.schema import PatientData
from api.predict import predict_diabetes

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is up!"}

@app.post("/predict")
def get_prediction(data: PatientData):
    prediction = predict_diabetes(data.dict())
    return {"prediction": prediction}
