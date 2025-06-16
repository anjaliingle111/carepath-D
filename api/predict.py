# api/predict.py

import joblib
import pandas as pd

model = joblib.load("models/final_random_forest.pkl")

def predict_diabetes(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return prediction
