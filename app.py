from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
import gdown

app = FastAPI()

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Google Drive File IDs
GDRIVE_FILES = {
    "final_random_forest.pkl": "1PfuiFk_lyCJ4aqPcyvsCrNLK9Cr6cO2_",
    "model_columns.pkl": "1DjNUvXHXC-VLprxk_-aKns10rXWlbWwA",
    "target_encoder.pkl": "1hqlt69U4H3sf1b3wGhKCQ-W0QcT6YKHk",
}

# Model paths
MODEL_PATH = os.path.join(MODELS_DIR, "final_random_forest.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "target_encoder.pkl")

# Function to download missing models
def ensure_models_exist():
    for filename, file_id in GDRIVE_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

# Add src/ to path and import preprocessing
sys.path.append(os.path.join(BASE_DIR, "src"))
from preprocessing import preprocess_data

# Define input schema
class PatientData(BaseModel):
    race: str
    gender: str
    age: str
    weight: str
    payer_code: str
    medical_specialty: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    insulin: str
    glyburide_metformin: str
    glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str
    change: str
    diabetesMed: str

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Lazy download
        ensure_models_exist()

        # Load models
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)

        # Preprocess input
        input_df = pd.DataFrame([data.model_dump()])
        X = preprocess_data(input_df, training=False)

        # Predict
        prediction = model.predict(X)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
