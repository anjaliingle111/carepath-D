from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
import gdown

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import preprocess_data

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "final_random_forest.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "target_encoder.pkl")  # Optional if needed

# Google Drive links
MODEL_URL = "https://drive.google.com/uc?id=1PfuiFk_lyCJ4aqPcyvsCrNLK9Cr6cO2_"
# Add these if needed
# COLUMNS_URL = "..."
# ENCODER_URL = "..."

# Download function
def download_if_missing(filepath, url):
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        gdown.download(url, filepath, quiet=False)

# Ensure all required files are present
download_if_missing(MODEL_PATH, MODEL_URL)
# download_if_missing(COLUMNS_PATH, COLUMNS_URL)
# download_if_missing(ENCODER_PATH, ENCODER_URL)

# Load model and metadata
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# Input Schema
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

# Prediction route
@app.post("/predict")
def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.model_dump()])

        # Preprocess for inference
        X = preprocess_data(input_df, training=False)

        # Predict
        prediction = model.predict(X)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
