# tests/test_api.py
import sys
import os

# Add root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80]",
        "weight": "?",
        "payer_code": "?",
        "medical_specialty": "?",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 1,
        "time_in_hospital": 3,
        "num_lab_procedures": 41,
        "num_procedures": 0,
        "num_medications": 1,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "428",
        "diag_2": "250.02",
        "diag_3": "401.9",
        "number_diagnoses": 3,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "insulin": "No",
        "glyburide_metformin": "No",
        "glipizide_metformin": "No",
        "glimepiride_pioglitazone": "No",
        "metformin_rosiglitazone": "No",
        "metformin_pioglitazone": "No",
        "change": "No",
        "diabetesMed": "Yes"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
