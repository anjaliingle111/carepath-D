# streamlit_app.py
import streamlit as st
import requests

st.title("CarePath AI - Diabetes Readmission Predictor")

st.write("Fill in patient details below:")

# Sample inputs (you can customize this)
data = {
    "race": st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]),
    "gender": st.selectbox("Gender", ["Male", "Female"]),
    "age": st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]),
    "weight": st.text_input("Weight", "?"),
    "payer_code": st.text_input("Payer Code", "?"),
    "medical_specialty": st.text_input("Medical Specialty", "?"),
    "admission_type_id": st.number_input("Admission Type ID", 1, 8, 1),
    "discharge_disposition_id": st.number_input("Discharge Disposition ID", 1, 30, 1),
    "admission_source_id": st.number_input("Admission Source ID", 1, 25, 1),
    "time_in_hospital": st.slider("Time in Hospital (days)", 1, 14, 3),
    "num_lab_procedures": st.slider("Number of Lab Procedures", 1, 132, 40),
    "num_procedures": st.slider("Number of Procedures", 0, 6, 1),
    "num_medications": st.slider("Number of Medications", 1, 81, 10),
    "number_outpatient": st.slider("Number Outpatient", 0, 42, 0),
    "number_emergency": st.slider("Number Emergency", 0, 76, 0),
    "number_inpatient": st.slider("Number Inpatient", 0, 21, 0),
    "diag_1": st.text_input("Diagnosis 1", "428"),
    "diag_2": st.text_input("Diagnosis 2", "276"),
    "diag_3": st.text_input("Diagnosis 3", "250"),
    "number_diagnoses": st.slider("Number of Diagnoses", 1, 16, 5),
    "max_glu_serum": st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"]),
    "A1Cresult": st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"]),
}

# Meds - keep them default for now
meds = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide_metformin", "glipizide_metformin",
    "glimepiride_pioglitazone", "metformin_rosiglitazone", "metformin_pioglitazone"
]

for med in meds:
    data[med] = "No"

data["change"] = "No"
data["diabetesMed"] = "Yes"

if st.button("Predict"):
    # Replace this with your deployed Render domain
    API_URL = "https://carepath-api.onrender.com/predict"
    
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {'Readmitted' if result['prediction'] else 'Not Readmitted'}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
