# 💊 CarePath AI – Diabetes Readmission Predictor

Predict whether a diabetic patient will be readmitted to a hospital using real clinical data. Built as a **complete MLOps pipeline** with deployment-ready **FastAPI** backend and CI/CD automation.

---

## 📌 Problem Statement

Hospital readmissions, especially among diabetic patients, are expensive and often preventable. By predicting which patients are at risk of being readmitted, hospitals can improve patient care and reduce costs.

---

## 🎯 Objective

Develop a machine learning pipeline that:
- Predicts patient readmission
- Serves predictions via a FastAPI endpoint
- Is production-ready with CI/CD and cloud deployment

---

## 📁 Dataset

- **Source**: [Kaggle - Diabetes Readmission](https://www.kaggle.com/datasets/aaron7sun/diabetes-health-indicators-dataset)
- **Records**: ~101,766
- **Features**: Demographics, diagnosis, medications, lab results, admission info
- **Target**: Binary classification (`Readmitted` vs `Not Readmitted`)

---

## 🧱 Project Structure

```
carepath-ai/
│
├── src/                   # Core ML code
│   ├── preprocessing.py
│   └── train.py
│
├── models/                # GDrive model downloads on runtime
│
├── app.py                 # FastAPI app
├── streamlit_app.py       # (Optional) Streamlit UI
│
├── .github/workflows/     # CI/CD pipelines
│   ├── ci.yml
│   ├── mlops.yml
│   └── render.yml
│
├── render.yaml            # Render deployment config
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Area             | Tools Used                            |
|------------------|----------------------------------------|
| Language         | Python 3.11                            |
| ML               | scikit-learn, pandas                   |
| API              | FastAPI, Uvicorn                       |
| CI/CD            | GitHub Actions                         |
| Deployment       | Render (free tier)                     |
| Model Hosting    | Google Drive + `gdown`                 |
| Testing          | Pytest                                |
| Optional UI      | Streamlit (local only)                 |

---

## 🚀 Deployment

### 🔗 Live API:  
**[https://carepath-d.onrender.com/docs](https://carepath-d.onrender.com/docs)**  
Use this Swagger UI to interact with the `/predict` endpoint.

### 🧠 Model Auto-Download
Models are too large for GitHub, so they are hosted on Google Drive. On app startup, models are downloaded automatically using `gdown`.

---

## ✅ CI/CD Pipelines

| Pipeline       | Description                                      |
|----------------|--------------------------------------------------|
| `ci.yml`       | Runs tests (`pytest`) on every push              |
| `mlops.yml`    | Automatically retrains model on code changes     |
| `render.yml`   | Deploys FastAPI to Render.com                    |

---

## 🧪 Example Input

Here's a sample POST request body for `/predict`:

```json
{
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
```

---

## 📉 Model Info

| Model             | Accuracy | Comments                        |
|------------------|----------|---------------------------------|
| Random Forest     | ~85%     | Chosen for best performance     |
| Others Tried      | Logistic, XGBoost, etc.                   |

Artifacts:
- `final_random_forest.pkl`
- `target_encoder.pkl`
- `model_columns.pkl`

---

## ⚠️ Known Limitations

- Render Free Tier limited to 512MB RAM → not ideal for heavy models
- Readmission labels are noisy (e.g. ambiguous timeframes)
- Streamlit not hosted (only local `.py`)

---

## 🔮 Future Improvements

- Add SHAP for model explainability
- Host Streamlit UI separately (e.g. Streamlit Cloud)
- Add logging & monitoring (e.g. Prometheus)
- Secure API access with API keys

---

## 👩‍💻 Author

Made with 💻 by **Anjali**  
GitHub: [@anjaliingle111](https://github.com/anjaliingle111)

---

## 📜 License

This project is open-source under the MIT License.
