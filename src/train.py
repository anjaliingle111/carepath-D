import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
import mlflow
import mlflow.sklearn
from datetime import datetime

from preprocessing import preprocess_data

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def train_model():
    # Load your dataset
    df = pd.read_csv("C:/Users/Anjali/carepath-ai/data/diabetic_data.csv")

    # Preprocess data
    X, y = preprocess_data(df, training=True)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    model_path = "models/final_random_forest.pkl"
    joblib.dump(model, model_path)

    # Save model columns
    feature_path = "models/model_columns.pkl"
    joblib.dump(X.columns.tolist(), feature_path)

    # Evaluate
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    print(f"Training Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    # ====================
    # MLflow Logging
    # ====================
    mlflow.set_experiment("diabetes_readmission")

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(feature_path)
        mlflow.log_artifact("models/target_encoder.pkl")

        # Log model (optional, useful for MLflow UI)
        mlflow.sklearn.log_model(model, "random_forest_model")

if __name__ == "__main__":
    train_model()
