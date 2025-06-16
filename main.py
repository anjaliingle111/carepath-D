import os
import joblib
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def load_data():
    df = pd.read_csv("data/processed.csv")
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    X_train, X_test, y_train, y_test = load_data()
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average="macro")
    return score


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(f"  F1 Score: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Final model training using best params
    X_train, X_test, y_train, y_test = load_data()
    final_model = RandomForestClassifier(**study.best_trial.params, random_state=42)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)
    final_score = f1_score(y_test, final_preds, average="macro")
    print(f"Final model F1 score on test set: {final_score}")

    # ✅ Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/final_random_forest.pkl")


if __name__ == "__main__":
    main()
