import pandas as pd
import category_encoders as ce
import joblib
import os

def preprocess_data(
    df,
    training=True,
    encoder_path='models/target_encoder.pkl',
    model_cols_path='models/model_columns.pkl',
    return_encoder=False
):
    df = df.copy()

    # Drop ID columns if present
    df.drop(columns=[col for col in ['encounter_id', 'patient_nbr'] if col in df.columns], inplace=True, errors='ignore')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop columns with >10% missing
    df.dropna(thresh=int(0.9 * len(df)), axis=1, inplace=True)

    # Drop constant columns
    constant_cols = df.columns[df.nunique() == 1].tolist()
    df.drop(columns=constant_cols, inplace=True)

    if training:
        # Clean target variable
        df = df[df['readmitted'] != 'NA']
        df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

        y = df['readmitted']
        X = df.drop(columns=['readmitted'])

        # Encode categorical
        cat_cols = X.select_dtypes(include='object').columns.tolist()
        encoder = ce.TargetEncoder(cols=cat_cols)
        X[cat_cols] = encoder.fit_transform(X[cat_cols], y)

        # Save encoder + model column order
        joblib.dump(encoder, encoder_path)
        joblib.dump(X.columns.tolist(), model_cols_path)

        return (X, y, encoder) if return_encoder else (X, y)

    else:
        # Inference mode
        if 'readmitted' in df.columns:
            df.drop(columns=['readmitted'], inplace=True)

        # Load encoder and column list
        encoder = joblib.load(encoder_path)
        model_cols = joblib.load(model_cols_path)

        cat_cols = encoder.cols

        # Ensure categorical columns exist
        for col in cat_cols:
            if col not in df.columns:
                df[col] = 'missing'

        df[cat_cols] = encoder.transform(df[cat_cols])

        # Add missing model columns
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns
        X = df[model_cols]

        return X
