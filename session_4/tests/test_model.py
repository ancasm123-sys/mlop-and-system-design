import os
import glob
import joblib
import pandas as pd
import pytest


def load_latest_model():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
    assert model_files, "No model found in session_4/models/"
    return joblib.load(max(model_files))


def create_transformed_sample():
    """Sample data in the format AFTER transformation (ready for model input)."""
    return pd.DataFrame(
        {
            "CreditScore": [700, 580, 650],
            "Gender": [1, 0, 1],          # Male=1, Female=0
            "Age": [35.0, 52.0, 28.0],
            "Tenure": [5, 2, 7],
            "Balance": [100000.0, 80000.0, 0.0],
            "NumOfProducts": [2, 1, 2],
            "HasCrCard": [1.0, 1.0, 0.0],
            "IsActiveMember": [1.0, 0.0, 1.0],
            "EstimatedSalary": [60000.0, 70000.0, 45000.0],
            "Geography_Germany": [0.0, 1.0, 0.0],
            "Geography_Spain": [0.0, 0.0, 1.0],
            "Geography_nan": [0.0, 0.0, 0.0],  # NaN geography records
        }
    )


def test_model_loads():
    model = load_latest_model()
    assert model is not None


def test_model_predictions_are_binary():
    model = load_latest_model()
    X = create_transformed_sample()
    predictions = model.predict(X)
    assert all(p in [0, 1] for p in predictions)


def test_model_output_shape():
    model = load_latest_model()
    X = create_transformed_sample()
    predictions = model.predict(X)
    assert len(predictions) == len(X)


def test_model_predict_proba():
    model = load_latest_model()
    X = create_transformed_sample()
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert all(round(p[0] + p[1], 5) == 1.0 for p in proba)


def test_model_single_row():
    model = load_latest_model()
    X = create_transformed_sample().iloc[[0]]
    predictions = model.predict(X)
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
