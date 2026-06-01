import pandas as pd
from src.transform import Transformer
from src.train import train_model
from src.store import store_model
import os
import glob


def create_churn_sample():
    """Churn dataset sample (same dataset used in session 3)."""
    return pd.DataFrame(
        {
            "RowNumber": [1, 2, 3, 4, 5, 6],
            "CustomerId": [1001, 1002, 1003, 1004, 1005, 1006],
            "Surname": ["Smith", "Jones", "Brown", "Lee", "Chen", "Wang"],
            "CreditScore": [700, 580, 650, 720, 610, 680],
            "Geography": ["France", "Germany", "Spain", "France", "Germany", "Spain"],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "Age": [35.0, 52.0, 28.0, 45.0, 33.0, 60.0],
            "Tenure": [5, 2, 7, 3, 8, 1],
            "Balance": [100000.0, 80000.0, 0.0, 50000.0, 120000.0, 0.0],
            "NumOfProducts": [2, 1, 2, 1, 2, 1],
            "HasCrCard": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "IsActiveMember": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "EstimatedSalary": [60000.0, 70000.0, 45000.0, 90000.0, 55000.0, 80000.0],
            "Exited": [0, 1, 0, 1, 0, 1],
        }
    )


def test_full_pipeline_transform_and_train():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    model = train_model(df=transformed, target_column="Exited")
    assert model is not None


def test_pipeline_predict_proba():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    model = train_model(df=transformed, target_column="Exited")
    X = transformed.drop(columns=["Exited"])
    proba = model.predict_proba(X)
    assert proba.shape[1] == 2
    assert all(round(p[0] + p[1], 5) == 1.0 for p in proba)


def test_pipeline_predictions_are_binary():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    model = train_model(df=transformed, target_column="Exited")
    X = transformed.drop(columns=["Exited"])
    predictions = model.predict(X)
    assert all(p in [0, 1] for p in predictions)


def test_store_model_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_FOLDER", str(tmp_path))

    import src.store as store_module
    original = store_module.MODELS_FOLDER

    store_module.MODELS_FOLDER = str(tmp_path)

    df = create_churn_sample()
    transformed = Transformer().transform(df)
    model = train_model(df=transformed, target_column="Exited")
    store_model(model=model, model_name="class_model-Andre")

    saved_files = list(tmp_path.glob("*.joblib"))
    assert len(saved_files) == 1

    store_module.MODELS_FOLDER = original
