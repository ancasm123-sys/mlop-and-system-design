import pandas as pd
from src.transform import Transformer
from src.train import train_model


def create_churn_sample():
    """Churn dataset sample (same dataset used in session 2)."""
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


def test_transformer_drops_identifier_columns():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    assert "RowNumber" not in transformed.columns
    assert "CustomerId" not in transformed.columns
    assert "Surname" not in transformed.columns


def test_transformer_encodes_gender():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    assert transformed["Gender"].isin([0, 1]).all()


def test_transformer_one_hot_encodes_geography():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    assert "Geography" not in transformed.columns
    ohe_cols = [c for c in transformed.columns if c.startswith("Geography_")]
    assert len(ohe_cols) > 0


def test_train_model_returns_predictions():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    model = train_model(df=transformed, target_column="Exited")
    X = transformed.drop(columns=["Exited"])
    predictions = model.predict(X)
    assert all(p in [0, 1] for p in predictions)


def test_train_model_output_shape():
    df = create_churn_sample()
    transformed = Transformer().transform(df)
    model = train_model(df=transformed, target_column="Exited")
    X = transformed.drop(columns=["Exited"])
    predictions = model.predict(X)
    assert len(predictions) == len(X)
