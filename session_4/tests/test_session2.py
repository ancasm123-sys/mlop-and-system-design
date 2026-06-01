import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def create_sample_churn_df():
    return pd.DataFrame(
        {
            "RowNumber": [1, 2, 3, 4],
            "CustomerId": [1001, 1002, 1003, 1004],
            "Surname": ["Smith", "Jones", "Brown", "Lee"],
            "CreditScore": [700, 650, 720, 580],
            "Geography": ["France", "Germany", "Spain", "France"],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Age": [35.0, 45.0, 28.0, 52.0],
            "Tenure": [5, 3, 7, 2],
            "Balance": [100000.0, 0.0, 50000.0, 80000.0],
            "NumOfProducts": [2, 1, 2, 1],
            "HasCrCard": [1.0, 0.0, 1.0, 1.0],
            "IsActiveMember": [1.0, 0.0, 1.0, 0.0],
            "EstimatedSalary": [60000.0, 80000.0, 45000.0, 70000.0],
            "Exited": [0, 1, 0, 1],
        }
    )


def test_drop_identifier_columns():
    df = create_sample_churn_df()
    columns_to_drop = ["RowNumber", "CustomerId", "Surname"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)
    assert "RowNumber" not in df.columns
    assert "CustomerId" not in df.columns
    assert "Surname" not in df.columns
    assert "CreditScore" in df.columns


def test_categorical_features_detected():
    df = create_sample_churn_df().drop(columns=["RowNumber", "CustomerId", "Surname"])
    X = df.drop(columns=["Exited"])
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    assert "Geography" in categorical_features
    assert "Gender" in categorical_features


def test_numeric_features_detected():
    df = create_sample_churn_df().drop(columns=["RowNumber", "CustomerId", "Surname"])
    X = df.drop(columns=["Exited"])
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    assert "CreditScore" in numeric_features
    assert "Age" in numeric_features
    assert "Balance" in numeric_features


def test_target_column_is_binary():
    df = create_sample_churn_df()
    assert df["Exited"].isin([0, 1]).all()


def test_pipeline_fits_without_error():
    df = create_sample_churn_df().drop(columns=["RowNumber", "CustomerId", "Surname"])
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    categorical_features = X.select_dtypes(include=["object"]).columns
    numeric_features = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
