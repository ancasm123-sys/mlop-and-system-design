import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def load_data():
    churn_df = pd.read_csv("Churn_modeling_train_test.csv")
    bank_df = pd.read_csv("bank-full_train_test.csv")

    print("Churn dataset shape:", churn_df.shape)
    print("Bank dataset shape:", bank_df.shape)

    return churn_df, bank_df


def train_churn_model(churn_df):
    print("\n========== CHURN MODEL ==========")

    # Drop columns that are identifiers or not useful for prediction
    columns_to_drop = ["RowNumber", "CustomerId", "Surname"]

    for col in columns_to_drop:
        if col in churn_df.columns:
            churn_df = churn_df.drop(columns=col)

    target = "Exited"

    X = churn_df.drop(columns=target)
    y = churn_df[target]

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))


def train_bank_model(bank_df):
    print("\n========== BANK MODEL ==========")

    target = "y"

    X = bank_df.drop(columns=target)
    y = bank_df[target].map({"no": 0, "yes": 1})

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))


def main():
    churn_df, bank_df = load_data()

    train_churn_model(churn_df)
    train_bank_model(bank_df)


if __name__ == "__main__":
    main()