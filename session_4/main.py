from src.source import load_data
from src.transform import Transformer
from src.train import train_model
from src.store import store_model
from metadata import MODEL_NAME, TARGET_COLUMN


def main():
    df = load_data(file_name="Churn_Modelling_train_test.csv")
    df = Transformer().transform(df)
    model = train_model(df=df, target_column=TARGET_COLUMN)
    store_model(model=model, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
