MODELS_FOLDER = "session_4/models"
DATASETS_FOLDER = "session_4/datasets"
MODEL_NAME = "class_model-Andre"

COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
BINARY_FEATURES = ["Gender"]
ONE_HOT_ENCODE_COLUMNS = ["Geography"]
TARGET_COLUMN = "Exited"

MODEL_PARAMS = {
    "max_depth": 5,
    "random_state": 42,
}
