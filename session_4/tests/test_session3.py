import importlib.util
import sys
import os
import unittest.mock
import pandas as pd

# Mock mlflow before loading exercise_support (it imports mlflow at module level)
for _mod in ["mlflow", "mlflow.models"]:
    sys.modules.setdefault(_mod, unittest.mock.MagicMock())

# Load ClassSuportTransformer from "Session 3/" (space in folder name requires importlib)
_session3_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Session 3", "exercise_support.py")
)
_spec = importlib.util.spec_from_file_location("exercise_support", _session3_file)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ClassSuportTransformer = _module.ClassSuportTransformer


def test_drop_columns_removes_identifiers():
    transformer = ClassSuportTransformer()
    df = pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [1001, 1002],
            "Surname": ["Smith", "Jones"],
            "CreditScore": [700, 650],
        }
    )
    result = transformer._drop_columns(df)
    assert "RowNumber" not in result.columns
    assert "CustomerId" not in result.columns
    assert "Surname" not in result.columns
    assert "CreditScore" in result.columns


def test_map_binary_female_is_one():
    # Session 3 maps Female -> 1, Male -> 0 (opposite of session 4)
    transformer = ClassSuportTransformer()
    df = pd.DataFrame({"Gender": ["Female", "Male", "Female"]})
    result = transformer._map_binary_variables(df)
    assert result["Gender"].iloc[0] == 1
    assert result["Gender"].iloc[1] == 0


def test_map_binary_male_is_zero():
    transformer = ClassSuportTransformer()
    df = pd.DataFrame({"Gender": ["Male"]})
    result = transformer._map_binary_variables(df)
    assert result["Gender"].iloc[0] == 0


def test_balance_dataset_equal_classes():
    transformer = ClassSuportTransformer()
    df = pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45],
            "Exited": [0, 0, 0, 1, 1],
        }
    )
    balanced = transformer.balance_dataset(df)
    counts = balanced["Exited"].value_counts()
    assert counts[0] == counts[1]


def test_balance_dataset_already_balanced():
    transformer = ClassSuportTransformer()
    df = pd.DataFrame(
        {
            "Age": [25, 30, 35, 40],
            "Exited": [0, 0, 1, 1],
        }
    )
    balanced = transformer.balance_dataset(df)
    counts = balanced["Exited"].value_counts()
    assert counts[0] == counts[1]
