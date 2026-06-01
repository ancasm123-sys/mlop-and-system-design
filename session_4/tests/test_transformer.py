import pandas as pd
from src.transform import Transformer


def test_map_binary_column_to_int():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female", "Male", "Female"],
        }
    )
    expected_df = pd.DataFrame({"Gender": [1, 0, 1, 0]})
    transformed_df = transformer._map_binary_column_to_int(df)
    pd.testing.assert_frame_equal(transformed_df, expected_df)


def test_map_binary_column_female_is_zero():
    transformer = Transformer()
    df = pd.DataFrame({"Gender": ["Female"]})
    result = transformer._map_binary_column_to_int(df)
    assert result["Gender"].iloc[0] == 0


def test_map_binary_column_male_is_one():
    transformer = Transformer()
    df = pd.DataFrame({"Gender": ["Male"]})
    result = transformer._map_binary_column_to_int(df)
    assert result["Gender"].iloc[0] == 1


def test_drop_columns():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [1001, 1002],
            "Surname": ["Smith", "Jones"],
            "CreditScore": [700, 650],
        }
    )
    result = df.drop(transformer.drop_columns, axis=1)
    assert "RowNumber" not in result.columns
    assert "CustomerId" not in result.columns
    assert "Surname" not in result.columns
    assert "CreditScore" in result.columns
