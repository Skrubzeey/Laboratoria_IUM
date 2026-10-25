import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from src.utils.data_prep import load_data, split_data


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature_1": range(10),
        "feature_2": range(10, 20),
        "Outcome": [0, 1] * 5
    })


def test_split_data_size(sample_df):
    X_train, X_test, y_train, y_test = split_data(
        sample_df, test_size=0.2, seed=42
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_split_data_total_size(sample_df):
    X_train, X_test, y_train, y_test = split_data(
        sample_df, test_size=0.3, seed=42
    )

    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)


def test_split_data_reproducibility(sample_df):
    result_1 = split_data(sample_df, 0.2, 42)
    result_2 = split_data(sample_df, 0.2, 42)

    for first, second in zip(result_1, result_2):
        if isinstance(first, pd.DataFrame):
            assert_frame_equal(first, second)
        else:
            assert_series_equal(first, second)



def test_load_data(tmp_path):
    # Tworzymy przykładowy plik CSV
    df = pd.DataFrame({
        "Glucose": [100, 120],
        "Outcome": [0, 1]
    })

    df.to_csv(tmp_path / "diabetes.csv", index=False)

    # Wczytujemy dane z folderu
    result = load_data(str(tmp_path))

    pd.testing.assert_frame_equal(result, df)


def test_load_data_missing_file(tmp_path):
    # Folder nie zawiera pliku diabetes.csv
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path))
