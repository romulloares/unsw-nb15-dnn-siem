import numpy as np
import pandas as pd

from src.data import prepare_data, split_features_target


def make_frame(size: int, start_id: int = 0) -> pd.DataFrame:
    labels = np.array(([0, 1] * ((size + 1) // 2))[:size])
    return pd.DataFrame(
        {
            "id": np.arange(start_id, start_id + size),
            "dur": np.linspace(0.1, 5.0, size),
            "sbytes": np.arange(size) * 10 + 1,
            "proto": ["tcp", "udp"] * (size // 2) + (["tcp"] if size % 2 else []),
            "service": ["http", "-"] * (size // 2) + (["http"] if size % 2 else []),
            "state": ["FIN", "INT"] * (size // 2) + (["FIN"] if size % 2 else []),
            "attack_cat": ["Normal" if value == 0 else "Generic" for value in labels],
            "label": labels,
        }
    )


def test_sensitive_columns_are_removed() -> None:
    features, target = split_features_target(make_frame(10))
    assert "id" not in features.columns
    assert "attack_cat" not in features.columns
    assert "label" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_preprocessing_produces_compatible_float32_arrays() -> None:
    train = make_frame(40)
    test = make_frame(12, start_id=100)
    test.loc[0, "proto"] = "icmp"  # categoria não vista deve ser ignorada sem erro

    data = prepare_data(train, test, validation_size=0.25, random_state=42)

    assert data.x_train.dtype == np.float32
    assert data.x_val.dtype == np.float32
    assert data.x_test.dtype == np.float32
    assert data.x_train.shape[1] == data.x_val.shape[1] == data.x_test.shape[1]
    assert not np.isnan(data.x_train).any()
    assert len(data.feature_names) == data.x_train.shape[1]
