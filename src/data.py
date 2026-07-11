"""Carregamento, validação e pré-processamento do UNSW-NB15."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    ATTACK_CATEGORY_COLUMN,
    DEFAULT_CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_FILENAME,
    TRAIN_FILENAME,
)


@dataclass(frozen=True)
class DataSplits:
    """Matrizes transformadas e rótulos de treino, validação e teste."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def _validate_frame(frame: pd.DataFrame, source: Path | str) -> None:
    if frame.empty:
        raise ValueError(f"O arquivo {source} não contém registros.")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Coluna alvo '{TARGET_COLUMN}' ausente em {source}. "
            f"Colunas encontradas: {list(frame.columns)}"
        )

    labels = set(pd.Series(frame[TARGET_COLUMN]).dropna().astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"A coluna '{TARGET_COLUMN}' deve ser binária (0/1), mas contém {sorted(labels)}."
        )


def load_official_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os CSVs oficiais de treino e teste.

    Parameters
    ----------
    data_dir:
        Diretório que contém os dois arquivos oficiais.
    """

    data_dir = Path(data_dir)
    train_path = data_dir / TRAIN_FILENAME
    test_path = data_dir / TEST_FILENAME

    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Arquivos do UNSW-NB15 não encontrados:\n- "
            + "\n- ".join(missing)
            + "\nBaixe-os da fonte oficial e salve-os em data/raw/."
        )

    train_frame = _normalize_columns(pd.read_csv(train_path, low_memory=False))
    test_frame = _normalize_columns(pd.read_csv(test_path, low_memory=False))
    _validate_frame(train_frame, train_path)
    _validate_frame(test_frame, test_path)

    train_frame[TARGET_COLUMN] = train_frame[TARGET_COLUMN].astype("int64")
    test_frame[TARGET_COLUMN] = test_frame[TARGET_COLUMN].astype("int64")
    return train_frame, test_frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos e alvo, removendo campos que causam vazamento."""

    _validate_frame(frame, "DataFrame")
    columns_to_drop = [
        column
        for column in (*IDENTIFIER_COLUMNS, ATTACK_CATEGORY_COLUMN, TARGET_COLUMN)
        if column in frame.columns
    ]
    features = frame.drop(columns=columns_to_drop).copy()
    target = frame[TARGET_COLUMN].astype("int64").copy()
    return features, target


def infer_column_groups(
    features: pd.DataFrame,
    preferred_categorical: Iterable[str] = DEFAULT_CATEGORICAL_COLUMNS,
) -> tuple[list[str], list[str]]:
    """Infere colunas categóricas e numéricas de forma defensiva."""

    categorical = {
        column for column in preferred_categorical if column in features.columns
    }
    categorical.update(
        column
        for column in features.select_dtypes(include=["object", "category", "string"]).columns
    )
    categorical_columns = sorted(categorical)
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("Nenhuma coluna de entrada foi encontrada após a remoção dos rótulos.")
    return categorical_columns, numeric_columns


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Cria o pré-processador sem ajustá-lo aos dados."""

    categorical_columns, numeric_columns = infer_column_groups(features)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Cria treino/validação/teste e ajusta o pré-processador apenas no treino."""

    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size deve estar entre 0 e 1.")

    x_full, y_full = split_features_target(train_frame)
    x_test_raw, y_test = split_features_target(test_frame)

    missing_in_test = sorted(set(x_full.columns) - set(x_test_raw.columns))
    extra_in_test = sorted(set(x_test_raw.columns) - set(x_full.columns))
    if missing_in_test or extra_in_test:
        raise ValueError(
            "As colunas de treino e teste não são compatíveis. "
            f"Ausentes no teste: {missing_in_test}; extras no teste: {extra_in_test}."
        )
    x_test_raw = x_test_raw[x_full.columns]

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_full,
        y_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_full,
    )

    preprocessor = build_preprocessor(x_train_raw)
    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float32, copy=False)
    x_val = preprocessor.transform(x_val_raw).astype(np.float32, copy=False)
    x_test = preprocessor.transform(x_test_raw).astype(np.float32, copy=False)

    feature_names = tuple(map(str, preprocessor.get_feature_names_out()))
    return DataSplits(
        x_train=x_train,
        y_train=y_train.to_numpy(dtype=np.int64),
        x_val=x_val,
        y_val=y_val.to_numpy(dtype=np.int64),
        x_test=x_test,
        y_test=y_test.to_numpy(dtype=np.int64),
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
