"""Testes de leitura, tipagem e validação do dataset."""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.config import CODE_LIKE_FEATURES, TARGET
from src.data.loader import (
    cast_code_like_columns,
    load_raw,
    validate_schema,
)


def test_load_raw_reads_csv(tmp_path, raw_frame: pd.DataFrame) -> None:
    path = tmp_path / "sample.csv"
    raw_frame.to_csv(path, index=False)
    assert len(load_raw(path)) == len(raw_frame)


def test_code_like_columns_become_strings(raw_frame: pd.DataFrame) -> None:
    casted = cast_code_like_columns(raw_frame)
    for column in CODE_LIKE_FEATURES:
        assert casted[column].dtype == object


def test_cast_does_not_mutate_the_input(raw_frame: pd.DataFrame) -> None:
    before = raw_frame["Browser"].dtype
    cast_code_like_columns(raw_frame)
    assert raw_frame["Browser"].dtype == before


def test_target_is_binary_and_features_exclude_it(features_target, expected_columns) -> None:
    features, target = features_target
    assert set(target.unique()) <= {0, 1}
    assert TARGET not in features.columns
    assert list(features.columns) == expected_columns


def test_validate_schema_accepts_valid_frame(features_target) -> None:
    features, _ = features_target
    assert len(validate_schema(features)) == len(features)


def test_validate_schema_rejects_out_of_range_bounce_rate(features_target) -> None:
    features, _ = features_target
    corrupted = features.copy()
    corrupted.loc[corrupted.index[0], "BounceRates"] = 5.0
    with pytest.raises(SchemaError):
        validate_schema(corrupted)
