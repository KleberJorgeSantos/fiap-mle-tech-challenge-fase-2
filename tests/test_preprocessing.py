"""Testes do pré-processador e da divisão treino/teste."""

import numpy as np
import pytest

from src.config import NUMERIC_FEATURES
from src.data.preprocessing import (
    build_preprocessor,
    get_feature_names,
    split_data,
    to_dense,
)


def test_preprocessor_expands_categoricals(features_target) -> None:
    features, _ = features_target
    transformed = build_preprocessor().fit_transform(features)
    # One-Hot sempre gera mais colunas do que as 17 de entrada.
    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > len(features.columns)


def test_numeric_columns_are_standardized(features_target) -> None:
    features, _ = features_target
    transformed = to_dense(build_preprocessor().fit_transform(features))
    numeric_block = transformed[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_feature_names_match_transformed_width(features_target) -> None:
    features, _ = features_target
    preprocessor = build_preprocessor().fit(features)
    transformed = to_dense(preprocessor.transform(features))
    assert len(get_feature_names(preprocessor)) == transformed.shape[1]


def test_unseen_category_does_not_break_transform(features_target) -> None:
    """``handle_unknown='ignore'`` é o que permite servir sessões novas."""
    features, _ = features_target
    preprocessor = build_preprocessor().fit(features)
    novel = features.head(1).copy()
    novel.loc[novel.index[0], "Month"] = "Jan"
    assert preprocessor.transform(novel).shape[0] == 1


def test_split_is_stratified_and_reproducible(features_target) -> None:
    features, target = features_target
    first = split_data(features, target, test_size=0.2, seed=42)
    second = split_data(features, target, test_size=0.2, seed=42)

    x_train, x_test, y_train, y_test = first
    assert len(x_train) + len(x_test) == len(features)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)
    assert list(x_train.index) == list(second[0].index)


def test_split_with_different_seed_produces_different_partition(features_target) -> None:
    features, target = features_target
    a = split_data(features, target, test_size=0.2, seed=1)[0]
    b = split_data(features, target, test_size=0.2, seed=2)[0]
    assert list(a.index) != list(b.index)


def test_to_dense_accepts_plain_arrays() -> None:
    assert to_dense([[1, 2], [3, 4]]).shape == (2, 2)
