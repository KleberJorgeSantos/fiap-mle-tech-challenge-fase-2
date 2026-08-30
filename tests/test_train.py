"""Testes da orquestração de treino."""

import pytest

from src.models.factory import build_model_pipeline
from src.models.train import cross_validate, select_best


@pytest.fixture
def datasets(features_target) -> dict:
    features, target = features_target
    split = len(features) // 2
    return {
        "x_train": features.iloc[:split],
        "y_train": target.iloc[:split],
        "x_test": features.iloc[split:],
        "y_test": target.iloc[split:],
    }


def test_cross_validate_returns_a_probability_like_score(datasets) -> None:
    model = build_model_pipeline("logistic_regression", seed=42, max_iter=100)
    score = cross_validate(model, datasets["x_train"], datasets["y_train"], folds=3, seed=42)
    assert 0.0 <= score <= 1.0


def test_cross_validate_is_deterministic(datasets) -> None:
    model = build_model_pipeline("logistic_regression", seed=42, max_iter=100)
    args = (datasets["x_train"], datasets["y_train"], 3, 42)
    assert cross_validate(model, *args) == cross_validate(model, *args)


def test_select_best_picks_the_highest_metric() -> None:
    results = {
        "a": {"pr_auc": 0.10},
        "b": {"pr_auc": 0.90},
        "c": {"pr_auc": 0.45},
    }
    assert select_best(results, "pr_auc") == "b"


def test_select_best_respects_the_chosen_metric() -> None:
    results = {
        "a": {"pr_auc": 0.9, "recall": 0.1},
        "b": {"pr_auc": 0.1, "recall": 0.9},
    }
    assert select_best(results, "recall") == "b"
