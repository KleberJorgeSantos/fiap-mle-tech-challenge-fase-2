"""Testes da orquestração de treino."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.factory import build_model_pipeline
from src.models.train import cross_validate, select_best, train_all_models, train_single_model


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


@patch("src.models.train.log_model_run")
def test_train_single_model_returns_metrics_and_the_mlflow_run_id(
    mock_log: MagicMock, datasets
) -> None:
    mock_log.return_value = "run-abc"

    metrics, run_id = train_single_model(
        "logistic_regression", {"max_iter": 50}, datasets, seed=42, folds=3
    )

    assert run_id == "run-abc"
    assert "cv_pr_auc" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    mock_log.assert_called_once()


@patch("src.models.train.log_model_run")
def test_train_single_model_uses_the_configured_hyperparams(
    mock_log: MagicMock, datasets
) -> None:
    mock_log.return_value = "run-x"

    train_single_model("logistic_regression", {"max_iter": 30}, datasets, seed=42, folds=3)

    logged_params = mock_log.call_args.kwargs["params"]
    assert logged_params["max_iter"] == 30
    assert logged_params["seed"] == 42


@patch("src.models.train.log_model_run")
def test_train_all_models_trains_every_configured_model(mock_log: MagicMock, datasets) -> None:
    mock_log.side_effect = ["run-a", "run-b"]

    results, run_ids = train_all_models(
        model_names=["dummy", "logistic_regression"],
        hyperparams={"logistic_regression": {"max_iter": 50}},
        datasets=datasets,
        seed=42,
        folds=3,
    )

    assert set(results) == {"dummy", "logistic_regression"}
    assert run_ids == {"dummy": "run-a", "logistic_regression": "run-b"}


@patch("src.models.train.log_model_run")
def test_train_all_models_falls_back_to_empty_hyperparams(
    mock_log: MagicMock, datasets
) -> None:
    """Um modelo sem entrada em `hyperparams` ainda deve treinar (defaults)."""
    mock_log.return_value = "run-dummy"

    results, _ = train_all_models(
        model_names=["dummy"], hyperparams={}, datasets=datasets, seed=42, folds=3
    )

    assert "dummy" in results
