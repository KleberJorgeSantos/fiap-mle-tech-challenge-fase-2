"""Testes das métricas e da análise de custo."""

import numpy as np
import pytest

from src.evaluation.metrics import comparison_table, cost_analysis, evaluate_model


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0, 0, 1, 1])
    return y_true, np.array([0.01, 0.02, 0.98, 0.99])


def test_perfect_predictions_score_one(perfect_predictions) -> None:
    metrics = evaluate_model(*perfect_predictions)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_all_expected_metrics_are_present(perfect_predictions) -> None:
    metrics = evaluate_model(*perfect_predictions)
    assert set(metrics) == {"roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"}


def test_threshold_changes_the_predicted_class() -> None:
    y_true = np.array([0, 1])
    y_proba = np.array([0.3, 0.6])
    assert evaluate_model(y_true, y_proba, threshold=0.5)["recall"] == pytest.approx(1.0)
    assert evaluate_model(y_true, y_proba, threshold=0.9)["recall"] == pytest.approx(0.0)


def test_cost_weights_false_negatives_more_heavily() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0])  # 1 FP e 1 FN
    result = cost_analysis(y_true, y_pred, cost_false_positive=5.0, cost_false_negative=50.0)
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["business_cost"] == pytest.approx(55.0)


def test_cost_is_zero_when_predictions_are_perfect() -> None:
    y_true = np.array([0, 1, 1])
    result = cost_analysis(y_true, y_true, cost_false_positive=5.0, cost_false_negative=50.0)
    assert result["business_cost"] == 0.0


def test_comparison_table_sorts_by_the_selection_metric() -> None:
    results = {
        "weak": {"pr_auc": 0.2, "roc_auc": 0.5},
        "strong": {"pr_auc": 0.8, "roc_auc": 0.9},
    }
    table = comparison_table(results, sort_by="pr_auc")
    assert list(table.index) == ["strong", "weak"]
    assert table.index.name == "model"
