"""Treino e comparação dos modelos candidatos.

Cada modelo passa por validação cruzada estratificada no conjunto de treino
(estimativa honesta da generalização) e depois é reajustado na base cheia de
treino para ser avaliado no teste. Um run do MLflow é aberto por modelo.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import evaluate_model
from src.models.factory import build_model_pipeline
from src.tracking.mlflow_utils import log_model_run

logger = logging.getLogger(__name__)

_CV_SCORING = "average_precision"
_INPUT_EXAMPLE_ROWS = 5


def cross_validate(
    model: Pipeline, features: pd.DataFrame, target: pd.Series, folds: int, seed: int
) -> float:
    """Calcula a PR-AUC média por validação cruzada estratificada.

    Args:
        model: Pipeline não ajustado.
        features: Features de treino.
        target: Alvo de treino.
        folds: Número de dobras.
        seed: Semente do embaralhamento.

    Returns:
        Média da métrica ``average_precision`` entre as dobras.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, features, target, cv=splitter, scoring=_CV_SCORING)
    return float(np.mean(scores))


def train_single_model(
    name: str,
    hyperparams: dict[str, Any],
    datasets: dict[str, Any],
    seed: int,
    folds: int,
) -> tuple[dict[str, float], str]:
    """Treina, avalia e registra um único modelo candidato.

    Args:
        name: Nome do modelo na fábrica.
        hyperparams: Hiperparâmetros vindos de ``params.yaml``.
        datasets: Mapa com ``x_train``, ``y_train``, ``x_test``, ``y_test``.
        seed: Semente de reprodutibilidade.
        folds: Dobras da validação cruzada.

    Returns:
        Tupla ``(métricas de teste, run_id do MLflow)``.
    """
    logger.info("--- Treinando %s ---", name)
    model = build_model_pipeline(name, seed, **hyperparams)

    cv_score = cross_validate(model, datasets["x_train"], datasets["y_train"], folds, seed)
    model.fit(datasets["x_train"], datasets["y_train"])

    proba = model.predict_proba(datasets["x_test"])[:, 1]
    metrics = evaluate_model(datasets["y_test"].to_numpy(), proba)
    metrics["cv_pr_auc"] = cv_score
    logger.info("%s → cv_pr_auc=%.4f | test_pr_auc=%.4f", name, cv_score, metrics["pr_auc"])

    run_id = log_model_run(
        run_name=name,
        params={"model": name, "seed": seed, "cv_folds": folds, **hyperparams},
        metrics=metrics,
        model=model,
        input_example=datasets["x_train"].head(_INPUT_EXAMPLE_ROWS),
    )
    return metrics, run_id


def train_all_models(
    model_names: list[str],
    hyperparams: dict[str, dict[str, Any]],
    datasets: dict[str, Any],
    seed: int,
    folds: int,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Treina todos os candidatos configurados.

    Args:
        model_names: Nomes dos modelos a treinar.
        hyperparams: Mapa ``nome -> hiperparâmetros``.
        datasets: Conjuntos de treino e teste.
        seed: Semente de reprodutibilidade.
        folds: Dobras da validação cruzada.

    Returns:
        Tupla ``(métricas por modelo, run_id por modelo)``.
    """
    results: dict[str, dict[str, float]] = {}
    run_ids: dict[str, str] = {}
    for name in model_names:
        results[name], run_ids[name] = train_single_model(
            name, hyperparams.get(name, {}), datasets, seed, folds
        )
    return results, run_ids


def select_best(results: dict[str, dict[str, float]], metric: str) -> str:
    """Elege o modelo com o maior valor da métrica de seleção.

    Args:
        results: Métricas por modelo.
        metric: Nome da métrica de seleção.

    Returns:
        Nome do modelo vencedor.
    """
    best = max(results, key=lambda name: results[name][metric])
    logger.info("Campeão: %s (%s=%.4f)", best, metric, results[best][metric])
    return best
