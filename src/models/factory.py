"""Fábrica de estimadores Scikit-Learn.

Centraliza a criação dos modelos em um único mapa nome → construtor. Para
acrescentar um algoritmo ao experimento basta registrar a fábrica aqui e
adicionar o nome em ``params.yaml``; nenhum outro módulo muda.
"""

from collections.abc import Callable
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor

# Estimadores que não aceitam random_state no construtor.
_STATELESS: frozenset[str] = frozenset({"dummy"})

MODEL_FACTORY: dict[str, Callable[..., BaseEstimator]] = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def create_estimator(name: str, seed: int, **hyperparams: Any) -> BaseEstimator:
    """Instancia um estimador pelo nome, com a semente já aplicada.

    Args:
        name: Chave registrada em :data:`MODEL_FACTORY`.
        seed: Semente de reprodutibilidade.
        **hyperparams: Hiperparâmetros vindos de ``params.yaml``.

    Returns:
        Estimador Scikit-Learn não ajustado.

    Raises:
        KeyError: Se ``name`` não estiver registrado.
    """
    if name not in MODEL_FACTORY:
        raise KeyError(f"Modelo desconhecido: {name!r}. Disponíveis: {sorted(MODEL_FACTORY)}")

    if name not in _STATELESS:
        hyperparams = {**hyperparams, "random_state": seed}
    return MODEL_FACTORY[name](**hyperparams)


def build_model_pipeline(name: str, seed: int, **hyperparams: Any) -> Pipeline:
    """Monta o pipeline completo: pré-processamento + classificador.

    Manter os dois passos em um único objeto significa que o artefato
    servido pela API recebe o DataFrame **cru** da sessão e cuida sozinho
    da padronização e do One-Hot — não há como treino e inferência
    divergirem.

    Args:
        name: Nome do modelo em :data:`MODEL_FACTORY`.
        seed: Semente de reprodutibilidade.
        **hyperparams: Hiperparâmetros do classificador.

    Returns:
        ``Pipeline`` não ajustado, pronto para ``fit``.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", create_estimator(name, seed, **hyperparams)),
        ]
    )
