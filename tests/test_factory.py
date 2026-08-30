"""Testes da fábrica de estimadores."""

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import load_params
from src.models.factory import MODEL_FACTORY, build_model_pipeline, create_estimator


def test_every_model_in_params_is_registered() -> None:
    for name in load_params()["train"]["models"]:
        assert name in MODEL_FACTORY


def test_create_estimator_injects_the_seed() -> None:
    estimator = create_estimator("random_forest", seed=42, n_estimators=5)
    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.random_state == 42


def test_dummy_does_not_receive_random_state() -> None:
    """``DummyClassifier`` com ``strategy='prior'`` não aceita semente útil."""
    estimator = create_estimator("dummy", seed=42, strategy="prior")
    assert isinstance(estimator, DummyClassifier)


def test_unknown_model_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Modelo desconhecido"):
        create_estimator("xgboost_turbo", seed=42)


def test_pipeline_has_preprocessor_then_classifier() -> None:
    pipeline = build_model_pipeline("logistic_regression", seed=42, max_iter=50)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]


def test_pipeline_trains_on_raw_dataframe(features_target) -> None:
    """O artefato servido recebe o DataFrame cru — sem pré-processo externo."""
    features, target = features_target
    pipeline = build_model_pipeline("random_forest", seed=42, n_estimators=10)
    pipeline.fit(features, target)
    proba = pipeline.predict_proba(features.head(3))
    assert proba.shape == (3, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
