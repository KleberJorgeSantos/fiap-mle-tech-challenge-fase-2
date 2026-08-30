"""Sanidade de importação e coerência entre configuração e artefatos."""

import importlib

import pytest

_MODULES = [
    "src.config",
    "src.logging_config",
    "src.data.download",
    "src.data.loader",
    "src.data.preprocessing",
    "src.models.factory",
    "src.models.train",
    "src.evaluation.metrics",
    "src.tracking.mlflow_utils",
    "src.tracking.registry",
    "src.api.main",
    "src.api.schemas",
    "src.pipelines.download",
    "src.pipelines.preprocess",
    "src.pipelines.train",
    "src.pipelines.evaluate",
    "src.pipelines.register",
]


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_api_schema_covers_every_configured_feature() -> None:
    """O contrato da API não pode divergir do esquema do pipeline."""
    from src.api.schemas import SessionFeatures
    from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    aliases = {field.alias for field in SessionFeatures.model_fields.values()}
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == aliases


def test_dvc_pipeline_declares_the_five_stages() -> None:
    import yaml

    with open("dvc.yaml", encoding="utf-8") as handle:
        stages = yaml.safe_load(handle)["stages"]
    assert list(stages) == ["download", "preprocess", "train", "evaluate", "register"]
