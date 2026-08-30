"""Testes da API de inferência.

O modelo é injetado diretamente em ``_state`` para que a suíte não dependa
de um artefato treinado nem de um servidor MLflow no ar.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.models.factory import build_model_pipeline

_EXAMPLE = json.loads(Path("examples/session.json").read_text(encoding="utf-8"))


@pytest.fixture
def client_without_model() -> TestClient:
    with TestClient(api.app) as client:
        api._state["model"] = None
        api._state["source"] = None
        yield client


@pytest.fixture
def client_with_model(features_target) -> TestClient:
    features, target = features_target
    model = build_model_pipeline("random_forest", seed=42, n_estimators=10)
    model.fit(features, target)

    with TestClient(api.app) as client:
        api._state["model"] = model
        api._state["source"] = "test-stub"
        api._state["threshold"] = 0.5
        yield client


def test_health_reports_degraded_without_a_model(client_without_model: TestClient) -> None:
    body = client_without_model.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_health_reports_ok_with_a_model(client_with_model: TestClient) -> None:
    body = client_with_model.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_source"] == "test-stub"


def test_predict_returns_503_without_a_model(client_without_model: TestClient) -> None:
    response = client_without_model.post("/predict", json=_EXAMPLE)
    assert response.status_code == 503


def test_predict_returns_a_probability(client_with_model: TestClient) -> None:
    body = client_with_model.post("/predict", json=_EXAMPLE).json()
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert isinstance(body["will_purchase"], bool)
    assert body["threshold"] == 0.5


def test_will_purchase_follows_the_threshold(client_with_model: TestClient) -> None:
    body = client_with_model.post("/predict", json=_EXAMPLE).json()
    assert body["will_purchase"] == (body["purchase_probability"] >= body["threshold"])


def test_out_of_range_bounce_rate_is_rejected(client_with_model: TestClient) -> None:
    payload = {**_EXAMPLE, "BounceRates": 7.5}
    assert client_with_model.post("/predict", json=payload).status_code == 422


def test_missing_field_is_rejected(client_with_model: TestClient) -> None:
    payload = {key: value for key, value in _EXAMPLE.items() if key != "PageValues"}
    assert client_with_model.post("/predict", json=payload).status_code == 422


def test_load_model_returns_none_when_no_source_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_load_from_registry", lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(api, "_load_from_disk", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    assert api.load_model() == (None, None)


def test_load_model_falls_back_to_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(api, "_load_from_registry", lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(api, "_load_from_disk", lambda: sentinel)
    model, source = api.load_model()
    assert model is sentinel
    assert source == "local-joblib"


def test_registry_is_skipped_when_sqlite_file_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita pagar as migrações do Alembic por um banco que não existe."""
    from src.config import Settings

    monkeypatch.setattr(
        api,
        "get_settings",
        lambda: Settings(_env_file=None, mlflow_tracking_uri="sqlite:///nao-existe.db"),
    )
    assert api._registry_is_reachable() is False


def test_remote_tracking_server_is_always_attempted(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import Settings

    monkeypatch.setattr(
        api,
        "get_settings",
        lambda: Settings(_env_file=None, mlflow_tracking_uri="http://mlflow.interno:5000"),
    )
    assert api._registry_is_reachable() is True
