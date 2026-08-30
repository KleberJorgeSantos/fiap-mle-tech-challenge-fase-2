"""API de inferência de propensão de compra.

Carrega o modelo campeão do MLflow Model Registry
(``models:/purchase-intent-classifier@champion``) e, se o Registry não
estiver disponível, cai para o artefato local ``models/model.joblib``
produzido pelo ``dvc repro``. Isso mantém a imagem Docker utilizável mesmo
sem acesso ao servidor de tracking.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.errors as pa_errors
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import HealthResponse, PredictResponse, SessionFeatures
from src.config import get_settings, load_params
from src.data.loader import validate_schema
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"model": None, "source": None}


def _registry_is_reachable() -> bool:
    """Diz se vale a pena tentar o Registry.

    Apontar o MLflow para um SQLite inexistente faz com que ele **crie** o
    banco e rode todas as migrações do Alembic — vários segundos de startup
    para depois falhar. Dentro do container, onde só existe o artefato
    local, esse caminho é puro desperdício.
    """
    uri = get_settings().mlflow_tracking_uri
    if uri.startswith("sqlite:///"):
        return Path(uri.removeprefix("sqlite:///")).exists()
    return True


def _load_from_registry() -> Any:
    """Tenta carregar o modelo campeão do MLflow Model Registry."""
    import mlflow

    settings = get_settings()
    if not _registry_is_reachable():
        raise FileNotFoundError(f"Backend do MLflow indisponível: {settings.mlflow_tracking_uri}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.sklearn.load_model(settings.model_uri)


def _load_from_disk() -> Any:
    """Carrega o artefato local gerado pelo pipeline DVC."""
    import joblib

    return joblib.load(get_settings().model_dir / "model.joblib")


def load_model() -> tuple[Any, str | None]:
    """Resolve o modelo a servir, com fallback do Registry para o disco.

    Returns:
        Tupla ``(modelo, origem)``; ``(None, None)`` se nenhuma fonte funcionar.
    """
    for source, loader in (
        ("mlflow-registry", _load_from_registry),
        ("local-joblib", _load_from_disk),
    ):
        try:
            model = loader()
        except Exception as error:  # noqa: BLE001 — fallback é intencional
            logger.warning("Falha ao carregar modelo via %s: %s", source, error)
            continue
        logger.info("Modelo carregado de %s", source)
        return model, source
    return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma única vez, na subida do processo."""
    setup_logging()
    _state["model"], _state["source"] = load_model()
    _state["threshold"] = load_params()["evaluate"]["threshold"]
    yield
    _state.clear()


app = FastAPI(
    title="Purchase Intent API",
    description="Propensão de compra em sessões de e-commerce — FIAP Tech Challenge Fase 2.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Informa se a API tem um modelo pronto para servir."""
    loaded = _state.get("model") is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_source=_state.get("source"),
    )


def _require_model() -> Any:
    """Devolve o modelo carregado ou recusa a requisição com 503."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Nenhum modelo carregado. Execute `dvc repro` antes de subir a API.",
        )
    return model


def _to_validated_frame(session: SessionFeatures) -> pd.DataFrame:
    """Converte o payload em DataFrame e aplica o contrato do Pandera.

    O mesmo schema que valida o dataset de treino valida a requisição — é o
    que impede a API de aceitar uma sessão que o pipeline rejeitaria.
    """
    frame = pd.DataFrame([session.model_dump(by_alias=True)])
    try:
        return validate_schema(frame)
    except pa_errors.SchemaError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Falha de validação do schema", "errors": str(error)},
        ) from error


@app.post("/predict", response_model=PredictResponse, tags=["inferência"])
def predict(session: SessionFeatures) -> PredictResponse:
    """Prevê a probabilidade de a sessão terminar em compra.

    Args:
        session: Features comportamentais da sessão.

    Returns:
        Probabilidade de conversão e a classe predita.

    Raises:
        HTTPException: 503 se não houver modelo; 422 se o schema falhar.
    """
    model = _require_model()
    frame = _to_validated_frame(session)

    threshold = _state.get("threshold", 0.5)
    probability = float(model.predict_proba(frame)[0, 1])
    return PredictResponse(
        purchase_probability=probability,
        will_purchase=probability >= threshold,
        threshold=threshold,
    )
