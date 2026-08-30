"""Promoção do modelo campeão no MLflow Model Registry.

Usa **alias** (``@champion``) em vez de ``stage="Production"``: os stages
foram depreciados no MLflow 2.9 em favor de aliases, que são ponteiros
móveis para uma versão específica. A API carrega o modelo por
``models:/purchase-intent-classifier@champion`` e passa a servir a nova
versão sem nenhuma alteração de código.
"""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def register_model(run_id: str, model_name: str) -> str:
    """Registra o modelo de um run como nova versão no Registry.

    Args:
        run_id: Run do MLflow que contém o artefato ``model``.
        model_name: Nome do modelo registrado.

    Returns:
        O número da versão criada.
    """
    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    logger.info("Modelo '%s' registrado como versão %s", model_name, version.version)
    return version.version


def promote_to_alias(
    model_name: str,
    version: str,
    alias: str,
    tags: dict[str, str] | None = None,
) -> None:
    """Aponta o alias para a versão indicada e grava tags de proveniência.

    Args:
        model_name: Nome do modelo registrado.
        version: Versão a ser promovida.
        alias: Alias a apontar (ex.: ``champion``).
        tags: Metadados opcionais gravados na versão.
    """
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    for key, value in (tags or {}).items():
        client.set_model_version_tag(name=model_name, version=version, key=key, value=value)
    logger.info("Versão %s promovida para @%s", version, alias)


def register_best_model(run_id: str, metric_name: str, metric_value: float) -> dict[str, str]:
    """Registra e promove o melhor run em uma única chamada.

    Args:
        run_id: Run vencedor.
        metric_name: Métrica usada na seleção.
        metric_value: Valor da métrica vencedora.

    Returns:
        Resumo do que foi promovido, para gravar em ``registered_model.json``.
    """
    settings = get_settings()
    version = register_model(run_id, settings.mlflow_registered_model)
    promote_to_alias(
        model_name=settings.mlflow_registered_model,
        version=version,
        alias=settings.mlflow_model_alias,
        tags={
            "selection_metric": metric_name,
            "selection_value": f"{metric_value:.4f}",
            "source_run_id": run_id,
        },
    )
    return _promotion_summary(settings, version, run_id, metric_name, metric_value)


def _promotion_summary(
    settings: Settings,
    version: str,
    run_id: str,
    metric_name: str,
    metric_value: float,
) -> dict[str, str]:
    """Monta o resumo serializável da promoção, com tudo em ``str``."""
    return {
        "model_name": settings.mlflow_registered_model,
        "version": version,
        "alias": settings.mlflow_model_alias,
        "model_uri": settings.model_uri,
        "run_id": run_id,
        "selection_metric": metric_name,
        "selection_value": f"{metric_value:.6f}",
    }
