"""Camada fina sobre o MLflow Tracking.

Encapsula o setup do tracking URI e o registro de runs para que o resto do
código não precise conhecer a API do MLflow. O backend é SQLite: o Model
Registry **não funciona** com o store baseado em arquivos (``./mlruns``).
"""

import logging
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator

from src.config import get_settings
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def setup_tracking() -> str:
    """Aponta o MLflow para o backend configurado e garante o experimento.

    Returns:
        O ID do experimento ativo.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(settings.mlflow_experiment_name)

    # Ao inicializar o backend, o MLflow reconfigura o logger raiz: troca o
    # nosso handler (stdout) pelo dele (stderr) e sobe o nível para WARNING,
    # engolindo todo o log INFO do pipeline. Reaplicar a nossa configuração
    # devolve o controle — sem isto o `dvc repro` roda mudo.
    setup_logging()

    logger.info(
        "MLflow: uri=%s | experimento=%s",
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )
    return experiment.experiment_id


def log_model_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: BaseEstimator,
    input_example: pd.DataFrame,
) -> str:
    """Registra um run completo: parâmetros, métricas e o modelo.

    A assinatura é inferida do exemplo de entrada, o que faz o MLflow
    validar o schema automaticamente na hora de servir.

    Args:
        run_name: Nome legível do run (o nome do modelo).
        params: Hiperparâmetros a logar.
        metrics: Métricas a logar.
        model: Pipeline Scikit-Learn já ajustado.
        input_example: Algumas linhas de features cruas, para a assinatura.

    Returns:
        O ``run_id`` gerado pelo MLflow.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=infer_signature(input_example, model.predict_proba(input_example)),
            input_example=input_example,
        )
        logger.info("Run '%s' registrado (run_id=%s)", run_name, run.info.run_id)
        return run.info.run_id
