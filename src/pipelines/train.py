"""Estágio DVC ``train`` — treina os candidatos e elege o campeão."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import TARGET, Settings, get_settings, load_params
from src.evaluation.metrics import comparison_table
from src.logging_config import setup_logging
from src.models.factory import build_model_pipeline
from src.models.train import select_best, train_all_models
from src.tracking.mlflow_utils import setup_tracking

logger = logging.getLogger(__name__)


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lê um Parquet gerado pelo estágio de pré-processamento.

    Args:
        path: Caminho do arquivo.

    Returns:
        Tupla ``(features, alvo)``.
    """
    frame = pd.read_parquet(path)
    return frame.drop(columns=[TARGET]), frame[TARGET]


def load_datasets(processed_dir: Path) -> dict[str, Any]:
    """Carrega treino e teste em um único dicionário.

    Args:
        processed_dir: Diretório com ``train.parquet`` e ``test.parquet``.

    Returns:
        Mapa com ``x_train``, ``y_train``, ``x_test`` e ``y_test``.
    """
    x_train, y_train = load_split(processed_dir / "train.parquet")
    x_test, y_test = load_split(processed_dir / "test.parquet")
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def fit_champion(name: str, datasets: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    """Reajusta o modelo vencedor no conjunto de treino.

    O ajuste é feito **apenas no treino**, não em treino+teste: o artefato
    servido precisa ser exatamente aquele que as métricas do relatório
    descrevem.

    Args:
        name: Nome do modelo campeão.
        datasets: Conjuntos carregados por :func:`load_datasets`.
        params: Conteúdo de ``params.yaml``.

    Returns:
        Pipeline ajustado, pronto para serialização.
    """
    hyperparams = params["train"]["hyperparams"].get(name, {})
    champion = build_model_pipeline(name, params["seed"], **hyperparams)
    champion.fit(datasets["x_train"], datasets["y_train"])
    return champion


def save_artifacts(
    champion: Pipeline,
    results: dict[str, dict[str, float]],
    metric: str,
    settings: Settings,
) -> None:
    """Grava o modelo campeão e a tabela de comparação.

    Args:
        champion: Pipeline ajustado.
        results: Métricas de todos os candidatos.
        metric: Métrica de ordenação da tabela.
        settings: Configuração com os diretórios de saída.
    """
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, settings.model_dir / "model.joblib")

    table = comparison_table(results, sort_by=metric)
    table.to_csv(settings.reports_dir / "comparison.csv")
    logger.info("Comparação:\n%s", table.to_string())


def write_best_run(
    name: str,
    run_id: str,
    metric: str,
    value: float,
    settings: Settings,
) -> None:
    """Registra o run vencedor para o estágio ``register`` consumir.

    Args:
        name: Nome do modelo campeão.
        run_id: Run correspondente no MLflow.
        metric: Métrica de seleção.
        value: Valor da métrica vencedora.
        settings: Configuração com o diretório de relatórios.
    """
    payload = {
        "model": name,
        "run_id": run_id,
        "selection_metric": metric,
        "selection_value": value,
    }
    (settings.reports_dir / "best_run.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def run_experiment(
    datasets: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Treina todos os candidatos configurados em ``params.yaml``.

    Args:
        datasets: Conjuntos de treino e teste.
        params: Conteúdo de ``params.yaml``.

    Returns:
        Tupla ``(métricas por modelo, run_id por modelo)``.
    """
    train_cfg = params["train"]
    return train_all_models(
        model_names=train_cfg["models"],
        hyperparams=train_cfg["hyperparams"],
        datasets=datasets,
        seed=params["seed"],
        folds=train_cfg["cv_folds"],
    )


def main() -> None:
    """Treina todos os modelos, registra no MLflow e salva o campeão."""
    setup_logging()
    settings = get_settings()
    params = load_params()

    setup_tracking()
    datasets = load_datasets(settings.processed_dir)
    results, run_ids = run_experiment(datasets, params)

    metric = params["train"]["selection_metric"]
    best = select_best(results, metric)

    save_artifacts(fit_champion(best, datasets, params), results, metric, settings)
    write_best_run(best, run_ids[best], metric, results[best][metric], settings)
    logger.info("Estágio 'train' concluído.")


if __name__ == "__main__":
    main()
