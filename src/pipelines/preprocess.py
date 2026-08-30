"""Estágio DVC ``preprocess`` — valida, tipa e divide o dataset.

Salva treino e teste como Parquet **com as features ainda cruas**. A
padronização e o One-Hot acontecem dentro do pipeline do modelo, ajustado
só no treino — assim não há como o teste vazar para o pré-processador.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import TARGET, get_settings, load_params
from src.data.loader import get_features_target, load_raw, validate_schema
from src.data.preprocessing import split_data
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def save_split(features: pd.DataFrame, target: pd.Series, path: Path) -> None:
    """Grava um par (features, alvo) em um único arquivo Parquet.

    Args:
        features: Matriz de features.
        target: Vetor alvo.
        path: Caminho do arquivo de saída.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    features.assign(**{TARGET: target}).to_parquet(path, index=False)
    logger.info("Salvo %s (%d linhas)", path, len(features))


def main() -> None:
    """Executa a validação e a divisão treino/teste."""
    setup_logging()
    settings = get_settings()
    params = load_params()

    raw = load_raw(settings.raw_data_path)
    features, target = get_features_target(raw)
    validate_schema(features)

    x_train, x_test, y_train, y_test = split_data(
        features=features,
        target=target,
        test_size=params["split"]["test_size"],
        seed=params["seed"],
        stratify=params["split"]["stratify"],
    )

    save_split(x_train, y_train, settings.processed_dir / "train.parquet")
    save_split(x_test, y_test, settings.processed_dir / "test.parquet")
    logger.info("Estágio 'preprocess' concluído.")


if __name__ == "__main__":
    main()
