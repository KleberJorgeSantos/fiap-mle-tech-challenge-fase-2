"""Leitura, tipagem e validação do dataset bruto.

O CSV da UCI vem quase limpo (zero nulos), mas tem duas armadilhas:

1. ``OperatingSystems``, ``Browser``, ``Region`` e ``TrafficType`` são
   inteiros que representam **categorias**, não quantidades.
2. ``Revenue`` e ``Weekend`` são booleanos — o alvo precisa virar 0/1 e a
   feature precisa virar string para o One-Hot Encoder.
"""

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

from src.config import (
    CATEGORICAL_FEATURES,
    CODE_LIKE_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path | str) -> pd.DataFrame:
    """Lê o CSV bruto do disco.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        DataFrame com as 18 colunas originais.
    """
    logger.info("Carregando dataset de %s", path)
    df = pd.read_csv(path)
    logger.info("Carregadas %d linhas e %d colunas", len(df), len(df.columns))
    return df


def cast_code_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas de código (int/bool) para string.

    Sem isso o ``StandardScaler`` trataria "Browser 13" como treze vezes
    "Browser 1", inventando uma ordem que não existe no domínio.

    Args:
        df: DataFrame de entrada.

    Returns:
        Cópia do DataFrame com as colunas de código como ``str``.
    """
    df = df.copy()
    for column in CODE_LIKE_FEATURES:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features do alvo binário.

    Args:
        df: DataFrame bruto vindo de :func:`load_raw`.

    Returns:
        Tupla ``(X, y)`` com as 17 features e o alvo ``Revenue`` em 0/1.
    """
    df = cast_code_like_columns(df)

    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = df[TARGET].astype(bool).astype(int)

    logger.info(
        "Features: %s | taxa de conversão: %.2f%%",
        features.shape,
        target.mean() * 100,
    )
    return features, target


_SCHEMA = DataFrameSchema(
    {
        "Administrative": Column(float, pa.Check.ge(0), coerce=True),
        "ProductRelated": Column(float, pa.Check.ge(0), coerce=True),
        "ProductRelated_Duration": Column(float, pa.Check.ge(0), coerce=True),
        "BounceRates": Column(float, pa.Check.in_range(0, 1), coerce=True),
        "ExitRates": Column(float, pa.Check.in_range(0, 1), coerce=True),
        "PageValues": Column(float, pa.Check.ge(0), coerce=True),
        "SpecialDay": Column(float, pa.Check.in_range(0, 1), coerce=True),
        "Month": Column(str, coerce=True),
        "VisitorType": Column(str, coerce=True),
        "Weekend": Column(str, coerce=True),
    },
    strict=False,
)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Valida tipos e faixas das colunas críticas com Pandera.

    Usado tanto no pipeline de treino quanto no ``/predict`` da API, onde
    um ``SchemaError`` vira HTTP 422 com as ``failure_cases``.

    Args:
        df: DataFrame de features já tipado.

    Returns:
        O DataFrame validado (com coerção aplicada).

    Raises:
        pandera.errors.SchemaError: Se alguma coluna violar o contrato.
    """
    return _SCHEMA.validate(df)
