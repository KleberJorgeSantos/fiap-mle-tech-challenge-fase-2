"""Construção do pré-processador e divisão treino/teste.

O ``ColumnTransformer`` é ajustado **apenas no conjunto de treino** e depois
aplicado ao teste — é o que impede vazamento das estatísticas (média, desvio,
categorias vistas) do conjunto de avaliação para dentro do modelo.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` de features numéricas e categóricas.

    Returns:
        Transformador não ajustado: mediana + padronização nas 10 numéricas,
        moda + One-Hot nas 7 categóricas (~75 colunas na saída).
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide os dados em treino e teste de forma estratificada.

    Args:
        features: Matriz de features.
        target: Vetor alvo binário.
        test_size: Fração reservada para teste.
        seed: Semente para reprodutibilidade.
        stratify: Se ``True``, preserva a proporção de classes nos dois lados.

    Returns:
        Tupla ``(X_train, X_test, y_train, y_test)``.
    """
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target if stratify else None,
        random_state=seed,
    )
    logger.info("Divisão → treino=%d | teste=%d", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Recupera os nomes das colunas geradas pelo pré-processador ajustado.

    Args:
        preprocessor: ``ColumnTransformer`` já ajustado.

    Returns:
        Lista de nomes na mesma ordem das colunas transformadas.
    """
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    return NUMERIC_FEATURES + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))


def to_dense(array: object) -> np.ndarray:
    """Normaliza a saída do transformador para um ``ndarray`` denso.

    Args:
        array: Matriz densa ou esparsa vinda do ``ColumnTransformer``.

    Returns:
        Array NumPy denso.
    """
    if hasattr(array, "toarray"):
        return array.toarray()
    return np.asarray(array)
