"""Fixtures compartilhadas pelos testes."""

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

_RNG_SEED = 7


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """DataFrame sintético com o mesmo esquema do CSV da UCI.

    Evita depender do download da UCI para rodar a suíte — os testes
    precisam passar offline, em CI e em qualquer clone limpo.
    """
    rng = np.random.default_rng(_RNG_SEED)
    size = 200
    data = {
        "Administrative": rng.integers(0, 10, size),
        "Administrative_Duration": rng.uniform(0, 500, size),
        "Informational": rng.integers(0, 5, size),
        "Informational_Duration": rng.uniform(0, 300, size),
        "ProductRelated": rng.integers(1, 100, size),
        "ProductRelated_Duration": rng.uniform(0, 3000, size),
        "BounceRates": rng.uniform(0, 0.2, size),
        "ExitRates": rng.uniform(0, 0.2, size),
        "PageValues": rng.uniform(0, 50, size),
        "SpecialDay": rng.choice([0.0, 0.4, 0.8, 1.0], size),
        "Month": rng.choice(["Feb", "Mar", "May", "Nov", "Dec"], size),
        "OperatingSystems": rng.integers(1, 8, size),
        "Browser": rng.integers(1, 13, size),
        "Region": rng.integers(1, 9, size),
        "TrafficType": rng.integers(1, 20, size),
        "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor", "Other"], size),
        "Weekend": rng.choice([True, False], size),
        TARGET: rng.choice([True, False], size, p=[0.3, 0.7]),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def features_target(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Par ``(X, y)`` derivado do DataFrame sintético."""
    from src.data.loader import get_features_target

    return get_features_target(raw_frame)


@pytest.fixture(scope="session")
def expected_columns() -> list[str]:
    """Colunas de feature esperadas, na ordem canônica."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
