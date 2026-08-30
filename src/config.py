"""Configuração central do projeto.

Duas fontes, com responsabilidades distintas:

* ``Settings`` — variáveis de **ambiente** (caminhos, URIs, credenciais),
  carregadas do ``.env`` via *pydantic-settings*. Mudam entre máquinas.
* ``params.yaml`` — **hiperparâmetros** do pipeline, lidos por :func:`load_params`.
  São rastreados pelo DVC e fazem parte da reprodutibilidade do experimento.

O esquema de features é constante do domínio e vive aqui como fonte única
de verdade — nenhum outro módulo deve redeclarar nomes de colunas.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"

# ── Esquema do dataset Online Shoppers Purchasing Intention ──────────────────

NUMERIC_FEATURES: list[str] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# OperatingSystems, Browser, Region e TrafficType chegam como int64 no CSV,
# mas são *códigos* categóricos — a distância entre o browser 2 e o 13 não
# significa nada. São convertidos para str antes do One-Hot Encoding.
CATEGORICAL_FEATURES: list[str] = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# Colunas numéricas na origem que precisam virar string antes do encoder.
CODE_LIKE_FEATURES: list[str] = [
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Weekend",
]

TARGET: str = "Revenue"


class Settings(BaseSettings):
    """Variáveis de ambiente do projeto, com defaults seguros para dev local."""

    # protected_namespaces=() libera nomes como `model_dir`, que o pydantic
    # reservaria por começarem com `model_`.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "purchase-intent"
    mlflow_registered_model: str = "purchase-intent-classifier"
    mlflow_model_alias: str = "champion"

    data_url: str = (
        "https://archive.ics.uci.edu/static/public/468/"
        "online+shoppers+purchasing+intention+dataset.zip"
    )
    raw_data_path: Path = Path("data/raw/online_shoppers_intention.csv")
    processed_dir: Path = Path("data/processed")

    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Redes com inspeção TLS (proxy corporativo, antivírus) reassinam o
    # HTTPS com uma CA própria, que o certifi desconhece. Apontar para um
    # bundle contendo essa raiz faz o download do dataset voltar a funcionar.
    ca_bundle: Path | None = None

    @property
    def model_uri(self) -> str:
        """URI do modelo campeão no MLflow Model Registry."""
        return f"models:/{self.mlflow_registered_model}@{self.mlflow_model_alias}"

    @property
    def request_verify(self) -> str | bool:
        """Valor de ``verify`` para o ``requests``: bundle próprio ou padrão."""
        return str(self.ca_bundle) if self.ca_bundle else True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devolve a instância única de :class:`Settings` (cacheada)."""
    return Settings()


@lru_cache(maxsize=1)
def load_params(path: Path | None = None) -> dict[str, Any]:
    """Carrega ``params.yaml`` como dicionário.

    Args:
        path: Caminho alternativo do arquivo. Usa ``PARAMS_PATH`` se omitido.

    Returns:
        Conteúdo do YAML já desserializado.
    """
    target = path or PARAMS_PATH
    with target.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
