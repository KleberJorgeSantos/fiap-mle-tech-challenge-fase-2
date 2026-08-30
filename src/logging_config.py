"""Configuração de logging compartilhada por todos os entrypoints.

O projeto não usa ``print()`` em lugar nenhum: todo output passa por
``logging.getLogger(__name__)``, o que permite controlar verbosidade via
a variável de ambiente ``LOG_LEVEL`` sem tocar no código.
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configura o logger raiz para escrever em stdout.

    Args:
        level: Nível de log (``DEBUG``, ``INFO``, ...). Usa ``LOG_LEVEL``
            do ``.env`` quando omitido.
    """
    from src.config import get_settings

    resolved = level or get_settings().log_level

    # O console do Windows usa cp1252 por padrão e quebra em acentos e setas.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=resolved.upper(),
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # MLflow e Alembic são verbosos demais em INFO: o Alembic despeja todas
    # as migrações do SQLite na primeira execução do pipeline.
    for noisy in ("mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
