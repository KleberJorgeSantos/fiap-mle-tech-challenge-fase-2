"""Download e extração do dataset bruto da UCI.

Primeiro estágio do pipeline DVC. A saída deste módulo é o único arquivo
de dados versionado pelo DVC — o hash do CSV entra no ``dvc.lock``, o que
garante que todo mundo treina exatamente sobre os mesmos 12.330 registros.
"""

import io
import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 120


def download_dataset(
    url: str,
    destination: Path,
    csv_name: str,
    verify: str | bool = True,
) -> Path:
    """Baixa o zip da UCI e extrai o CSV para ``destination``.

    Args:
        url: URL do arquivo ``.zip`` publicado pela UCI.
        destination: Caminho final do CSV extraído.
        csv_name: Nome do arquivo CSV dentro do zip.
        verify: Bundle de CAs a confiar, ou ``True`` para o padrão do
            ``certifi``. Redes com inspeção TLS precisam do bundle próprio.

    Returns:
        O caminho do CSV extraído.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando dataset de %s", url)

    response = requests.get(url, timeout=_TIMEOUT_SECONDS, verify=verify)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        destination.write_bytes(archive.read(csv_name))

    size_kb = destination.stat().st_size / 1024
    logger.info("CSV salvo em %s (%.1f KB)", destination, size_kb)
    return destination
