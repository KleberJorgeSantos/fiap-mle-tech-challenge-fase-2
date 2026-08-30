"""Estágio DVC ``download`` — obtém o CSV bruto da UCI."""

import logging

from src.config import get_settings, load_params
from src.data.download import download_dataset
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Baixa o dataset para o caminho configurado."""
    setup_logging()
    settings = get_settings()
    params = load_params()

    download_dataset(
        url=params["data"]["url"],
        destination=settings.raw_data_path,
        csv_name=params["data"]["csv_name"],
        verify=settings.request_verify,
    )
    logger.info("Estágio 'download' concluído.")


if __name__ == "__main__":
    main()
