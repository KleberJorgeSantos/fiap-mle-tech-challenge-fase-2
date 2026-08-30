"""Estágio DVC ``register`` — promove o campeão no MLflow Model Registry."""

import json
import logging

from src.config import get_settings, load_params
from src.logging_config import setup_logging
from src.tracking.mlflow_utils import setup_tracking
from src.tracking.registry import register_best_model

logger = logging.getLogger(__name__)


def main() -> None:
    """Registra o melhor run e aponta o alias ``@champion`` para ele."""
    setup_logging()
    settings = get_settings()
    load_params()
    setup_tracking()

    best = json.loads((settings.reports_dir / "best_run.json").read_text(encoding="utf-8"))
    summary = register_best_model(
        run_id=best["run_id"],
        metric_name=best["selection_metric"],
        metric_value=best["selection_value"],
    )

    (settings.reports_dir / "registered_model.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Modelo disponível em %s", summary["model_uri"])
    logger.info("Estágio 'register' concluído.")


if __name__ == "__main__":
    main()
