"""Estágio DVC ``evaluate`` — mede o campeão no conjunto de teste.

Produz ``reports/metrics.json`` (lido por ``dvc metrics show``) e as curvas
ROC e Precision-Recall em ``reports/figures/``.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sem display — obrigatório dentro do container
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from src.config import get_settings, load_params  # noqa: E402
from src.evaluation.metrics import cost_analysis, evaluate_model  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.pipelines.train import load_split  # noqa: E402

logger = logging.getLogger(__name__)


def save_curves(y_true: np.ndarray, y_proba: np.ndarray, output_dir: Path) -> None:
    """Gera e salva as curvas ROC e Precision-Recall.

    Args:
        y_true: Rótulos verdadeiros.
        y_proba: Probabilidades preditas.
        output_dir: Diretório de destino das figuras.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, display in (
        ("roc_curve", RocCurveDisplay),
        ("precision_recall_curve", PrecisionRecallDisplay),
    ):
        figure, axes = plt.subplots(figsize=(6, 5))
        display.from_predictions(y_true, y_proba, ax=axes)
        axes.set_title(name.replace("_", " ").title())
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=120)
        plt.close(figure)
    logger.info("Curvas salvas em %s", output_dir)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    eval_cfg: dict,
) -> dict[str, float]:
    """Junta as métricas estatísticas e a tradução em custo de negócio.

    Args:
        y_true: Rótulos verdadeiros.
        y_proba: Probabilidades preditas.
        eval_cfg: Seção ``evaluate`` do ``params.yaml``.

    Returns:
        Dicionário único com métricas e custo.
    """
    threshold = eval_cfg["threshold"]
    metrics = evaluate_model(y_true, y_proba, threshold=threshold)
    metrics.update(
        cost_analysis(
            y_true=y_true,
            y_pred=(y_proba >= threshold).astype(int),
            cost_false_positive=eval_cfg["cost_false_positive"],
            cost_false_negative=eval_cfg["cost_false_negative"],
        )
    )
    return metrics


def main() -> None:
    """Avalia o modelo campeão e grava métricas e figuras."""
    setup_logging()
    settings = get_settings()
    eval_cfg = load_params()["evaluate"]

    model = joblib.load(settings.model_dir / "model.joblib")
    x_test, y_test = load_split(settings.processed_dir / "test.parquet")

    y_true = y_test.to_numpy()
    proba = model.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_true, proba, eval_cfg)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_curves(y_true, proba, settings.reports_dir / "figures")

    logger.info("Métricas finais: %s", json.dumps(metrics, indent=2))
    logger.info("Estágio 'evaluate' concluído.")


if __name__ == "__main__":
    main()
