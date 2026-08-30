"""Métricas de avaliação e análise de custo de negócio.

A seleção do modelo campeão usa **PR-AUC**, não accuracy: com 15,5% de
conversão, um classificador que responde sempre "não compra" já acerta
84,5% das sessões sem gerar nenhum valor.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula as métricas de classificação binária.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_proba: Probabilidade predita da classe positiva.
        threshold: Limiar de decisão.

    Returns:
        Dicionário com ``roc_auc``, ``pr_auc``, ``f1``, ``precision``,
        ``recall`` e ``accuracy``.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_false_positive: float,
    cost_false_negative: float,
) -> dict[str, float]:
    """Traduz os erros do modelo em custo monetário.

    Um falso negativo (sessão que ia converter e não recebeu ação) custa a
    margem da venda perdida; um falso positivo custa apenas o cupom gasto
    à toa. A assimetria é o que justifica otimizar recall/PR-AUC.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_pred: Classes preditas (0/1).
        cost_false_positive: Custo unitário de um falso positivo.
        cost_false_negative: Custo unitário de um falso negativo.

    Returns:
        Dicionário com contagens de FP/FN e o custo total.
    """
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
    total = false_positives * cost_false_positive + false_negatives * cost_false_negative
    logger.info("FP=%d | FN=%d | custo total=R$ %.2f", false_positives, false_negatives, total)
    return {
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "business_cost": float(total),
    }


def comparison_table(results: dict[str, dict[str, float]], sort_by: str) -> pd.DataFrame:
    """Consolida as métricas de todos os modelos em uma tabela ordenada.

    Args:
        results: Mapa ``nome_do_modelo -> métricas``.
        sort_by: Métrica usada para ordenar (decrescente).

    Returns:
        DataFrame indexado por modelo, do melhor para o pior.
    """
    table = pd.DataFrame(
        [{"model": name, **metrics} for name, metrics in results.items()]
    ).set_index("model")
    return table.sort_values(sort_by, ascending=False)
