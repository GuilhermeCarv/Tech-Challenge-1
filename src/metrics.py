"""Métricas de avaliação para classificação e regressão.

Este módulo centraliza as funções de cálculo de métricas para reutilização por
baseline, MLP e outros componentes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)


def calcular_metricas_classificacao(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    resultados = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            resultados['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            resultados['roc_auc'] = float('nan')
    else:
        resultados['roc_auc'] = float('nan')
    return resultados


def calcular_metricas_regressao(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metricas: dict[str, float] = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float('nan'),
    }
    return metricas
