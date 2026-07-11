"""Métricas e gráficos para classificação binária."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_auc(metric, y_true: np.ndarray, y_probability: np.ndarray) -> float | None:
    try:
        return float(metric(y_true, y_probability))
    except ValueError:
        return None


def evaluate_binary(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calcula métricas de classificação e contagens da matriz de confusão."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_probability.shape[0]:
        raise ValueError("y_true e y_probability devem ter o mesmo número de exemplos.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre 0 e 1.")

    y_pred = (y_probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    false_negative_rate = float(fn / (fn + tp)) if fn + tp else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(roc_auc_score, y_true, y_probability),
        "pr_auc": _safe_auc(average_precision_score, y_true, y_probability),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(y_true.shape[0]),
    }


def threshold_maximizing_f1(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> float:
    """Seleciona na validação o limiar que maximiza F1."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_probability)
    if thresholds.size == 0:
        return 0.5

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def save_metrics(metrics: dict[str, Any], destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_evaluation_plots(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    output_dir: Path,
    prefix: str,
) -> None:
    """Salva matriz de confusão, curva ROC e curva Precision-Recall."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_probability) >= threshold).astype(np.int64)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Normal", "Ataque"])
    ax.set_yticks([0, 1], labels=["Normal", "Ataque"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão - limiar {threshold:.3f}")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_probability)
        auc_value = roc_auc_score(y_true, y_probability)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
        ax.set_xlabel("Taxa de falsos positivos")
        ax.set_ylabel("Recall / TPR")
        ax.set_title("Curva ROC")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=160)
        plt.close(fig)
    except ValueError:
        pass

    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_precision_recall_curve.png", dpi=160)
    plt.close(fig)
