"""
heart_disease_classifier.evaluation.metrics
============================================
Computes, stores, and visualises classification metrics for all trained models.

Metrics reported
----------------
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- AUC (Area Under the ROC Curve)

The module also generates:
- A formatted comparison table (``pd.DataFrame``)
- A confusion matrix heatmap (``matplotlib`` figure)
- ROC curve overlay for all models (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]


# --------------------------------------------------------------------------- #
# Core evaluation functions
# --------------------------------------------------------------------------- #
def evaluate_model(
    name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float | str]:
    """Compute all metrics for a single model.

    Parameters
    ----------
    name : str
        Model identifier used in the output row.
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Binary predictions.
    y_proba : array-like
        Positive-class probability estimates (for AUC).

    Returns
    -------
    dict
        Metric values keyed by column name.
    """
    return {
        "Algorithm": name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
        "AUC":       roc_auc_score(y_true, y_proba),
    }


def evaluate_all(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Evaluate every model and return a comparison DataFrame.

    Parameters
    ----------
    predictions : dict
        ``{model_name: (y_pred, y_proba)}`` as returned by
        :meth:`~heart_disease_classifier.models.trainer.ModelTrainer.predict_all`.
    y_true : array-like
        Ground-truth labels for the test set.

    Returns
    -------
    pd.DataFrame
        One row per model, columns: Algorithm, Accuracy, Precision,
        Recall, F1-Score, AUC.
    """
    rows = [
        evaluate_model(name, y_true, y_pred, y_proba)
        for name, (y_pred, y_proba) in predictions.items()
    ]
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    logger.info("Evaluation complete:\n%s", df.round(4).to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Used in the plot title.
    output_path : Path or str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Confusion matrix saved to %s", output_path)

    return fig


def plot_roc_curves(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray | pd.Series,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Overlay ROC curves for all models on a single axes.

    Parameters
    ----------
    predictions : dict
        ``{model_name: (y_pred, y_proba)}``.
    y_true : array-like
        Ground-truth labels.
    output_path : Path or str, optional
        If provided, saves the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, (_, y_proba) in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("ROC curves saved to %s", output_path)

    return fig
