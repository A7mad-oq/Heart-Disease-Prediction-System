"""
tests/test_evaluation.py
=========================
Unit tests for the evaluation module.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart_disease_classifier.evaluation.metrics import (
    METRIC_COLUMNS,
    evaluate_all,
    evaluate_model,
)


def _dummy_predictions(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    preds = {
        "Model A": (rng.integers(0, 2, n), rng.uniform(0, 1, n)),
        "Model B": (rng.integers(0, 2, n), rng.uniform(0, 1, n)),
    }
    return y_true, preds


class TestEvaluateModel:
    def test_returns_all_metric_keys(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        row = evaluate_model("Test", y_true, y_pred, y_proba)
        for col in METRIC_COLUMNS:
            assert col in row, f"Missing key: {col}"

    def test_perfect_predictions_score_one(self):
        y = np.array([0, 1, 0, 1, 1])
        row = evaluate_model("Perfect", y, y, y.astype(float))
        assert row["Accuracy"] == 1.0
        assert row["F1-Score"] == 1.0


class TestEvaluateAll:
    def test_returns_dataframe_with_correct_shape(self):
        y_true, preds = _dummy_predictions()
        df = evaluate_all(preds, y_true)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (len(preds), len(METRIC_COLUMNS))

    def test_column_names_correct(self):
        y_true, preds = _dummy_predictions()
        df = evaluate_all(preds, y_true)
        assert list(df.columns) == METRIC_COLUMNS

    def test_metrics_in_valid_range(self):
        y_true, preds = _dummy_predictions()
        df = evaluate_all(preds, y_true)
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
            assert df[col].between(0.0, 1.0).all(), f"{col} out of [0, 1]"
