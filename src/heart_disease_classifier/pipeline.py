"""
heart_disease_classifier.pipeline
===================================
High-level orchestrator that wires together data loading, preprocessing,
model training, and evaluation in a single :class:`HeartDiseasePipeline`.

Typical usage
-------------
>>> from heart_disease_classifier import HeartDiseasePipeline
>>> pipe = HeartDiseasePipeline(data_path="data/raw/heart_prediction_quantum.csv")
>>> pipe.run()
>>> print(pipe.results_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from heart_disease_classifier.data.loader import load_dataset
from heart_disease_classifier.evaluation.metrics import (
    evaluate_all,
    plot_confusion_matrix,
    plot_roc_curves,
)
from heart_disease_classifier.models.trainer import ModelTrainer
from heart_disease_classifier.preprocessing.pipeline import Preprocessor
from heart_disease_classifier.utils.persistence import save_artifact

logger = logging.getLogger(__name__)


class HeartDiseasePipeline:
    """End-to-end pipeline for heart disease prediction.

    Parameters
    ----------
    data_path : str or Path
        CSV file with the raw dataset.
    output_dir : str or Path, optional
        Directory where results CSV, plots, and model artefacts are saved.
        Defaults to ``outputs/``.
    test_size : float
        Fraction of data used for evaluation. Default ``0.25``.
    random_state : int
        Global random seed. Default ``42``.

    Attributes
    ----------
    results_df : pd.DataFrame or None
        Metric comparison table populated after :meth:`run`.
    best_model_name : str or None
        Name of the best-performing model (highest F1-Score) after :meth:`run`.
    """

    def __init__(
        self,
        data_path: str | Path,
        output_dir: str | Path = "outputs",
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor: Optional[Preprocessor] = None
        self.trainer: Optional[ModelTrainer] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.best_model_name: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, save_outputs: bool = True) -> "HeartDiseasePipeline":
        """Execute all pipeline stages.

        Stages
        ------
        1. Load raw data.
        2. Preprocess (impute → scale → encode → split).
        3. Train all four classifiers.
        4. Evaluate and rank models.
        5. Optionally save results, plots, and the best model.

        Parameters
        ----------
        save_outputs : bool
            Whether to persist results to :attr:`output_dir`. Default ``True``.

        Returns
        -------
        HeartDiseasePipeline
            *self* — supports method chaining.
        """
        logger.info("=" * 60)
        logger.info("Heart Disease Classifier — Pipeline Start")
        logger.info("=" * 60)

        # ── 1. Load ──────────────────────────────────────────────────── #
        df = load_dataset(self.data_path)

        # ── 2. Preprocess ─────────────────────────────────────────────── #
        self.preprocessor = Preprocessor(
            test_size=self.test_size,
            random_state=self.random_state,
        )
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df)

        # ── 3. Train ──────────────────────────────────────────────────── #
        self.trainer = ModelTrainer()
        self.trainer.train_all(X_train, y_train)

        # ── 4. Evaluate ───────────────────────────────────────────────── #
        predictions = self.trainer.predict_all(X_test)
        self.results_df = evaluate_all(predictions, y_test)

        # Identify best model by F1-Score
        best_idx = self.results_df["F1-Score"].idxmax()
        self.best_model_name = self.results_df.loc[best_idx, "Algorithm"]
        best_row = self.results_df.loc[best_idx]

        logger.info("-" * 60)
        logger.info("Best model: %s", self.best_model_name)
        logger.info(
            "  Accuracy=%.4f  F1=%.4f  AUC=%.4f",
            best_row["Accuracy"],
            best_row["F1-Score"],
            best_row["AUC"],
        )
        logger.info("-" * 60)

        # ── 5. Save outputs ───────────────────────────────────────────── #
        if save_outputs:
            self._save_outputs(predictions, y_test)

        logger.info("Pipeline complete.")
        return self

    def print_summary(self) -> None:
        """Pretty-print the model comparison table and conclusion."""
        if self.results_df is None:
            print("Pipeline has not been run yet. Call run() first.")
            return

        print("\n" + "=" * 60)
        print("  Heart Disease Prediction — Model Comparison")
        print("=" * 60)
        print(self.results_df.round(4).to_string(index=False))
        print("-" * 60)
        print(f"  Best Model → {self.best_model_name}")
        best = self.results_df[
            self.results_df["Algorithm"] == self.best_model_name
        ].iloc[0]
        print(
            f"  Accuracy: {best['Accuracy']:.4f} | "
            f"F1-Score: {best['F1-Score']:.4f} | "
            f"AUC: {best['AUC']:.4f}"
        )
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _save_outputs(self, predictions: dict, y_test) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results CSV
        results_path = self.output_dir / "model_comparison.csv"
        self.results_df.to_csv(results_path, index=False)
        logger.info("Results saved → %s", results_path)

        # Confusion matrix for best model
        best_pred, _ = predictions[self.best_model_name]
        plot_confusion_matrix(
            y_test,
            best_pred,
            model_name=self.best_model_name,
            output_path=self.output_dir / "confusion_matrix.png",
        )

        # ROC curves for all models
        plot_roc_curves(
            predictions,
            y_test,
            output_path=self.output_dir / "roc_curves.png",
        )

        # Best model artefact
        _, best_estimator = self.trainer.get_best_model(
            dict(zip(self.results_df["Algorithm"], self.results_df["F1-Score"]))
        )
        save_artifact(
            best_estimator,
            self.output_dir / "best_model.joblib",
        )
        save_artifact(
            self.preprocessor,
            self.output_dir / "preprocessor.joblib",
        )
