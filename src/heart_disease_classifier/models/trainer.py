"""
heart_disease_classifier.models.trainer
=========================================
Trains and manages the four classification models used in this project:
  - Logistic Regression (baseline linear model)
  - Decision Tree (interpretable rule-based model)
  - Random Forest (best-performing ensemble model)
  - Support Vector Machine (SVM) with probability estimates

Each model is wrapped inside :class:`ModelTrainer`, which stores fitted
instances and exposes a clean :meth:`~ModelTrainer.train_all` / predict API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Default model definitions
# --------------------------------------------------------------------------- #
def _default_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM": SVC(probability=True, random_state=42),
    }


# --------------------------------------------------------------------------- #
# ModelTrainer
# --------------------------------------------------------------------------- #
@dataclass
class ModelTrainer:
    """Train and store one or more sklearn classifiers.

    Parameters
    ----------
    models : dict[str, estimator], optional
        Mapping of model name → unfitted sklearn estimator.
        Defaults to all four models used in the original project.

    Attributes
    ----------
    fitted_models : dict[str, estimator]
        Populated with fitted estimators after :meth:`train_all` is called.
    """

    models: dict[str, Any] = field(default_factory=_default_models)
    fitted_models: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "ModelTrainer":
        """Fit every model in :attr:`models` on the training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Pre-processed feature matrix.
        y_train : pd.Series
            Binary target (0 = no disease, 1 = disease).

        Returns
        -------
        ModelTrainer
            *self* — allows method chaining.
        """
        for name, model in self.models.items():
            logger.info("Training %s …", name)
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
            logger.info("%s trained successfully.", name)
        return self

    def predict(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions for model *name*."""
        return self._get(name).predict(X)

    def predict_proba(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates for model *name* (positive class)."""
        return self._get(name).predict_proba(X)[:, 1]

    def predict_all(
        self, X: pd.DataFrame
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run ``predict`` and ``predict_proba`` for every fitted model.

        Returns
        -------
        dict
            ``{model_name: (y_pred, y_proba)}``.
        """
        self._assert_fitted()
        return {
            name: (self.predict(name, X), self.predict_proba(name, X))
            for name in self.fitted_models
        }

    def get_best_model(self, metric_scores: dict[str, float]) -> tuple[str, Any]:
        """Return ``(name, estimator)`` of the model with the highest score.

        Parameters
        ----------
        metric_scores : dict
            ``{model_name: score}`` — e.g. F1-scores from evaluation.
        """
        best_name = max(metric_scores, key=metric_scores.__getitem__)
        return best_name, self.fitted_models[best_name]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _get(self, name: str) -> Any:
        self._assert_fitted()
        if name not in self.fitted_models:
            raise KeyError(
                f"No fitted model named '{name}'. "
                f"Available: {list(self.fitted_models)}"
            )
        return self.fitted_models[name]

    def _assert_fitted(self) -> None:
        if not self.fitted_models:
            raise RuntimeError("Call train_all() before making predictions.")
