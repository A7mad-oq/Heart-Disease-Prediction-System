"""
heart_disease_classifier.preprocessing.pipeline
================================================
Applies all preprocessing steps to the raw Heart Disease dataset:

1. Impute missing values (median for numeric, mode for categorical).
2. Scale numeric features with StandardScaler.
3. Discretise BloodPressure into clinical categories, one-hot-encode them.
4. Split into train / test sets.

A fitted :class:`Preprocessor` can be persisted with ``joblib`` so the
same scaler is reused at inference time without re-fitting on test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NUMERIC_COLS: list[str] = [
    "Age",
    "BloodPressure",
    "Cholesterol",
    "HeartRate",
    "QuantumPatternFeature",
]
CATEGORICAL_COLS: list[str] = ["Gender"]
SCALE_COLS: list[str] = ["Age", "Cholesterol", "HeartRate", "QuantumPatternFeature"]
TARGET_COL: str = "HeartDisease"

BP_BINS: list[float] = [0, 90, 120, 140, 200]
BP_LABELS: list[str] = ["Low", "Normal", "High", "VeryHigh"]


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
@dataclass
class Preprocessor:
    """Stateful preprocessor that can be fitted on training data and applied
    consistently to new (unseen) data.

    Parameters
    ----------
    test_size : float
        Proportion of the dataset reserved for the test set. Default ``0.25``.
    random_state : int
        Random seed for reproducibility. Default ``42``.
    """

    test_size: float = 0.25
    random_state: int = 42

    # Private state set after fit()
    _scaler: Optional[StandardScaler] = field(default=None, repr=False)
    _bp_dummy_columns: list[str] = field(default_factory=list, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fit the preprocessor on *df* and return train/test splits.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset as returned by :func:`~heart_disease_classifier.data.loader.load_dataset`.

        Returns
        -------
        X_train, X_test : pd.DataFrame
        y_train, y_test : pd.Series
        """
        df = df.copy()

        df = self._impute(df)
        df, self._scaler = self._scale(df)
        df, self._bp_dummy_columns = self._encode_blood_pressure(df)

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self._is_fitted = True
        logger.info(
            "Preprocessing complete — train: %d rows, test: %d rows",
            len(X_train),
            len(X_test),
        )
        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to new data (no re-fitting).

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with the same schema as the training set. The
            ``HeartDisease`` target column is optional.

        Returns
        -------
        pd.DataFrame
            Preprocessed feature matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        df = df.copy()
        has_target = TARGET_COL in df.columns
        df = self._impute(df)

        # Scale using already-fitted scaler
        df[SCALE_COLS] = self._scaler.transform(df[SCALE_COLS])

        # Encode blood pressure
        df, _ = self._encode_blood_pressure(df)

        # Align columns to what the model expects
        feature_cols = [c for c in df.columns if c != TARGET_COL]
        for col in self._bp_dummy_columns:
            if col not in df.columns:
                df[col] = False

        if has_target:
            df = df.drop(columns=[TARGET_COL])

        return df[self._bp_dummy_columns + [c for c in feature_cols if c not in self._bp_dummy_columns and c != "BloodPressure"]]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values: median for numerics, mode for categoricals."""
        for col in NUMERIC_COLS:
            if col in df.columns and df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)
                logger.debug("Imputed '%s' with median=%.4f", col, median)

        for col in CATEGORICAL_COLS:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.debug("Imputed '%s' with mode=%s", col, mode_val)

        return df

    @staticmethod
    def _scale(
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """Standardise numeric feature columns (zero mean, unit variance)."""
        if scaler is None:
            scaler = StandardScaler()
            df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
        else:
            df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
        return df, scaler

    @staticmethod
    def _encode_blood_pressure(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Bin BloodPressure into clinical categories, then one-hot-encode."""
        df["BP_Category"] = pd.cut(
            df["BloodPressure"], bins=BP_BINS, labels=BP_LABELS
        )
        df = pd.get_dummies(df, columns=["BP_Category"], prefix="BP")
        df = df.drop(columns=["BloodPressure"], errors="ignore")

        bp_cols = [c for c in df.columns if c.startswith("BP_")]
        return df, bp_cols
