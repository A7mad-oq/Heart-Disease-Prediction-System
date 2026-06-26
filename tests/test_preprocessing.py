"""
tests/test_preprocessing.py
============================
Unit tests for the Preprocessor class.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart_disease_classifier.preprocessing.pipeline import Preprocessor


def _make_sample_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Generate a minimal synthetic dataset for testing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Age":                  rng.integers(20, 80, n).astype(float),
            "Gender":               rng.integers(0, 2, n).astype(float),
            "BloodPressure":        rng.integers(60, 180, n).astype(float),
            "Cholesterol":          rng.integers(150, 300, n).astype(float),
            "HeartRate":            rng.integers(50, 130, n).astype(float),
            "QuantumPatternFeature": rng.uniform(5, 15, n),
            "HeartDisease":         rng.integers(0, 2, n),
        }
    )
    return df


class TestPreprocessor:
    def test_fit_transform_returns_four_splits(self):
        df = _make_sample_df()
        prep = Preprocessor(test_size=0.25, random_state=42)
        result = prep.fit_transform(df)
        assert len(result) == 4, "Should return (X_train, X_test, y_train, y_test)"

    def test_split_proportions(self):
        n = 200
        df = _make_sample_df(n=n)
        prep = Preprocessor(test_size=0.25, random_state=42)
        X_train, X_test, y_train, y_test = prep.fit_transform(df)
        assert len(X_test) == pytest.approx(n * 0.25, abs=1)
        assert len(X_train) == pytest.approx(n * 0.75, abs=1)

    def test_blood_pressure_column_removed(self):
        df = _make_sample_df()
        prep = Preprocessor()
        X_train, X_test, _, _ = prep.fit_transform(df)
        assert "BloodPressure" not in X_train.columns
        assert "BloodPressure" not in X_test.columns

    def test_bp_dummy_columns_present(self):
        df = _make_sample_df(n=300)
        prep = Preprocessor()
        X_train, _, _, _ = prep.fit_transform(df)
        bp_cols = [c for c in X_train.columns if c.startswith("BP_")]
        assert len(bp_cols) > 0, "One-hot encoded BP columns should be present"

    def test_no_missing_values_after_transform(self):
        df = _make_sample_df()
        # Introduce some NaNs
        df.loc[0, "Age"] = np.nan
        df.loc[1, "Gender"] = np.nan
        df.loc[2, "Cholesterol"] = np.nan

        prep = Preprocessor()
        X_train, X_test, _, _ = prep.fit_transform(df)
        assert not X_train.isna().any().any(), "Training set should have no NaNs"
        assert not X_test.isna().any().any(), "Test set should have no NaNs"

    def test_transform_raises_if_not_fitted(self):
        df = _make_sample_df()
        prep = Preprocessor()
        with pytest.raises(RuntimeError, match="fit_transform"):
            prep.transform(df)

    def test_reproducibility(self):
        df = _make_sample_df()
        p1 = Preprocessor(random_state=99)
        p2 = Preprocessor(random_state=99)
        _, X_test1, _, y_test1 = p1.fit_transform(df)
        _, X_test2, _, y_test2 = p2.fit_transform(df)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_test1, y_test2)
