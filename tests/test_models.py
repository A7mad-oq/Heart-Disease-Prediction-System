"""
tests/test_models.py
=====================
Unit tests for ModelTrainer.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart_disease_classifier.models.trainer import ModelTrainer
from heart_disease_classifier.preprocessing.pipeline import Preprocessor


def _synthetic_splits(n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Age":                   rng.integers(20, 80, n).astype(float),
            "Gender":                rng.integers(0, 2, n).astype(float),
            "BloodPressure":         rng.integers(60, 180, n).astype(float),
            "Cholesterol":           rng.integers(150, 300, n).astype(float),
            "HeartRate":             rng.integers(50, 130, n).astype(float),
            "QuantumPatternFeature": rng.uniform(5, 15, n),
            "HeartDisease":          rng.integers(0, 2, n),
        }
    )
    prep = Preprocessor(test_size=0.25, random_state=seed)
    return prep.fit_transform(df)


class TestModelTrainer:
    def setup_method(self):
        self.X_train, self.X_test, self.y_train, self.y_test = _synthetic_splits()

    def test_train_all_populates_fitted_models(self):
        trainer = ModelTrainer()
        trainer.train_all(self.X_train, self.y_train)
        assert len(trainer.fitted_models) == len(trainer.models)

    def test_predict_returns_binary_array(self):
        trainer = ModelTrainer()
        trainer.train_all(self.X_train, self.y_train)
        for name in trainer.fitted_models:
            preds = trainer.predict(name, self.X_test)
            assert set(preds).issubset({0, 1}), f"{name}: predictions must be 0 or 1"

    def test_predict_proba_range(self):
        trainer = ModelTrainer()
        trainer.train_all(self.X_train, self.y_train)
        for name in trainer.fitted_models:
            probas = trainer.predict_proba(name, self.X_test)
            assert probas.min() >= 0.0
            assert probas.max() <= 1.0

    def test_predict_all_covers_all_models(self):
        trainer = ModelTrainer()
        trainer.train_all(self.X_train, self.y_train)
        preds = trainer.predict_all(self.X_test)
        assert set(preds.keys()) == set(trainer.fitted_models.keys())

    def test_raises_before_training(self):
        trainer = ModelTrainer()
        with pytest.raises(RuntimeError, match="train_all"):
            trainer.predict_all(self.X_test)

    def test_unknown_model_name_raises_key_error(self):
        trainer = ModelTrainer()
        trainer.train_all(self.X_train, self.y_train)
        with pytest.raises(KeyError):
            trainer.predict("NonExistentModel", self.X_test)
