"""
heart_disease_classifier.data.loader
=====================================
Responsible for loading the Heart Disease dataset from disk.
Provides basic data-quality checks after loading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns coming from the raw CSV
REQUIRED_COLUMNS = {
    "Age",
    "Gender",
    "BloodPressure",
    "Cholesterol",
    "HeartRate",
    "QuantumPatternFeature",
    "HeartDisease",
}


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load the Heart Disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file (e.g. ``data/raw/heart_prediction_quantum.csv``).

    Returns
    -------
    pd.DataFrame
        Raw dataset — no preprocessing applied.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are missing from the file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info("Loading dataset from %s", filepath)
    df = pd.read_csv(filepath)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    logger.info(
        "Dataset loaded — %d rows, %d columns", df.shape[0], df.shape[1]
    )
    return df
