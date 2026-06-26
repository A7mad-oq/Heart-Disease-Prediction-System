"""
heart_disease_classifier.utils.persistence
===========================================
Save and load fitted artefacts (preprocessor, models) using ``joblib``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def save_artifact(obj: Any, path: str | Path) -> None:
    """Serialise *obj* to *path* using joblib.

    Parameters
    ----------
    obj : any
        Fitted sklearn estimator or custom object (e.g. :class:`~heart_disease_classifier.preprocessing.pipeline.Preprocessor`).
    path : str or Path
        Destination file (e.g. ``models/random_forest.joblib``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Artefact saved → %s", path)


def load_artifact(path: str | Path) -> Any:
    """Deserialise an artefact previously saved with :func:`save_artifact`.

    Parameters
    ----------
    path : str or Path
        File produced by :func:`save_artifact`.

    Returns
    -------
    object
        The restored Python object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    obj = joblib.load(path)
    logger.info("Artefact loaded ← %s", path)
    return obj
