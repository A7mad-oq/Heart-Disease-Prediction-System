"""
scripts/train.py
=================
Command-line entry point for training and evaluating the Heart Disease
Classifier.

Usage examples
--------------
# Basic run with defaults
python scripts/train.py

# Specify a custom dataset path and output directory
python scripts/train.py \\
    --data data/raw/heart_prediction_quantum.csv \\
    --output outputs/ \\
    --test-size 0.25

# Debug-level logging
python scripts/train.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart_disease_classifier.pipeline import HeartDiseasePipeline
from heart_disease_classifier.utils.logging import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="heart-disease-classifier",
        description="Train and evaluate ML models for heart disease prediction.",
    )
    parser.add_argument(
        "--data",
        default="data/raw/heart_prediction_quantum.csv",
        metavar="PATH",
        help="Path to the raw CSV dataset (default: data/raw/heart_prediction_quantum.csv).",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        metavar="DIR",
        help="Directory for results, plots, and model artefacts (default: outputs/).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help="Proportion of data reserved for testing (default: 0.25).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving outputs to disk (dry-run mode).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=getattr(logging, args.log_level))

    pipeline = HeartDiseasePipeline(
        data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    pipeline.run(save_outputs=not args.no_save)
    pipeline.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
