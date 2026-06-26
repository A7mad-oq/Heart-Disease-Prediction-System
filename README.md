# Heart Disease Classifier

> **ML project** — Ahmad Al-oqdeh (202310777)  
> Course: Machine Learning · Instructor: Dr. Hossam Mustafa

A production-level Python package for predicting the presence of heart disease using four classification algorithms:

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable rule-based |
| **Random Forest** | ⭐ Best performer (ensemble) |
| SVM | High-sensitivity boundary |

---

## Project Structure

```
heart_disease_classifier/
├── data/
│   ├── raw/                   ← Original CSV dataset
│   └── processed/             ← Results and artefacts
├── docs/                      ← Extended documentation
├── notebooks/
│   └── exploration.ipynb      ← Original exploratory notebook
├── outputs/                   ← Auto-created on first run
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── best_model.joblib
│   └── preprocessor.joblib
├── scripts/
│   └── train.py               ← CLI entry point
├── src/
│   └── heart_disease_classifier/
│       ├── __init__.py
│       ├── pipeline.py        ← End-to-end orchestrator
│       ├── data/
│       │   └── loader.py
│       ├── preprocessing/
│       │   └── pipeline.py
│       ├── models/
│       │   └── trainer.py
│       ├── evaluation/
│       │   └── metrics.py
│       └── utils/
│           ├── logging.py
│           └── persistence.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
# Clone and install in editable mode
git clone <your-repo-url>
cd heart_disease_classifier

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package with dev dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### From the command line

```bash
python scripts/train.py
# or with options:
python scripts/train.py \
    --data data/raw/heart_prediction_quantum.csv \
    --output outputs/ \
    --test-size 0.25 \
    --log-level INFO
```

### From Python

```python
from heart_disease_classifier import HeartDiseasePipeline
from heart_disease_classifier.utils import setup_logging

setup_logging()

pipe = HeartDiseasePipeline(
    data_path="data/raw/heart_prediction_quantum.csv",
    output_dir="outputs/",
    test_size=0.25,
    random_state=42,
)
pipe.run()
pipe.print_summary()

# Access results programmatically
print(pipe.results_df)
print("Best model:", pipe.best_model_name)
```

---

## Running the Tests

```bash
pytest                          # run all tests
pytest --cov                    # with coverage report
pytest tests/test_preprocessing.py -v   # single file
```

---

## Dataset

| Column | Description |
|---|---|
| Age | Patient age (years) |
| Gender | 0 = female, 1 = male |
| BloodPressure | Systolic BP (mmHg) — discretised during preprocessing |
| Cholesterol | Total cholesterol (mg/dL) |
| HeartRate | Resting heart rate (bpm) |
| QuantumPatternFeature | Domain-specific engineered feature |
| HeartDisease | **Target** — 0 = no disease, 1 = disease |

### Preprocessing steps

1. **Imputation** — median for numeric columns, mode for `Gender`.
2. **Scaling** — StandardScaler on `Age`, `Cholesterol`, `HeartRate`, `QuantumPatternFeature`.
3. **Blood Pressure encoding** — continuous BP binned into `Low / Normal / High / VeryHigh`, then one-hot encoded; original column dropped.
4. **Train / test split** — 75 % / 25 %, `random_state=42`.

---

## Results

After running the pipeline, `outputs/model_comparison.csv` contains:

| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | … | … | … | … | … |
| Decision Tree | … | … | … | … | … |
| **Random Forest** | **…** | **…** | **…** | **…** | **…** |
| SVM | … | … | … | … | … |

Random Forest achieved the highest F1-Score, making it the recommended model for deployment.

---

## License

MIT — see `LICENSE` for details.
