import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models_saved"

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# File names (supports standard Kaggle competition naming)
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

# Model Hyperparameters
LIGHTGBM_HEAD_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 400,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

LIGHTGBM_REL_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 45,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "n_estimators": 350,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
