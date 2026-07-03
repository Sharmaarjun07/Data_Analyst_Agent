"""
utils/config.py

Single source of truth for constants shared across the pipeline (splitting,
cross-validation, directory paths, explainability/report defaults). Import
this instead of re-declaring RANDOM_STATE, TEST_SIZE, etc. in individual
modules.

Usage
-----
    from utils.config import config

    X_train, X_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

Every value can be overridden via environment variable at import time (e.g.
`AUTOML_RANDOM_STATE=7`), which is useful for tests and deployment without
editing this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"AUTOML_{name}", default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"AUTOML_{name}", default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"AUTOML_{name}", default)


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(f"AUTOML_{name}", default))


@dataclass(frozen=True)
class Config:
    # --- Reproducibility & splitting ---
    RANDOM_STATE: int = _env_int("RANDOM_STATE", 42)
    TEST_SIZE: float = _env_float("TEST_SIZE", 0.2)
    CV_FOLDS: int = _env_int("CV_FOLDS", 5)

    # --- Directory paths ---
    MODELS_DIR: Path = _env_path("MODELS_DIR", "models")
    REPORTS_DIR: Path = _env_path("REPORTS_DIR", "reports")

    # --- Explainability defaults (see services/explainability.py) ---
    TOP_FEATURES: int = _env_int("TOP_FEATURES", 10)
    PERMUTATION_REPEATS: int = _env_int("PERMUTATION_REPEATS", 10)
    SHAP_BACKGROUND_SAMPLES: int = _env_int("SHAP_BACKGROUND_SAMPLES", 100)
    SHAP_MAX_DISPLAY: int = _env_int("SHAP_MAX_DISPLAY", 10)
    FIGURE_DPI: int = _env_int("FIGURE_DPI", 150)

    # --- Model artifact filenames (see services/model_saver.py) ---
    MODEL_FILENAME: str = _env_str("MODEL_FILENAME", "best_model.pkl")
    PREPROCESSOR_FILENAME: str = _env_str("PREPROCESSOR_FILENAME", "preprocessor.pkl")
    LABEL_ENCODER_FILENAME: str = _env_str("LABEL_ENCODER_FILENAME", "label_encoder.pkl")
    FEATURE_NAMES_FILENAME: str = _env_str("FEATURE_NAMES_FILENAME", "feature_names.pkl")
    METADATA_FILENAME: str = _env_str("METADATA_FILENAME", "metadata.json")
    METRICS_FILENAME: str = _env_str("METRICS_FILENAME", "metrics.json")
    LEADERBOARD_FILENAME: str = _env_str("LEADERBOARD_FILENAME", "leaderboard.csv")

    # --- Thresholds ---
    CLASSIFICATION_THRESHOLD: float = _env_float("CLASSIFICATION_THRESHOLD", 0.5)
    MIN_ROWS_FOR_TRAINING: int = _env_int("MIN_ROWS_FOR_TRAINING", 20)

    # --- Logging ---
    LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO")


config = Config()
