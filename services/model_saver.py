"""
model_saver.py

Model *artifact* management for the AutoML pipeline — not just the trained
estimator, but everything required to reproduce a prediction on new data:

    models/
    <run_name>/
        best_model.pkl          (or a versioned name, e.g. RandomForest_2026_07_03.pkl)
        preprocessor.pkl
        label_encoder.pkl
        feature_names.pkl
        metadata.json
        metrics.json
    leaderboard.csv              (appended to across runs, at the top level)

Typical usage
-------------
    from model_saver import save_artifacts, load_artifacts

    paths = save_artifacts(
        model=best_model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        metrics={"accuracy": 0.961, "f1": 0.958, "roc_auc": 0.982},
        metadata={"problem_type": "classification", "target": "Churn", "model": "Random Forest"},
        label_encoder=label_encoder,   # optional
        run_name="RandomForest_v1",    # optional, auto-generated if omitted
    )

    artifacts = load_artifacts(paths["run_dir"])
    artifacts["model"]
    artifacts["preprocessor"]
    artifacts["metadata"]["training_timestamp"]

Or use the individual building blocks:

    save_model() / load_model()
    save_preprocessor() / load_preprocessor()
    save_label_encoder() / load_label_encoder()
    save_feature_names() / load_feature_names()
    save_metadata() / load_metadata()
    save_metrics() / load_metrics()
    update_leaderboard()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import pandas as pd

try:
    from sklearn.base import BaseEstimator
except ImportError:  # sklearn is a soft dependency, used only for typing
    BaseEstimator = object  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ModelSaverConfig:
    """Central configuration instead of a hardcoded MODELS_DIR constant."""
    models_dir: Path = field(default_factory=lambda: Path("models"))
    leaderboard_filename: str = "leaderboard.csv"
    model_filename: str = "best_model.pkl"
    preprocessor_filename: str = "preprocessor.pkl"
    label_encoder_filename: str = "label_encoder.pkl"
    feature_names_filename: str = "feature_names.pkl"
    metadata_filename: str = "metadata.json"
    metrics_filename: str = "metrics.json"


DEFAULT_CONFIG = ModelSaverConfig()


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _validate_not_none(obj: Any, name: str) -> None:
    if obj is None:
        raise ValueError(f"Refusing to save '{name}': received None.")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_dir(run_name: Optional[str], config: ModelSaverConfig) -> Path:
    """Resolve (and create) the directory for a given run/version."""
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return _ensure_dir(config.models_dir / run_name)


# --------------------------------------------------------------------------- #
# 1. Model
# --------------------------------------------------------------------------- #

def save_model(
    model: BaseEstimator,
    filename: str = DEFAULT_CONFIG.model_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Path:
    """Persist the trained estimator. Raises ValueError if model is None."""
    _validate_not_none(model, "model")
    target_dir = run_dir if run_dir is not None else _ensure_dir(config.models_dir)
    path = target_dir / filename

    logger.info("Saving model (%s)...", type(model).__name__)
    joblib.dump(model, path)
    logger.info("Model saved successfully. Path: %s", path)
    return path


def load_model(
    filename: str = DEFAULT_CONFIG.model_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> BaseEstimator:
    """Load a previously saved estimator. Raises FileNotFoundError if missing."""
    target_dir = run_dir if run_dir is not None else config.models_dir
    path = Path(target_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")

    logger.info("Loading model from %s...", path)
    model = joblib.load(path)
    logger.info("Model loaded successfully.")
    return model


# --------------------------------------------------------------------------- #
# 2. Preprocessor (ColumnTransformer, Pipeline, etc.)
# --------------------------------------------------------------------------- #

def save_preprocessor(
    preprocessor: Any,
    filename: str = DEFAULT_CONFIG.preprocessor_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Path:
    """Persist the fitted preprocessing pipeline (e.g. ColumnTransformer)."""
    _validate_not_none(preprocessor, "preprocessor")
    target_dir = run_dir if run_dir is not None else _ensure_dir(config.models_dir)
    path = target_dir / filename

    joblib.dump(preprocessor, path)
    logger.info("Preprocessor saved successfully. Path: %s", path)
    return path


def load_preprocessor(
    filename: str = DEFAULT_CONFIG.preprocessor_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Any:
    """Load the fitted preprocessing pipeline. Raises FileNotFoundError if missing."""
    target_dir = run_dir if run_dir is not None else config.models_dir
    path = Path(target_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No saved preprocessor found at {path}. "
            "Predictions on new data will be inconsistent without it."
        )

    preprocessor = joblib.load(path)
    logger.info("Preprocessor loaded successfully from %s", path)
    return preprocessor


# --------------------------------------------------------------------------- #
# 3. Label encoder (optional — only for encoded classification targets)
# --------------------------------------------------------------------------- #

def save_label_encoder(
    label_encoder: Any,
    filename: str = DEFAULT_CONFIG.label_encoder_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Path:
    """Persist the label encoder used on the target column, if any."""
    _validate_not_none(label_encoder, "label_encoder")
    target_dir = run_dir if run_dir is not None else _ensure_dir(config.models_dir)
    path = target_dir / filename

    joblib.dump(label_encoder, path)
    logger.info("Label encoder saved successfully. Path: %s", path)
    return path


def load_label_encoder(
    filename: str = DEFAULT_CONFIG.label_encoder_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Optional[Any]:
    """
    Load the label encoder, if one was saved.

    Returns None (rather than raising) when absent, since not every problem
    (e.g. regression, or an already-numeric target) has one.
    """
    target_dir = run_dir if run_dir is not None else config.models_dir
    path = Path(target_dir) / filename
    if not path.exists():
        logger.info("No label encoder found at %s (this is expected for regression/unencoded targets).", path)
        return None

    encoder = joblib.load(path)
    logger.info("Label encoder loaded successfully from %s", path)
    return encoder


# --------------------------------------------------------------------------- #
# 4. Feature names
# --------------------------------------------------------------------------- #

def save_feature_names(
    feature_names: Sequence[str],
    filename: str = DEFAULT_CONFIG.feature_names_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Path:
    """Persist the ordered list of feature names (needed later for SHAP, importance, and predictions)."""
    _validate_not_none(feature_names, "feature_names")
    target_dir = run_dir if run_dir is not None else _ensure_dir(config.models_dir)
    path = target_dir / filename

    joblib.dump(list(feature_names), path)
    logger.info("Feature names saved successfully (%d features). Path: %s", len(feature_names), path)
    return path


def load_feature_names(
    filename: str = DEFAULT_CONFIG.feature_names_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> list:
    """Load the saved feature name list. Raises FileNotFoundError if missing."""
    target_dir = run_dir if run_dir is not None else config.models_dir
    path = Path(target_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved feature names found at {path}")

    feature_names = joblib.load(path)
    logger.info("Feature names loaded successfully (%d features).", len(feature_names))
    return feature_names


# --------------------------------------------------------------------------- #
# 5. Metadata (JSON)
# --------------------------------------------------------------------------- #

def save_metadata(
    metadata: dict,
    filename: str = DEFAULT_CONFIG.metadata_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
    model: Optional[BaseEstimator] = None,
) -> Path:
    """
    Persist run metadata as JSON. Automatically stamps a training_timestamp,
    and, if `model` is given, records its type and get_params() output.
    """
    _validate_not_none(metadata, "metadata")
    target_dir = run_dir if run_dir is not None else _ensure_dir(config.models_dir)
    path = target_dir / filename

    enriched = dict(metadata)
    enriched.setdefault("training_timestamp", datetime.now().isoformat())
    if model is not None:
        enriched.setdefault("model_type", type(model).__name__)
        if hasattr(model, "get_params"):
            try:
                enriched.setdefault("model_params", model.get_params())
            except Exception as exc:
                logger.warning("Could not capture model.get_params(): %s", exc)

    with open(path, "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    logger.info("Metadata saved successfully. Path: %s", path)
    return path


def load_metadata(
    filename: str = DEFAULT_CONFIG.metadata_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> dict:
    """Load run metadata. Raises FileNotFoundError if missing."""
    target_dir = run_dir if run_dir is not None else config.models_dir
    path = Path(target_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved metadata found at {path}")

    with open(path) as f:
        metadata = json.load(f)
    logger.info("Metadata loaded successfully from %s", path)
    return metadata


# --------------------------------------------------------------------------- #
# 6. Metrics (JSON)
# --------------------------------------------------------------------------- #

def save_metrics(
    metrics: dict,
    filename: str = DEFAULT_CONFIG.metrics_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Path:
    """Persist evaluation metrics as JSON (e.g. accuracy, f1, roc_auc)."""
    _validate_not_none(metrics, "metrics")
    target_dir = run_dir if run_dir is not None else _ensure_dir(config.models_dir)
    path = target_dir / filename

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved successfully. Path: %s", path)
    return path


def load_metrics(
    filename: str = DEFAULT_CONFIG.metrics_filename,
    run_dir: Optional[Path] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> dict:
    """Load evaluation metrics. Raises FileNotFoundError if missing."""
    target_dir = run_dir if run_dir is not None else config.models_dir
    path = Path(target_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved metrics found at {path}")

    with open(path) as f:
        metrics = json.load(f)
    logger.info("Metrics loaded successfully from %s", path)
    return metrics


# --------------------------------------------------------------------------- #
# 7. Leaderboard (CSV, appended across runs)
# --------------------------------------------------------------------------- #

def update_leaderboard(
    run_name: str,
    metrics: dict,
    metadata: Optional[dict] = None,
    filename: str = DEFAULT_CONFIG.leaderboard_filename,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> Path:
    """
    Append (or update) a row for this run in the top-level leaderboard.csv,
    so multiple training runs can be compared at a glance.

    If `run_name` already exists in the leaderboard, its row is overwritten
    rather than duplicated.
    """
    _ensure_dir(config.models_dir)
    path = config.models_dir / filename

    row = {"run_name": run_name, "timestamp": datetime.now().isoformat(), **metrics}
    if metadata:
        row.update({f"meta_{k}": v for k, v in metadata.items() if k in ("model", "model_type", "problem_type", "target")})

    if path.exists():
        board = pd.read_csv(path)
        board = board[board["run_name"] != run_name]
        board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
    else:
        board = pd.DataFrame([row])

    board.to_csv(path, index=False)
    logger.info("Leaderboard updated (%d runs total). Path: %s", len(board), path)
    return path


def load_leaderboard(
    filename: str = DEFAULT_CONFIG.leaderboard_filename,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load the leaderboard as a DataFrame. Returns an empty DataFrame if none exists yet."""
    path = config.models_dir / filename
    if not path.exists():
        logger.info("No leaderboard found at %s yet.", path)
        return pd.DataFrame()
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# 8. All-in-one save / load
# --------------------------------------------------------------------------- #

def save_artifacts(
    model: BaseEstimator,
    preprocessor: Any,
    feature_names: Sequence[str],
    metrics: dict,
    metadata: dict,
    label_encoder: Optional[Any] = None,
    run_name: Optional[str] = None,
    config: ModelSaverConfig = DEFAULT_CONFIG,
    update_leaderboard_entry: bool = True,
) -> dict:
    """
    Save every artifact needed to reproduce predictions, in one call:
    model, preprocessor, feature names, metadata, metrics, and (optionally)
    a label encoder — all versioned under models/<run_name>/, plus a row in
    the top-level leaderboard.

    Example
    -------
        save_artifacts(
            model=best_model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            metrics={"accuracy": 0.961, "f1": 0.958},
            metadata={"problem_type": "classification", "target": "Churn"},
            run_name="RandomForest_v1",
        )

    Returns a dict of every path written, plus "run_dir" and "run_name".
    """
    for obj, name in [(model, "model"), (preprocessor, "preprocessor"), (feature_names, "feature_names"),
                       (metrics, "metrics"), (metadata, "metadata")]:
        _validate_not_none(obj, name)

    resolved_run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = _run_dir(resolved_run_name, config)

    logger.info("Saving all artifacts for run '%s' to %s ...", resolved_run_name, run_dir)

    paths = {
        "run_name": resolved_run_name,
        "run_dir": run_dir,
        "model": save_model(model, run_dir=run_dir, config=config),
        "preprocessor": save_preprocessor(preprocessor, run_dir=run_dir, config=config),
        "feature_names": save_feature_names(feature_names, run_dir=run_dir, config=config),
        "metadata": save_metadata(metadata, run_dir=run_dir, config=config, model=model),
        "metrics": save_metrics(metrics, run_dir=run_dir, config=config),
    }

    if label_encoder is not None:
        paths["label_encoder"] = save_label_encoder(label_encoder, run_dir=run_dir, config=config)

    if update_leaderboard_entry:
        paths["leaderboard"] = update_leaderboard(resolved_run_name, metrics, metadata, config=config)

    logger.info("All artifacts for run '%s' saved successfully.", resolved_run_name)
    return paths


def load_artifacts(
    run_dir: Path,
    config: ModelSaverConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Load every artifact for a given run directory in one call.

    Returns a dict with keys: model, preprocessor, feature_names, metadata,
    metrics, label_encoder (None if not saved for this run).
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"No run directory found at {run_dir}")

    logger.info("Loading all artifacts from %s ...", run_dir)

    artifacts = {
        "model": load_model(run_dir=run_dir, config=config),
        "preprocessor": load_preprocessor(run_dir=run_dir, config=config),
        "feature_names": load_feature_names(run_dir=run_dir, config=config),
        "metadata": load_metadata(run_dir=run_dir, config=config),
        "metrics": load_metrics(run_dir=run_dir, config=config),
        "label_encoder": load_label_encoder(run_dir=run_dir, config=config),
    }

    logger.info("All artifacts loaded successfully from %s", run_dir)
    return artifacts
