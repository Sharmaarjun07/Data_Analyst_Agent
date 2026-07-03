"""
services/prediction.py

Prediction pipeline, split out of ml_router.py so the router stays thin:

    load artifacts -> validate dataset -> transform -> predict
    -> predict_proba -> CSV for download

ml_router.py should call this module's functions rather than containing
prediction logic itself.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

from services import model_saver as saver
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class PredictionError(Exception):
    """Base class for prediction-pipeline errors. Callers (e.g. the router) can catch this
    to map failures to an appropriate HTTP status, without needing to know the specific subclass."""


class ModelNotFoundError(PredictionError):
    """Raised when the requested run_name has no saved artifacts."""


class DatasetValidationError(PredictionError):
    """Raised when the uploaded dataset doesn't match what the model expects."""


# --------------------------------------------------------------------------- #
# 1. Load artifacts
# --------------------------------------------------------------------------- #

def load_prediction_artifacts(run_name: str) -> dict:
    """
    Load model, preprocessor, feature_names, and label_encoder for a saved run.
    Raises ModelNotFoundError if the run doesn't exist.
    """
    run_dir = config.MODELS_DIR / run_name
    if not run_dir.exists():
        raise ModelNotFoundError(f"No saved model found for run '{run_name}'.")

    logger.info("Loading prediction artifacts for run '%s'", run_name)
    try:
        return saver.load_artifacts(run_dir)
    except FileNotFoundError as exc:
        # A run directory exists but is missing a required file (e.g. partial save).
        raise ModelNotFoundError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# 2. Validate dataset
# --------------------------------------------------------------------------- #

def validate_dataset(df: pd.DataFrame, feature_names: list[str]) -> None:
    """
    Confirm the uploaded dataframe has every column the model was trained on.
    Raises DatasetValidationError with an actionable message otherwise.
    """
    if df is None or df.empty:
        raise DatasetValidationError("Uploaded dataset contains no rows.")

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing columns required by this model: {missing}. "
            f"Expected columns: {feature_names}."
        )


# --------------------------------------------------------------------------- #
# 3. Transform
# --------------------------------------------------------------------------- #

def transform(df: pd.DataFrame, preprocessor, feature_names: list[str]):
    """Apply the saved, fitted preprocessor to new data, restricted to the trained feature set."""
    X = df[feature_names]
    if hasattr(preprocessor, "transform"):
        return preprocessor.transform(X)
    logger.warning("Saved preprocessor has no .transform(); passing raw feature columns through unchanged.")
    return X


# --------------------------------------------------------------------------- #
# 4 & 5. Predict / predict_proba
# --------------------------------------------------------------------------- #

def predict(run_name: str, df: pd.DataFrame, return_proba: bool = False) -> dict:
    """
    Run the full prediction pipeline for a saved model: load artifacts,
    validate, transform, predict (and optionally predict_proba).

    Returns
    -------
    dict with: run_name, n_rows, predictions, and (if return_proba=True)
    probabilities + classes (both None if the model has no predict_proba).
    """
    artifacts = load_prediction_artifacts(run_name)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_names = artifacts["feature_names"]
    label_encoder = artifacts["label_encoder"]

    validate_dataset(df, feature_names)
    X_transformed = transform(df, preprocessor, feature_names)

    logger.info("Predicting %d row(s) with run '%s' (%s)", len(df), run_name, type(model).__name__)
    try:
        raw_predictions = model.predict(X_transformed)
    except Exception as exc:
        raise PredictionError(f"Model prediction failed: {exc}") from exc

    if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
        predictions = label_encoder.inverse_transform(raw_predictions).tolist()
    else:
        predictions = raw_predictions.tolist() if hasattr(raw_predictions, "tolist") else list(raw_predictions)

    result: dict = {"run_name": run_name, "n_rows": len(predictions), "predictions": predictions}

    if return_proba:
        proba, classes = _predict_proba(model, X_transformed, label_encoder)
        result["probabilities"] = proba
        result["classes"] = classes

    return result


def _predict_proba(model, X_transformed, label_encoder) -> tuple[Optional[list], Optional[list]]:
    """Best-effort predict_proba; returns (None, None) if the model doesn't support it."""
    if not hasattr(model, "predict_proba"):
        logger.info("Model %s has no predict_proba(); skipping probabilities.", type(model).__name__)
        return None, None

    try:
        proba = model.predict_proba(X_transformed)
    except Exception as exc:
        logger.warning("predict_proba() failed: %s", exc)
        return None, None

    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        classes = label_encoder.classes_.tolist()
    elif hasattr(model, "classes_"):
        classes = model.classes_.tolist()
    else:
        classes = [f"class_{i}" for i in range(proba.shape[1])]

    return proba.tolist(), classes


# --------------------------------------------------------------------------- #
# 6. Download CSV
# --------------------------------------------------------------------------- #

def predict_to_dataframe(run_name: str, df: pd.DataFrame, return_proba: bool = False) -> pd.DataFrame:
    """Same as predict(), but returns the original rows with prediction (and probability) columns appended."""
    result = predict(run_name, df, return_proba=return_proba)

    out = df.copy()
    out["prediction"] = result["predictions"]

    if return_proba and result.get("probabilities") is not None:
        classes = result["classes"] or [f"class_{i}" for i in range(len(result["probabilities"][0]))]
        proba_df = pd.DataFrame(
            result["probabilities"],
            columns=[f"proba_{c}" for c in classes],
            index=out.index,
        )
        out = pd.concat([out, proba_df], axis=1)

    return out


def predictions_to_csv_bytes(run_name: str, df: pd.DataFrame, return_proba: bool = False) -> bytes:
    """Generate a downloadable CSV (as bytes) of predictions appended to the input data."""
    out_df = predict_to_dataframe(run_name, df, return_proba=return_proba)
    buffer = StringIO()
    out_df.to_csv(buffer, index=False)
    logger.info("Generated prediction CSV (%d rows, proba=%s) for run '%s'", len(out_df), return_proba, run_name)
    return buffer.getvalue().encode("utf-8")


def save_predictions_csv(
    run_name: str,
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    return_proba: bool = False,
) -> Path:
    """Write predictions to disk as CSV and return the path. Defaults to models/<run_name>/predictions.csv."""
    out_df = predict_to_dataframe(run_name, df, return_proba=return_proba)
    path = output_path or (config.MODELS_DIR / run_name / "predictions.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)
    logger.info("Saved predictions CSV to %s", path)
    return path
