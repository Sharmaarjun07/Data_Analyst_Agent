"""
/train-model, /predict, /models, /leaderboard, /model/{run_name}

Router for the ML Service. Training orchestration lives in services.ml_service,
prediction logic lives in services.prediction, and report generation lives in
services.report_generator. This file is responsible only for HTTP wiring:
request/response shapes, dataset lookup, and mapping domain errors to status
codes.

NOTE on dataset storage
------------------------
`load_cleaned_dataframe(dataset_id)` is still a placeholder — it needs to be
wired to wherever the Cleaning/Metadata Service actually persists cleaned
dataframes (e.g. a session store, object storage + a dataset registry table,
or a shared cache). To unblock local development immediately without a real
Cleaning Service, `POST /train-model/upload` accepts a CSV directly.
"""

from __future__ import annotations

import shutil
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from services.ml_service import run_ml_pipeline
from services import model_saver as saver
from services import prediction
from services import report_generator as rg
from utils.config import config
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@router.post("/train-model")
async def train_model(dataset_id: str, target_column: str | None = None):
    """
    dataset_id: identifier used to look up the cleaned dataframe (from Cleaning/Metadata Service).
    target_column: optional, lets the user manually pick the target from the frontend.
    """
    try:
        df = load_cleaned_dataframe(dataset_id)
        results = run_ml_pipeline(df, user_target=target_column)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.exception("Model training failed for dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")


@router.post("/train-model/upload")
async def train_model_upload(file: UploadFile = File(...), target_column: str | None = None):
    """
    Alternative entry point that skips dataset_id lookup entirely: upload a
    CSV and train directly. Useful for local development and for a frontend
    flow of "Upload CSV -> Train" without a Cleaning Service in the loop yet.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse uploaded CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no rows.")

    try:
        results = run_ml_pipeline(df, user_target=target_column)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Model training failed for uploaded file=%s", file.filename)
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")


# --------------------------------------------------------------------------- #
# Prediction — delegates entirely to services.prediction
# --------------------------------------------------------------------------- #

@router.post("/predict")
async def predict(run_name: str, return_proba: bool = False, file: UploadFile = File(...)):
    """
    Upload a new CSV and get predictions (JSON) from a previously trained &
    saved model. Set return_proba=true to also include class probabilities.
    """
    df = await _read_uploaded_csv(file)

    try:
        result = prediction.predict(run_name, df, return_proba=return_proba)
    except prediction.ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except prediction.DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except prediction.PredictionError as e:
        logger.exception("Prediction failed for run_name=%s", run_name)
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.post("/predict/download")
async def predict_download(run_name: str, return_proba: bool = False, file: UploadFile = File(...)):
    """Same as /predict, but returns a downloadable CSV (original rows + prediction column(s))."""
    from fastapi.responses import StreamingResponse

    df = await _read_uploaded_csv(file)

    try:
        csv_bytes = prediction.predictions_to_csv_bytes(run_name, df, return_proba=return_proba)
    except prediction.ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except prediction.DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except prediction.PredictionError as e:
        logger.exception("Prediction (download) failed for run_name=%s", run_name)
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_name}_predictions.csv"'},
    )


async def _read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    try:
        raw = await file.read()
        return pd.read_csv(BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse uploaded CSV: {e}")


# --------------------------------------------------------------------------- #
# Reports — delegates entirely to services.report_generator
# --------------------------------------------------------------------------- #

@router.post("/model/{run_name}/report")
async def generate_report(run_name: str):
    """Generate training_report.pdf, metrics.csv, feature_importance.csv, leaderboard.csv, and model_card.md."""
    try:
        paths = rg.generate_full_report(run_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Report generation failed for run_name=%s", run_name)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return {k: (str(v) if v else None) for k, v in paths.items()}


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

@router.get("/models")
async def list_models():
    """List every saved training run (model registry), with its metadata and metrics."""
    models_dir = config.MODELS_DIR
    if not models_dir.exists():
        return {"models": []}

    entries = []
    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            metadata = saver.load_metadata(run_dir=run_dir)
        except FileNotFoundError:
            continue
        try:
            metrics = saver.load_metrics(run_dir=run_dir)
        except FileNotFoundError:
            metrics = {}

        entries.append({"run_name": run_dir.name, "metadata": metadata, "metrics": metrics})

    return {"models": entries}


@router.get("/leaderboard")
async def get_leaderboard():
    """Return the cross-run leaderboard (model, score, training time, CV score) for comparison."""
    board = saver.load_leaderboard()
    if board.empty:
        return {"leaderboard": []}
    return {"leaderboard": board.to_dict(orient="records")}


@router.delete("/model/{run_name}")
async def delete_model(run_name: str):
    """Delete a saved training run and all its artifacts."""
    run_dir = config.MODELS_DIR / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"No saved model found for run '{run_name}'.")

    try:
        shutil.rmtree(run_dir)
    except Exception as e:
        logger.exception("Failed to delete run_name=%s", run_name)
        raise HTTPException(status_code=500, detail=f"Could not delete model: {e}")

    logger.info("Deleted saved run '%s'.", run_name)
    return {"deleted": run_name}


# --------------------------------------------------------------------------- #
# Dataset lookup (placeholder — wire to Cleaning/Metadata Service)
# --------------------------------------------------------------------------- #

def load_cleaned_dataframe(dataset_id: str) -> pd.DataFrame:
    """
    Placeholder — replace with your actual storage lookup.

    TODO: wire this to wherever the Cleaning Service persists its output,
    e.g.:
        return cleaning_service_client.get_cleaned_dataframe(dataset_id)
    or a shared object store / session cache keyed by dataset_id.

    Until this is wired up, use POST /train-model/upload to train directly
    from an uploaded CSV instead of a dataset_id.
    """
    raise NotImplementedError(
        f"No dataset storage backend configured — cannot resolve dataset_id='{dataset_id}'. "
        "Wire load_cleaned_dataframe() to your Cleaning/Metadata Service, "
        "or use POST /train-model/upload to train from an uploaded CSV instead."
    )
