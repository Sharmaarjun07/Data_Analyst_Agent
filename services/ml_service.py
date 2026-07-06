"""
ML Service — orchestrates the full machine learning pipeline:
detect target -> preprocess (incl. feature engineering) -> split ->
train & compare models -> select best -> evaluate -> explain -> save
everything -> return results.

IMPORTANT — assumed collaborator APIs
--------------------------------------
This file was updated to fix the target-vs-problem_type flow assuming
`services.preprocessing` now exposes a class-based API:

    processor = pp.DataPreprocessor()
    data = processor.prepare_dataset(df, user_target=user_target)
    # data.target, data.problem_type, data.X_train, data.X_test,
    # data.y_train, data.y_test, data.feature_names, data.target_encoder,
    # data.preprocessor  (the fitted transformer, for saving)

and that `prepare_dataset` now owns ALL feature engineering (datetime, ids,
encoding) so `services.feature_engineering` is no longer called on the main
supervised path — it's only used for the clustering fallback, where there is
no target to drive `DataPreprocessor`.

It also assumes `model_selector.train_and_compare` optionally returns, per
model, `cv_mean`, `cv_std`, `training_time_seconds`, and `best_params` when
tuning was performed, and pulls them defensively (`.get(...)`) so this file
does not break if those keys aren't populated yet.

If your actual `preprocessing` / `feature_engineering` / `model_selector`
signatures differ from the above, adjust the calls in `run_ml_pipeline` and
`_run_clustering_fallback` accordingly — everything else (logging, progress
callback, experiment id, artifact saving) is independent of those details.
ML Service — orchestrates the full machine learning pipeline:
detect target -> preprocess (incl. feature engineering) -> split ->
train & compare models -> select best -> evaluate -> explain -> save
everything -> return results.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional

import pandas as pd
import numpy as np
from services import preprocessing as pp
from services import feature_engineering as fe
from services import model_selector as ms
from services import explainability as ex
from services import model_saver as saver


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ProgressCallback = Callable[[int, str], None]


def to_python(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalar types to native Python types
    so the returned dict is JSON-serialisable."""
    if isinstance(obj, dict):
        return {
            to_python(k): to_python(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [to_python(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_python(v) for v in obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    return obj


def _noop_progress(percent: int, message: str) -> None:
    """Default progress callback: just logs. Swap in a real callback (e.g. websocket push) from the caller."""
    logger.info("[%3d%%] %s", percent, message)


def run_ml_pipeline(
    df: pd.DataFrame,
    user_target: str | None = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """
    Entry point called by the /train-model endpoint.

    Parameters
    ----------
    df : cleaned dataframe (already passed through the Cleaning Service).
    user_target : optional column name chosen manually by the user in the frontend.
    progress_callback : optional callable(percent: int, message: str) so the
        frontend can render a progress bar ("40% — Training Random Forest").
        Defaults to a no-op that just logs.

    Returns
    -------
    dict with experiment_id, target_column, problem_type, best_model,
    metrics, model_comparison (leaderboard), best_parameters, cross_validation,
    feature_importance, and artifact_paths.
    """
    progress = progress_callback or _noop_progress
    experiment_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info("Experiment %s: training started.", experiment_id)
    progress(0, "Training started")

    # -------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------
    progress(10, "Preprocessing data")

    processor = pp.DataPreprocessor()

    data = processor.prepare_dataset(
        df,
        user_target=user_target,
    )

    target = data["target"]
    problem_type = data["problem_type"]

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_test = data["y_test"]

    feature_names = list(X_train.columns)

    target_encoder = None

    fitted_preprocessor = data["preprocessor"]

    logger.info("Experiment %s: problem_type=%s, %d features.", experiment_id, problem_type, len(feature_names))

    # --- Train & compare candidate models ---
    progress(30, f"Training candidate models ({problem_type})")
    comparison = ms.train_and_compare(
        X_train, X_test, y_train, y_test, problem_type,
        preprocessor=fitted_preprocessor,
    )
    best_name = comparison["best_model_name"]
    best_model = comparison["best_model"]
    best_metrics = next(
        r for r in comparison["results"]
        if r["name"] == best_name
    )
    progress(60, f"Best model so far: {best_name}")

    leaderboard = comparison["leaderboard"]
    best_params = best_metrics.get("params")
    cross_validation = {
        "cv_mean": best_metrics.get("cv_mean"),
        "cv_std": best_metrics.get("cv_std"),
    }

    # --- Explainability ---
    progress(75, "Computing feature importance")
    try:
        explanation = ex.explain_model(best_model, X_test, y_test, feature_names, save_reports=True)
        feature_importance = explanation["feature_importance"]
    except AttributeError:
        # Fallback if the explainability module hasn't been upgraded to explain_model() yet.
        feature_importance = ex.get_feature_importance(best_model, feature_names)

    # --- Persist all artifacts together, not just the model ---
    progress(90, "Saving model artifacts")
    artifact_paths = saver.save_artifacts(
        model=best_model,
        preprocessor=fitted_preprocessor,
        feature_names=list(feature_names),
        metrics=best_metrics["metrics"],
        metadata={
            "experiment_id": experiment_id,
            "problem_type": problem_type,
            "target": target,
            "model": best_name,
        },
        label_encoder=target_encoder,
        run_name=f"{best_name}_{experiment_id[:8]}",
    )

    elapsed = round(time.time() - start_time, 2)
    logger.info("Experiment %s: finished in %.2fs. Best model: %s", experiment_id, elapsed, best_name)
    progress(100, f"Finished — best model: {best_name} ({elapsed}s)")

    return to_python({
        "experiment_id": experiment_id,
        "target_column": target,
        "problem_type": problem_type,
        "best_model": best_name,
        "metrics": best_metrics["metrics"],
        "model_comparison": leaderboard,
        "best_parameters": best_params,
        "cross_validation": cross_validation,
        "feature_importance": feature_importance,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "training_time_seconds": elapsed,
        "best_score": (
            best_metrics["metrics"].get("accuracy")
            or best_metrics["metrics"].get("r2")
            or best_metrics["metrics"].get("f1")
        ),
    })


def _run_clustering_fallback(
    df: pd.DataFrame,
    experiment_id: Optional[str] = None,
    progress: ProgressCallback = _noop_progress,
) -> dict[str, Any]:
    """No usable target column found — fall back to KMeans clustering."""
    experiment_id = experiment_id or str(uuid.uuid4())
    start_time = time.time()

    progress(20, "No target found — preparing features for clustering")
    feature_result = fe.engineer_features(
        df.assign(_dummy_target=0),
        "_dummy_target",
    )
    X = feature_result["data"]

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable feature columns remain for clustering after preprocessing")

    progress(50, "Running KMeans clustering")
    clustering = ms.get_candidate_models("clustering")["KMeans"]
    labels = clustering.fit_predict(X)

    cluster_sizes = {int(k): int(v) for k, v in pd.Series(labels).value_counts().to_dict().items()}

    progress(85, "Saving clustering artifacts")
    artifact_paths = saver.save_artifacts(
        model=clustering,
        preprocessor=X,  # no separate fitted preprocessor object available for the clustering path
        feature_names=list(X.columns),
        metrics={"n_clusters": int(clustering.n_clusters)},
        metadata={
            "experiment_id": experiment_id,
            "problem_type": "clustering",
            "target": None,
            "model": "KMeans",
        },
        run_name=f"KMeans_{experiment_id[:8]}",
    )

    elapsed = round(time.time() - start_time, 2)
    logger.info("Experiment %s: clustering finished in %.2fs.", experiment_id, elapsed)
    progress(100, f"Finished — KMeans with {clustering.n_clusters} clusters ({elapsed}s)")

    return to_python({
        "experiment_id": experiment_id,
        "target_column": None,
        "problem_type": "clustering",
        "best_model": "KMeans",
        "n_clusters": int(clustering.n_clusters),
        "cluster_sizes": cluster_sizes,
        "feature_importance": {},
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "training_time_seconds": elapsed,
    })


def _build_leaderboard(results: dict) -> list[dict]:
    """
    Turn the raw {model_name: metrics_dict} comparison results into a clean
    leaderboard list of {Model, Accuracy, Training Time, CV Score}, sorted
    best-first, for easy frontend display.
    """
    rows = []
    for model_name, metrics in results.items():
        primary_score = (
            metrics.get("accuracy")
            or metrics.get("r2")
            or metrics.get("roc_auc")
            or metrics.get("f1")
        )
        rows.append({
            "Model": model_name,
            "Score": primary_score,
            "Training Time (s)": metrics.get("training_time_seconds"),
            "CV Score": metrics.get("cv_mean"),
        })

    rows.sort(key=lambda r: (r["Score"] is None, -(r["Score"] or 0)))
    return rows