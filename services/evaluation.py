"""
Evaluation metrics, leaderboard building, plotting, and metric persistence
for the ML pipeline — the central reporting layer of the AutoML system.

Backward compatibility note: `evaluate()` still returns a flat dict of scalar
metrics (same contract model_selector.py already relies on), so existing
callers keep working unchanged. Everything else here — ROC/log-loss,
confusion matrices, classification reports, plots, leaderboards, persistence —
is additive.
"""

import json
import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score, median_absolute_error,
)

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

logger = logging.getLogger(__name__)

try:
    from utils.config import CLASSIFICATION_PRIMARY_METRIC, REGRESSION_PRIMARY_METRIC
except ImportError:
    CLASSIFICATION_PRIMARY_METRIC = "accuracy"
    REGRESSION_PRIMARY_METRIC = "r2"

REPORTS_DIR = "reports"


def _round(value, digits: int):
    return None if value is None else round(float(value), digits)


# ─────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────
def evaluate_classification(
    y_true, y_pred, y_proba=None, round_output: bool = True
) -> dict[str, float]:
    """
    Core classification metrics (accuracy/precision/recall/f1), plus ROC AUC
    and log loss when class probabilities are available. Both extras are
    computed defensively — a failure in either never blocks the core metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
    }

    if y_proba is not None:
        n_classes = len(np.unique(y_true))
        try:
            if n_classes == 2:
                proba_positive = y_proba[:, 1] if getattr(y_proba, "ndim", 1) > 1 else y_proba
                metrics["roc_auc"] = roc_auc_score(y_true, proba_positive)
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception as e:
            logger.warning("Could not compute ROC AUC: %s", e)
            metrics["roc_auc"] = None

        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except Exception as e:
            logger.warning("Could not compute log loss: %s", e)
            metrics["log_loss"] = None

    if round_output:
        for key in ("accuracy", "precision", "recall", "f1"):
            metrics[key] = _round(metrics[key], 2)
        for key in ("roc_auc", "log_loss"):
            if key in metrics:
                metrics[key] = _round(metrics[key], 4)

    return metrics


def confusion_matrix_dict(y_true, y_pred, labels: Optional[list] = None) -> dict:
    """JSON-serializable confusion matrix, ready to hand to a frontend."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "matrix": matrix.tolist(),
        "labels": labels if labels is not None else sorted(np.unique(np.concatenate([y_true, y_pred])).tolist()),
    }


def classification_report_dict(y_true, y_pred) -> dict:
    """Per-class precision/recall/f1/support as a JSON-serializable dict."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


# ─────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────
def evaluate_regression(y_true, y_pred, round_output: bool = True) -> dict[str, float]:
    """Core + extended regression metrics: MAE/MSE/RMSE/R2 plus MAPE, explained
    variance, and median absolute error (robust to outliers)."""
    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
    }

    try:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred) * 100
    except Exception as e:
        logger.warning("Could not compute MAPE (likely a zero in y_true): %s", e)
        metrics["mape"] = None

    if round_output:
        digits = {"mae": 4, "mse": 4, "rmse": 4, "r2": 4,
                  "explained_variance": 4, "median_absolute_error": 4, "mape": 2}
        for key, d in digits.items():
            metrics[key] = _round(metrics[key], d)

    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Unified entry point (same contract model_selector.py already calls)
# ─────────────────────────────────────────────────────────────────────────
def evaluate(
    y_true, y_pred, problem_type: str, y_proba=None, round_output: bool = True
) -> dict[str, float]:
    """
    Compute metrics for a single model's predictions. Returns a flat dict of
    scalar metrics only (no confusion matrix / classification report — those
    stay non-scalar and live in the dedicated *_dict() functions above so
    this stays safe to drop straight into a leaderboard DataFrame).
    """
    logger.info("Evaluating %s predictions on %d samples", problem_type, len(y_true))

    if problem_type == "classification":
        metrics = evaluate_classification(y_true, y_pred, y_proba, round_output)
    elif problem_type == "regression":
        metrics = evaluate_regression(y_true, y_pred, round_output)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    logger.info("Result: %s", metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────
def _timed(func: Callable, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = round(time.perf_counter() - start, 4)
    return result, elapsed


def calculate_training_time(fit_func: Callable, *args, **kwargs) -> tuple:
    """Run `fit_func(*args, **kwargs)`, returning (result, seconds_elapsed)."""
    return _timed(fit_func, *args, **kwargs)


def calculate_prediction_time(predict_func: Callable, *args, **kwargs) -> tuple:
    """Run `predict_func(*args, **kwargs)`, returning (result, seconds_elapsed)."""
    return _timed(predict_func, *args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Model selection / leaderboard
# ─────────────────────────────────────────────────────────────────────────
def pick_best_model(results: dict[str, dict], problem_type: str) -> str:
    """Given {model_name: metrics_dict}, return the name of the best model."""
    metric = CLASSIFICATION_PRIMARY_METRIC if problem_type == "classification" else REGRESSION_PRIMARY_METRIC
    return max(results, key=lambda name: results[name][metric])


def create_leaderboard(results: dict[str, dict], problem_type: str) -> pd.DataFrame:
    """
    Lightweight leaderboard from {model_name: metrics_dict}, sorted best-to-worst
    on the primary metric.

    Note: model_selector.py has its own, richer create_leaderboard() that also
    includes CV scores, timing, and the fitted pipeline objects — use that one
    when you have full training results. This version is for quick, ad-hoc
    metric comparisons when all you have is {name: evaluate()-output}.
    """
    metric = CLASSIFICATION_PRIMARY_METRIC if problem_type == "classification" else REGRESSION_PRIMARY_METRIC
    rows = [{"Model": name, **metrics} for name, metrics in results.items()]
    board = pd.DataFrame(rows).sort_values(by=metric, ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", board.index + 1)
    return board


# ─────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────
def save_metrics(metrics: dict, filepath: Optional[str] = None) -> str:
    """Write a metrics dict to disk as JSON. Defaults to reports/metrics_<timestamp>.json."""
    if filepath is None:
        filepath = os.path.join(REPORTS_DIR, f"metrics_{int(time.time())}.json")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Saved metrics to %s", filepath)
    return filepath


# ─────────────────────────────────────────────────────────────────────────
# Visualizations (all optional — degrade gracefully without matplotlib)
# ─────────────────────────────────────────────────────────────────────────
def _resolve_plot_path(save_path: Optional[str], default_name: str) -> str:
    path = save_path or os.path.join(REPORTS_DIR, default_name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def plot_confusion_matrix(y_true, y_pred, labels: Optional[list] = None,
                           save_path: Optional[str] = None) -> Optional[str]:
    """Save a confusion matrix heatmap to disk. Returns the file path, or None if matplotlib is unavailable."""
    if not _HAS_MPL:
        logger.warning("matplotlib not installed — skipping confusion matrix plot")
        return None

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = labels if labels is not None else sorted(np.unique(np.concatenate([y_true, y_pred])))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = _resolve_plot_path(save_path, "confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", path)
    return path


def plot_roc_curve(y_true, y_proba, save_path: Optional[str] = None) -> Optional[str]:
    """Save a binary ROC curve to disk. Multi-class targets are skipped (returns None) — plot one-vs-rest curves separately if needed."""
    if not _HAS_MPL:
        logger.warning("matplotlib not installed — skipping ROC curve plot")
        return None
    if len(np.unique(y_true)) != 2:
        logger.warning("plot_roc_curve only supports binary targets — skipping")
        return None

    from sklearn.metrics import roc_curve

    proba_positive = y_proba[:, 1] if getattr(y_proba, "ndim", 1) > 1 else y_proba
    fpr, tpr, _ = roc_curve(y_true, proba_positive)
    auc = roc_auc_score(y_true, proba_positive)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = _resolve_plot_path(save_path, "roc_curve.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved ROC curve plot to %s", path)
    return path


def plot_residuals(y_true, y_pred, save_path: Optional[str] = None) -> Optional[str]:
    """Save predicted-vs-actual and residual-distribution plots for a regression model."""
    if not _HAS_MPL:
        logger.warning("matplotlib not installed — skipping residual plots")
        return None

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.scatter(y_true, y_pred, alpha=0.5, s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, linestyle="--", color="gray")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual")

    ax2.hist(residuals, bins=30, color="steelblue", edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")

    fig.tight_layout()
    path = _resolve_plot_path(save_path, "residuals.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved residual plots to %s", path)
    return path
