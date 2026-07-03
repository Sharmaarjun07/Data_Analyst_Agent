"""
explainability.py

Model explainability utilities for the AutoML pipeline.

Supports:
    - Native tree-based importances (feature_importances_)
    - Native linear-model importances (coef_), with signed +/- breakdown
    - Permutation importance (model-agnostic fallback: KNN, SVM, NN, etc.)
    - SHAP global (summary) and local (waterfall / single-prediction) explanations
    - Tabular (pd.DataFrame) output for easy frontend / Streamlit consumption
    - Plot + CSV export to a reports/ directory
    - Defensive validation and logging throughout

Typical usage
-------------
    from explainability import explain_model

    result = explain_model(model, X_test, y_test, feature_names)
    result["feature_importance_df"]      # ranked pd.DataFrame
    result["method"]                     # "tree" | "linear" | "permutation" | "shap"

Or use the individual building blocks directly:

    get_feature_importance(model, feature_names)
    get_permutation_importance(model, X, y, feature_names)
    compute_shap_values(model, X, feature_names)
    create_feature_dataframe(importance_dict)
    save_feature_importance_plot(df)
    save_shap_summary_plot(shap_values, X)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator
except ImportError:  # sklearn is a soft dependency for typing only
    BaseEstimator = object  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ExplainabilityConfig:
    """Central configuration instead of magic numbers scattered through the module."""
    top_features: int = 10
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    permutation_repeats: int = 10
    permutation_random_state: int = 42
    shap_max_display: int = 10
    shap_background_samples: int = 100  # subsample used to build SHAP background/explainer
    figure_dpi: int = 150


DEFAULT_CONFIG = ExplainabilityConfig()


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _validate_feature_alignment(values: np.ndarray, feature_names: Sequence[str]) -> None:
    """Raise a clear, actionable error if importances and feature names don't line up."""
    if len(feature_names) != len(values):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"the number of importance values ({len(values)}). "
            "Check that X_train columns match the feature_names passed in."
        )


def _ensure_reports_dir(reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# --------------------------------------------------------------------------- #
# 1. Native importance (tree models + linear models)
# --------------------------------------------------------------------------- #

def get_feature_importance(
    model: BaseEstimator,
    feature_names: Sequence[str],
    top_n: int = DEFAULT_CONFIG.top_features,
    signed: bool = False,
) -> dict:
    """
    Return native feature importance, when the model exposes it.

    Parameters
    ----------
    model : fitted estimator (tree-based or linear)
    feature_names : names aligned to model input columns
    top_n : how many top features to keep
    signed : if True and the model is linear, keep the sign of coef_ instead of
             taking the absolute value, so callers can see positive vs. negative
             drivers separately.

    Returns
    -------
    dict: {feature: normalized_importance}, or {} if the model doesn't support
    native importance (e.g. KNN, SVM-RBF, neural nets). Use
    get_permutation_importance() as a model-agnostic fallback in that case.
    """
    importances = None
    method = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        method = "tree"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        raw = coef[0] if coef.ndim > 1 else coef
        importances = raw if signed else np.abs(raw)
        method = "linear"

    if importances is None:
        logger.info(
            "Model %s has no native feature_importances_/coef_. "
            "Falling back to permutation importance is recommended.",
            type(model).__name__,
        )
        return {}

    try:
        _validate_feature_alignment(importances, feature_names)
    except ValueError as exc:
        logger.error("Feature importance extraction failed: %s", exc)
        return {}

    series = pd.Series(importances, index=feature_names)

    # Normalize on the magnitude so signed values still sum (in abs) to 1.
    denom = series.abs().sum()
    normalized = series / denom if denom != 0 else series

    ordered = normalized.reindex(normalized.abs().sort_values(ascending=False).index).head(top_n)
    logger.info("Extracted %s-based feature importance for %d features.", method, len(ordered))

    return {name: round(float(score), 4) for name, score in ordered.items()}


# --------------------------------------------------------------------------- #
# 2. Permutation importance (model-agnostic fallback)
# --------------------------------------------------------------------------- #

def get_permutation_importance(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = DEFAULT_CONFIG.top_features,
    n_repeats: int = DEFAULT_CONFIG.permutation_repeats,
    random_state: int = DEFAULT_CONFIG.permutation_random_state,
    scoring: Optional[str] = None,
) -> dict:
    """
    Model-agnostic importance via sklearn.inspection.permutation_importance.

    Works for any fitted model with a .predict / .score method, including
    KNN, SVM, and neural networks, where native importances aren't available.

    Returns {} (with a logged reason) instead of raising, so callers can
    treat this the same way as get_feature_importance().
    """
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        logger.error("scikit-learn is required for get_permutation_importance().")
        return {}

    if feature_names is None:
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]

    try:
        _validate_feature_alignment(np.zeros(X.shape[1]), feature_names)
    except ValueError as exc:
        logger.error("Permutation importance failed: %s", exc)
        return {}

    try:
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
        )
    except Exception as exc:  # model may not be fitted, incompatible X, etc.
        logger.error("Permutation importance computation failed: %s", exc)
        return {}

    series = pd.Series(result.importances_mean, index=feature_names)
    denom = series.abs().sum()
    normalized = series / denom if denom != 0 else series
    top = normalized.sort_values(ascending=False).head(top_n)

    logger.info("Computed permutation importance (%d repeats) for %d features.", n_repeats, len(top))
    return {name: round(float(score), 4) for name, score in top.items()}


# --------------------------------------------------------------------------- #
# 3. SHAP (global + local explanations)
# --------------------------------------------------------------------------- #

def compute_shap_values(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    background_samples: int = DEFAULT_CONFIG.shap_background_samples,
):
    """
    Compute SHAP values using shap.Explainer, which auto-selects the right
    explainer (Tree, Linear, Kernel, etc.) for the given model type.

    Returns a shap.Explanation object, or None if the `shap` package isn't
    installed or computation fails (e.g. unsupported model/data shape).
    """
    try:
        import shap
    except ImportError:
        logger.error("The 'shap' package is not installed. Run `pip install shap` to enable SHAP explanations.")
        return None

    if isinstance(X, pd.DataFrame) and feature_names is None:
        feature_names = list(X.columns)

    X_sample = X.sample(min(background_samples, len(X)), random_state=0) if isinstance(X, pd.DataFrame) and len(X) > background_samples else X

    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
    except Exception as exc:
        logger.warning("shap.Explainer failed (%s); falling back to KernelExplainer.", exc)
        try:
            background = shap.sample(X_sample, min(50, len(X_sample)))
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer(X_sample)
        except Exception as exc2:
            logger.error("SHAP computation failed entirely: %s", exc2)
            return None

    logger.info("Computed SHAP values for %d samples.", len(X_sample))
    return shap_values


def explain_prediction(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    index: int,
    feature_names: Optional[Sequence[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    config: ExplainabilityConfig = DEFAULT_CONFIG,
):
    """
    Local explanation for a single prediction (e.g. "explain prediction #17"),
    rendered as a SHAP waterfall plot.

    Returns the matplotlib Figure, or None if SHAP is unavailable / index is
    out of range.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("shap and matplotlib are required for explain_prediction().")
        return None

    if index < 0 or index >= len(X):
        logger.error("Index %d out of range for data of length %d.", index, len(X))
        return None

    shap_values = compute_shap_values(model, X, feature_names, background_samples=config.shap_background_samples)
    if shap_values is None:
        return None

    try:
        row_index = min(index, len(shap_values) - 1)
        fig = plt.figure()
        shap.plots.waterfall(shap_values[row_index], max_display=config.shap_max_display, show=False)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=config.figure_dpi, bbox_inches="tight")
            logger.info("Saved local explanation (prediction #%d) to %s", index, save_path)
        return fig
    except Exception as exc:
        logger.error("Failed to render SHAP waterfall plot: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# 4. Tabular output for frontends (Streamlit, etc.)
# --------------------------------------------------------------------------- #

def create_feature_dataframe(importance_dict: dict) -> pd.DataFrame:
    """
    Convert an {feature: importance} dict into a ranked DataFrame:

        Rank  Feature  Importance
        1     Age      0.34
        2     Salary   0.21

    Returns an empty DataFrame with the right columns if importance_dict is empty,
    so callers can safely call st.dataframe(...) either way.
    """
    if not importance_dict:
        return pd.DataFrame(columns=["Rank", "Feature", "Importance"])

    df = (
        pd.Series(importance_dict, name="Importance")
        .sort_values(ascending=False, key=abs)
        .rename_axis("Feature")
        .reset_index()
    )
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------- #
# 5. Plot + CSV export
# --------------------------------------------------------------------------- #

def save_feature_importance_plot(
    df: pd.DataFrame,
    filename: str = "feature_importance.png",
    config: ExplainabilityConfig = DEFAULT_CONFIG,
) -> Optional[Path]:
    """Bar chart of feature importances, saved to config.reports_dir. Returns the path, or None on failure."""
    if df.empty:
        logger.warning("Empty feature importance DataFrame; skipping plot generation.")
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for save_feature_importance_plot().")
        return None

    reports_dir = _ensure_reports_dir(config.reports_dir)
    out_path = reports_dir / filename

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df))))
    plot_df = df.sort_values("Importance")
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in plot_df["Importance"]]
    ax.barh(plot_df["Feature"], plot_df["Importance"], color=colors)
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.figure_dpi)
    plt.close(fig)

    logger.info("Saved feature importance plot to %s", out_path)
    return out_path


def save_shap_summary_plot(
    shap_values,
    X: Union[pd.DataFrame, np.ndarray],
    filename: str = "shap_summary.png",
    config: ExplainabilityConfig = DEFAULT_CONFIG,
) -> Optional[Path]:
    """Save a SHAP beeswarm/summary plot to config.reports_dir. Returns the path, or None on failure."""
    if shap_values is None:
        logger.warning("No SHAP values provided; skipping SHAP summary plot.")
        return None

    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("shap and matplotlib are required for save_shap_summary_plot().")
        return None

    reports_dir = _ensure_reports_dir(config.reports_dir)
    out_path = reports_dir / filename

    try:
        fig = plt.figure()
        shap.summary_plot(shap_values, X, max_display=config.shap_max_display, show=False)
        fig.savefig(out_path, dpi=config.figure_dpi, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.error("Failed to render/save SHAP summary plot: %s", exc)
        return None

    logger.info("Saved SHAP summary plot to %s", out_path)
    return out_path


def save_feature_importance_csv(
    df: pd.DataFrame,
    filename: str = "feature_importance.csv",
    config: ExplainabilityConfig = DEFAULT_CONFIG,
) -> Optional[Path]:
    """Export the ranked feature importance DataFrame as CSV. Returns the path, or None if df is empty."""
    if df.empty:
        logger.warning("Empty feature importance DataFrame; skipping CSV export.")
        return None

    reports_dir = _ensure_reports_dir(config.reports_dir)
    out_path = reports_dir / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved feature importance CSV to %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 6. High-level orchestrator
# --------------------------------------------------------------------------- #

def explain_model(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = DEFAULT_CONFIG.top_features,
    use_shap: bool = True,
    save_reports: bool = True,
    config: ExplainabilityConfig = DEFAULT_CONFIG,
) -> dict:
    """
    One-call pipeline: figure out the best available explanation method for
    `model`, compute it, and (optionally) persist plots + CSV to reports/.

    Resolution order:
        1. Native importance (tree / linear)      -> method = "tree" | "linear"
        2. Permutation importance (needs X and y)  -> method = "permutation"
        3. SHAP (if use_shap=True and shap installed) computed regardless, for
           richer visual explanations, whenever the earlier steps produced a
           usable result.

    Returns
    -------
    dict with keys:
        "feature_importance": dict[str, float]
        "feature_importance_df": pd.DataFrame
        "method": str
        "normalized": bool
        "shap_values": shap.Explanation | None
        "plot_path": Path | None
        "shap_plot_path": Path | None
        "csv_path": Path | None
    """
    if feature_names is None:
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]

    if hasattr(model, "feature_importances_"):
        method = "tree"
    elif hasattr(model, "coef_"):
        method = "linear"
    else:
        method = "unsupported"

    importance_dict = get_feature_importance(model, feature_names, top_n=top_n) if method != "unsupported" else {}

    if not importance_dict:
        if y is None:
            logger.warning(
                "Model has no native importance and no y was provided, so "
                "permutation importance can't be computed. Returning empty result."
            )
            method = "unsupported"
        else:
            method = "permutation"
            importance_dict = get_permutation_importance(model, X, y, feature_names, top_n=top_n)
            if not importance_dict:
                method = "unsupported"

    df = create_feature_dataframe(importance_dict)

    shap_values = None
    shap_plot_path = None
    if use_shap and not df.empty:
        shap_values = compute_shap_values(model, X, feature_names, background_samples=config.shap_background_samples)

    plot_path = None
    csv_path = None
    if save_reports and not df.empty:
        plot_path = save_feature_importance_plot(df, config=config)
        csv_path = save_feature_importance_csv(df, config=config)
        if shap_values is not None:
            # Use the exact rows SHAP was computed on (shap_values.data), since
            # compute_shap_values() may have subsampled X internally.
            shap_X = getattr(shap_values, "data", X)
            shap_plot_path = save_shap_summary_plot(shap_values, shap_X, config=config)

    return {
        "feature_importance": importance_dict,
        "feature_importance_df": df,
        "method": method,
        "normalized": True,
        "shap_values": shap_values,
        "plot_path": plot_path,
        "shap_plot_path": shap_plot_path,
        "csv_path": csv_path,
    }
