"""
Defines candidate models per problem type and runs the AutoML training loop:
build pipeline -> cross-validate -> (optionally) tune -> fit -> time -> evaluate
-> leaderboard -> select best.

Scaling/encoding live in preprocessing.py, not here. This file accepts an
optional sklearn-compatible `preprocessor` (e.g. a ColumnTransformer) and
wraps every model in a Pipeline([("preprocessor", preprocessor), ("model", model)]).
If no preprocessor is given, models train directly on X — useful for
already-encoded input or for models that don't need scaling (tree ensembles).
"""

import logging
import time
from typing import Callable, Optional
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from services import evaluation as ev

logger = logging.getLogger(__name__)

try:
    from utils.config import Config
    RANDOM_STATE = Config.RANDOM_STATE
except ImportError:
    RANDOM_STATE = 42

CV_FOLDS = 5
CV_SCORING = {"classification": "accuracy", "regression": "r2"}

# Models that support class_weight="balanced" for imbalanced classification targets
CLASS_WEIGHT_CAPABLE = {"Logistic Regression", "Random Forest", "Extra Trees",
                         "Decision Tree", "SVM"}

# Small, cheap search spaces for optional hyperparameter tuning.
# Kept intentionally narrow (RandomizedSearchCV with low n_iter) so AutoML
# stays fast; widen these once you have async/background training.
PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 8, 16, 24],
        "model__min_samples_split": [2, 5, 10],
    },
    "Extra Trees": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 8, 16, 24],
        "model__min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
    },
    "Decision Tree": {
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Candidate models
# ─────────────────────────────────────────────────────────────────────────
def get_candidate_models(problem_type: str, class_weight: Optional[str] = None) -> dict:
    """Return {model_name: unfitted estimator} for the given problem type."""
    cw = class_weight  # None or "balanced"

    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight=cw),
            "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight=cw),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=cw),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, class_weight=cw),
            "Naive Bayes": GaussianNB(),
        }

    if problem_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
            "Extra Trees": ExtraTreesRegressor(random_state=RANDOM_STATE),
            "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "SVR": SVR(),
        }

    if problem_type == "clustering":
        return {"KMeans": KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)}

    raise ValueError(f"Unsupported problem type: {problem_type}")


# ─────────────────────────────────────────────────────────────────────────
# Pipeline construction
# ─────────────────────────────────────────────────────────────────────────
def build_pipeline(model, preprocessor=None) -> Pipeline:
    """Wrap a model in a Pipeline with the shared preprocessor, if one is given."""
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps)


# ─────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────
def cross_validate_model(pipeline: Pipeline, X, y, problem_type: str, cv: int = CV_FOLDS) -> dict:
    """Return {'mean': float, 'std': float} from k-fold cross-validation."""
    scoring = CV_SCORING[problem_type]
    n_splits = min(cv, len(X)) if len(X) < cv else cv
    if n_splits < 2:
        return {"mean": None, "std": None}

    scores = cross_val_score(pipeline, X, y, cv=n_splits, scoring=scoring, error_score="raise")
    return {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}


# ─────────────────────────────────────────────────────────────────────────
# Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────
def hyperparameter_search(
    pipeline: Pipeline, model_name: str, X, y, problem_type: str,
    n_iter: int = 10, cv: int = 3,
) -> Pipeline:
    """
    Run RandomizedSearchCV if a param grid is defined for this model, otherwise
    return the pipeline untouched. Cheap by design — small n_iter, few folds.
    """
    grid = PARAM_GRIDS.get(model_name)
    if not grid:
        return pipeline

    scoring = CV_SCORING[problem_type]
    search = RandomizedSearchCV(
        pipeline, param_distributions=grid, n_iter=n_iter, cv=cv,
        scoring=scoring, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    logger.info("Tuned %s -> best CV %s=%.4f, params=%s",
                model_name, scoring, search.best_score_, search.best_params_)
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────────────────
# Single-model training
# ─────────────────────────────────────────────────────────────────────────
def train_single_model(
    name: str, model, X_train, X_test, y_train, y_test, problem_type: str,
    preprocessor=None, cv: int = CV_FOLDS, tune: bool = False,
) -> Optional[dict]:
    """
    Build the pipeline, optionally tune it, cross-validate, fit, time everything,
    and evaluate on the held-out test set. Returns None (and logs a warning)
    if this model fails, so one bad model can't take down the whole run.
    """
    try:
        pipeline = build_pipeline(model, preprocessor)

        if tune:
            pipeline = hyperparameter_search(pipeline, name, X_train, y_train, problem_type)

        cv_result = cross_validate_model(pipeline, X_train, y_train, problem_type, cv=cv)

        train_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = round(time.perf_counter() - train_start, 4)

        predict_start = time.perf_counter()
        predictions = pipeline.predict(X_test)
        predict_time = round(time.perf_counter() - predict_start, 4)

        metrics = ev.evaluate(y_test, predictions, problem_type)

        logger.info("Trained %-20s | train %.3fs | predict %.4fs | %s",
                    name, train_time, predict_time, metrics)

        return {
            "name": name,
            "pipeline": pipeline,
            "metrics": metrics,
            "cv_mean": cv_result["mean"],
            "cv_std": cv_result["std"],
            "train_time_sec": train_time,
            "predict_time_sec": predict_time,
            "feature_count": X_train.shape[1],
            "params": pipeline.named_steps["model"].get_params(),
        }

    except Exception as e:
        logger.warning("Skipping %s — training failed: %s", name, e)
        return None


# ─────────────────────────────────────────────────────────────────────────
# Train every candidate
# ─────────────────────────────────────────────────────────────────────────
def train_all_models(
    X_train, X_test, y_train, y_test, problem_type: str,
    preprocessor=None, cv: int = CV_FOLDS, tune: bool = False,
    class_weight: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """
    Train every candidate model for `problem_type`, skipping any that fail.
    `progress_callback(current, total, model_name)` lets the frontend show a
    live progress bar, e.g. "Training Random Forest… 3/8".
    """
    candidates = get_candidate_models(problem_type, class_weight=class_weight)
    total = len(candidates)
    results = []

    for i, (name, model) in enumerate(candidates.items(), start=1):
        if progress_callback:
            progress_callback(i, total, name)

        result = train_single_model(
            name, model, X_train, X_test, y_train, y_test, problem_type,
            preprocessor=preprocessor, cv=cv, tune=tune,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(f"All candidate models failed to train for problem_type={problem_type}")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────
def create_leaderboard(results: list[dict], problem_type: str) -> pd.DataFrame:
    """Build a leaderboard DataFrame sorted best-to-worst on the primary metric."""
    primary_metric = (
        ev.CLASSIFICATION_PRIMARY_METRIC if problem_type == "classification"
        else ev.REGRESSION_PRIMARY_METRIC
    )

    rows = []
    for r in results:
        row = {"Model": r["name"], **r["metrics"]}
        row["CV Mean"] = r["cv_mean"]
        row["CV Std"] = r["cv_std"]
        row["Training Time (s)"] = r["train_time_sec"]
        row["Inference Time (s)"] = r["predict_time_sec"]
        rows.append(row)

    board = pd.DataFrame(rows)
    board = board.sort_values(by=primary_metric, ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", board.index + 1)
    return board


# ─────────────────────────────────────────────────────────────────────────
# Best-model selection
# ─────────────────────────────────────────────────────────────────────────
def select_best_model(results: list[dict], leaderboard: pd.DataFrame) -> dict:
    """Pick the #1 leaderboard row and return its full result dict."""
    best_name = leaderboard.iloc[0]["Model"]
    return next(r for r in results if r["name"] == best_name)


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────
def train_and_compare(
    X_train, X_test, y_train, y_test, problem_type: str,
    preprocessor=None, cv: int = CV_FOLDS, tune: bool = False,
    class_weight: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Train every candidate model, build a leaderboard, and return the best one
    alongside full comparison data for reporting/explainability.
    """
    results = train_all_models(
        X_train, X_test, y_train, y_test, problem_type,
        preprocessor=preprocessor, cv=cv, tune=tune,
        class_weight=class_weight, progress_callback=progress_callback,
    )

    leaderboard = create_leaderboard(results, problem_type)
    best = select_best_model(results, leaderboard)

    return {
        "leaderboard": leaderboard,
        "results": results,
        "best_model_name": best["name"],
        "best_model": best["pipeline"],
        "cv_score": best["cv_mean"],
        "training_time_sec": best["train_time_sec"],
        "prediction_time_sec": best["predict_time_sec"],
        "feature_count": best["feature_count"],
    }
