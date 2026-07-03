"""
Feature engineering for the ML pipeline.

Responsibility boundary:
  preprocessing.py    -> validation, train/test split, encoding (OneHotEncoder),
                          scaling (StandardScaler) inside the modeling pipeline.
  feature_engineering.py (this file) -> everything that shapes *which* features
                          exist before that pipeline ever sees the data:
                          ID removal, datetime expansion, boolean normalization,
                          constant/low-variance/correlated feature pruning,
                          rare-category consolidation, optional interaction
                          features, and optional feature selection.

This file intentionally does NOT one-hot encode or scale anything — that stays
in preprocessing.py so the two modules don't do the same work twice.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression,
)

logger = logging.getLogger(__name__)

# ── Default thresholds (override via function args; move to config later) ──
DEFAULT_ID_NAME_HINTS = ("id", "_id", "index", "unnamed")
DEFAULT_DATE_PARSE_SUCCESS_RATIO = 0.9
DEFAULT_RARE_CATEGORY_THRESHOLD = 0.01       # categories below 1% frequency -> "Other"
DEFAULT_LOW_VARIANCE_THRESHOLD = 0.01
DEFAULT_CORRELATION_THRESHOLD = 0.95


# ─────────────────────────────────────────────────────────────────────────
# ID removal
# ─────────────────────────────────────────────────────────────────────────
def remove_id_columns(df: pd.DataFrame, name_hints=DEFAULT_ID_NAME_HINTS) -> pd.DataFrame:
    """Drop columns that look like identifiers rather than predictive features."""

    def is_id_column(col: str) -> bool:
        if any(hint in col.strip().lower() for hint in name_hints):
            return True
        # Fully-unique text/integer columns are usually identifiers;
        # fully-unique float columns are usually genuine continuous features.
        is_discrete = (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_integer_dtype(df[col])
        )
        return is_discrete and df[col].nunique() == len(df)

    id_cols = [col for col in df.columns if is_id_column(col)]
    if id_cols:
        logger.info("Removed ID columns: %s", id_cols)
    return df.drop(columns=id_cols, errors="ignore")


# ─────────────────────────────────────────────────────────────────────────
# Datetime feature extraction
# ─────────────────────────────────────────────────────────────────────────
def extract_datetime_features(
    df: pd.DataFrame,
    parse_success_ratio: float = DEFAULT_DATE_PARSE_SUCCESS_RATIO,
    cyclic: bool = True,
) -> pd.DataFrame:
    """
    Detect object columns that are really dates and expand them into
    year / month / day / weekday / quarter / week_of_year / is_weekend /
    is_month_start / is_month_end. Adds sin/cos cyclic encodings for month
    and weekday when `cyclic=True`, which captures seasonality better than
    raw integers (December and January are 1 month apart, not 11).
    """
    df = df.copy()
    generated_cols = []

    for col in df.columns:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue

        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() <= parse_success_ratio:
            continue

        df[f"{col}_year"] = parsed.dt.year
        df[f"{col}_month"] = parsed.dt.month
        df[f"{col}_day"] = parsed.dt.day
        df[f"{col}_weekday"] = parsed.dt.weekday
        df[f"{col}_quarter"] = parsed.dt.quarter
        df[f"{col}_week_of_year"] = parsed.dt.isocalendar().week.astype("int32")
        df[f"{col}_is_weekend"] = (parsed.dt.weekday >= 5).astype(int)
        df[f"{col}_is_month_start"] = parsed.dt.is_month_start.astype(int)
        df[f"{col}_is_month_end"] = parsed.dt.is_month_end.astype(int)

        new_cols = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday",
                    f"{col}_quarter", f"{col}_week_of_year", f"{col}_is_weekend",
                    f"{col}_is_month_start", f"{col}_is_month_end"]

        if cyclic:
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * parsed.dt.month / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * parsed.dt.month / 12)
            df[f"{col}_weekday_sin"] = np.sin(2 * np.pi * parsed.dt.weekday / 7)
            df[f"{col}_weekday_cos"] = np.cos(2 * np.pi * parsed.dt.weekday / 7)
            new_cols += [f"{col}_month_sin", f"{col}_month_cos",
                         f"{col}_weekday_sin", f"{col}_weekday_cos"]

        df = df.drop(columns=[col])
        generated_cols.extend(new_cols)

    if generated_cols:
        logger.info("Expanded %d datetime column(s) into: %s",
                     len(generated_cols), generated_cols)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Boolean normalization
# ─────────────────────────────────────────────────────────────────────────
def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean-dtype columns to 1/0 so models treat them as numeric."""
    df = df.copy()
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]

    for col in bool_cols:
        df[col] = df[col].astype("Int64")

    if bool_cols:
        logger.info("Converted boolean columns to 1/0: %s", bool_cols)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Rare category consolidation (NOT encoding — just merges long-tail labels)
# ─────────────────────────────────────────────────────────────────────────
def handle_rare_categories(
    df: pd.DataFrame, threshold: float = DEFAULT_RARE_CATEGORY_THRESHOLD
) -> pd.DataFrame:
    """Merge categorical values that appear in fewer than `threshold` fraction of rows into 'Other'."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    affected = []

    for col in cat_cols:
        freqs = df[col].value_counts(normalize=True)
        rare_values = freqs[freqs < threshold].index
        if len(rare_values) > 0:
            df[col] = df[col].where(~df[col].isin(rare_values), "Other")
            affected.append(col)

    if affected:
        logger.info("Merged rare categories into 'Other' for: %s", affected)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Constant / low-variance / correlated feature pruning
# ─────────────────────────────────────────────────────────────────────────
def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with only a single unique value — they carry zero predictive signal."""
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        logger.info("Removed constant columns: %s", constant_cols)
    return df.drop(columns=constant_cols, errors="ignore")


def remove_low_variance_features(
    df: pd.DataFrame, threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD
) -> pd.DataFrame:
    """Drop numeric columns whose variance falls below `threshold` using sklearn's VarianceThreshold."""
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] == 0:
        return df

    selector = VarianceThreshold(threshold=threshold)
    try:
        selector.fit(numeric_df.fillna(numeric_df.mean()))
    except ValueError:
        # All numeric columns removed or no variance to compute against
        return df

    kept_numeric = numeric_df.columns[selector.get_support()]
    dropped = set(numeric_df.columns) - set(kept_numeric)
    if dropped:
        logger.info("Removed low-variance columns (threshold=%s): %s", threshold, sorted(dropped))

    non_numeric_cols = df.columns.difference(numeric_df.columns)
    return df[list(non_numeric_cols) + list(kept_numeric)]


def remove_correlated_features(
    df: pd.DataFrame, threshold: float = DEFAULT_CORRELATION_THRESHOLD
) -> pd.DataFrame:
    """
    Drop numeric columns that are highly correlated (|r| > threshold) with another
    column already kept, so the model isn't fed near-duplicate signals.
    """
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return df

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info("Removed highly correlated columns (threshold=%s): %s", threshold, to_drop)
    return df.drop(columns=to_drop, errors="ignore")


# ─────────────────────────────────────────────────────────────────────────
# Optional: interaction features
# ─────────────────────────────────────────────────────────────────────────
def generate_interaction_features(
    df: pd.DataFrame, pairs: Optional[list[tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Optionally create product/ratio features from explicit column pairs, e.g.
    pairs=[("Age", "Income")] adds "Age_x_Income" and "Age_div_Income".
    Disabled by default (pairs=None) to avoid exploding the feature space.
    """
    if not pairs:
        return df

    df = df.copy()
    generated = []
    for col_a, col_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        if not (pd.api.types.is_numeric_dtype(df[col_a]) and pd.api.types.is_numeric_dtype(df[col_b])):
            continue

        prod_name, ratio_name = f"{col_a}_x_{col_b}", f"{col_a}_div_{col_b}"
        df[prod_name] = df[col_a] * df[col_b]
        df[ratio_name] = df[col_a] / df[col_b].replace(0, np.nan)
        generated += [prod_name, ratio_name]

    if generated:
        logger.info("Generated interaction features: %s", generated)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Optional: outlier capping (winsorizing) via IQR clipping
# ─────────────────────────────────────────────────────────────────────────
def cap_outliers_iqr(df: pd.DataFrame, columns: Optional[list[str]] = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Clip numeric values to [Q1 - factor*IQR, Q3 + factor*IQR] instead of dropping rows.
    If `columns` is None, applies to all numeric columns. Disabled by default in
    `engineer_features` — call explicitly when you want it.
    """
    df = df.copy()
    numeric_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[col] = df[col].clip(lower, upper)

    return df


# ─────────────────────────────────────────────────────────────────────────
# Optional: feature selection
# ─────────────────────────────────────────────────────────────────────────
def select_best_features(
    X: pd.DataFrame, y: pd.Series, problem_type: str, k: int = 20
) -> pd.DataFrame:
    """
    Keep the top-k features most informative about the target, using mutual
    information. Only run this on already-numeric/encoded data — call it after
    preprocessing.py's encoding step, not before. Returns X unchanged if it
    already has <= k columns.
    """
    if X.shape[1] <= k:
        return X

    numeric_X = X.select_dtypes(include=np.number).fillna(0)
    if numeric_X.shape[1] == 0:
        return X

    scorer = mutual_info_classif if problem_type == "classification" else mutual_info_regression
    selector = SelectKBest(score_func=scorer, k=min(k, numeric_X.shape[1]))
    selector.fit(numeric_X, y)

    selected_cols = numeric_X.columns[selector.get_support()]
    non_numeric_cols = X.columns.difference(numeric_X.columns)
    kept = list(non_numeric_cols) + list(selected_cols)

    logger.info("Feature selection kept %d/%d columns", len(kept), X.shape[1])
    return X[kept]


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────
def engineer_features(
    df: pd.DataFrame,
    target: str,
    *,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
    rare_category_threshold: float = DEFAULT_RARE_CATEGORY_THRESHOLD,
    interaction_pairs: Optional[list[tuple[str, str]]] = None,
    select_k: Optional[int] = None,
    problem_type: Optional[str] = None,
) -> dict:
    """
    Run the full feature engineering pipeline on the feature matrix (target excluded).

    Returns a metadata dict rather than a bare dataframe, so callers can report
    on and explain what happened:
        {
            "data": <engineered dataframe>,
            "removed_columns": [...],
            "generated_columns": [...],
            "selected_features": [...] | None,
        }
    """
    original_cols = set(df.columns) - {target}
    features = df.drop(columns=[target])

    features = extract_datetime_features(features)
    features = remove_id_columns(features)
    features = convert_boolean_columns(features)
    features = handle_rare_categories(features, threshold=rare_category_threshold)
    features = remove_constant_features(features)
    features = remove_low_variance_features(features, threshold=low_variance_threshold)
    features = remove_correlated_features(features, threshold=correlation_threshold)
    features = generate_interaction_features(features, pairs=interaction_pairs)
    features = features.fillna(0)

    selected_features = None
    if select_k is not None and problem_type is not None:
        y = df[target]
        features = select_best_features(features, y, problem_type, k=select_k)
        selected_features = list(features.columns)

    generated_columns = [c for c in features.columns if c not in original_cols]
    removed_columns = [c for c in original_cols if c not in features.columns]

    return {
        "data": features,
        "removed_columns": removed_columns,
        "generated_columns": generated_columns,
        "selected_features": selected_features,
    }
