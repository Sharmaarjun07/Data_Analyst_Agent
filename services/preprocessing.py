"""
Industry-Level Data Preprocessing Module

Features
--------
- Dataset validation
- Automatic target detection
- Problem type detection
- Feature type detection
- Datetime conversion
- ID column detection
- Target validation
- Class imbalance detection
- Train/Test splitting
- Sklearn preprocessing pipeline
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

CLASSIFICATION_MAX_UNIQUE_RATIO = 0.05
CLASSIFICATION_MAX_UNIQUE_ABS = 20

TARGET_NAME_HINTS = {
    "target",
    "label",
    "class",
    "y",
    "price",
    "salary",
    "revenue",
    "score",
    "result",
    "churn",
    "outcome",
}

ID_COLUMN_HINTS = {
    "id",
    "customer_id",
    "order_id",
    "emp_id",
    "employee_id",
    "uuid",
    "index",
}

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ============================================================
# Custom Exceptions
# ============================================================


class InvalidDatasetError(Exception):
    """Raised when dataset is invalid."""


class TargetNotFoundError(Exception):
    """Raised when target column cannot be detected."""


class UnsupportedTargetError(Exception):
    """Raised when target is unsuitable for ML."""


# ============================================================
# Data Preprocessor
# ============================================================


class DataPreprocessor:
    """
    Complete preprocessing engine.
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):

        self.test_size = test_size
        self.random_state = random_state

    # =======================================================
    # Dataset Validation
    # =======================================================

    def validate_dataset(self, df: pd.DataFrame) -> None:

        if df.empty:
            raise InvalidDatasetError("Dataset is empty.")

        if len(df) < 20:
            raise InvalidDatasetError(
                "Dataset must contain at least 20 rows."
            )

        if len(df.columns) < 2:
            raise InvalidDatasetError(
                "Dataset must contain at least two columns."
            )

        duplicated_columns = df.columns[df.columns.duplicated()]

        if len(duplicated_columns):
            raise InvalidDatasetError(
                f"Duplicate columns found: {list(duplicated_columns)}"
            )

        if df.duplicated().any():
            logger.warning("Dataset contains duplicate rows.")

    # =======================================================
    # Target Detection
    # =======================================================

    def detect_target_column(
        self,
        df: pd.DataFrame,
        user_target: str | None = None,
    ) -> str:

        if user_target:

            if user_target not in df.columns:
                raise TargetNotFoundError(
                    f"{user_target} not found."
                )

            return user_target

        for column in df.columns:

            if column.lower().strip() in TARGET_NAME_HINTS:
                return column

        last_column = df.columns[-1]

        unique_ratio = (
            df[last_column].nunique(dropna=True) / len(df)
        )

        if unique_ratio < 0.90:
            return last_column

        raise TargetNotFoundError(
            "Unable to detect target column."
        )

    # =======================================================
    # Target Validation
    # =======================================================

    def validate_target(
        self,
        df: pd.DataFrame,
        target: str,
    ) -> None:

        series = df[target]

        if series.isna().all():
            raise UnsupportedTargetError(
                "Target contains only missing values."
            )

        if series.nunique(dropna=True) <= 1:
            raise UnsupportedTargetError(
                "Target contains only one unique value."
            )

    # =======================================================
    # Problem Type Detection
    # =======================================================

    def detect_problem_type(
        self,
        df: pd.DataFrame,
        target: str,
    ) -> str:

        y = df[target]

        if (
            y.dtype == object
            or str(y.dtype) == "category"
            or str(y.dtype) == "bool"
        ):
            return "classification"

        unique_count = y.nunique()

        unique_ratio = unique_count / len(y)

        if (
            unique_ratio
            <= CLASSIFICATION_MAX_UNIQUE_RATIO
            or unique_count
            <= CLASSIFICATION_MAX_UNIQUE_ABS
        ):
            return "classification"

        return "regression"

    # =======================================================
    # Feature / Target Split
    # =======================================================

    @staticmethod
    def split_features_target(
        df: pd.DataFrame,
        target: str,
    ) -> tuple[pd.DataFrame, pd.Series]:

        X = df.drop(columns=[target])

        y = df[target]

        return X, y

    # =======================================================
    # Feature Type Detection
    # =======================================================

    @staticmethod
    def detect_feature_types(
        X: pd.DataFrame,
    ) -> dict[str, list[str]]:

        numerical = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        boolean = X.select_dtypes(
            include="bool"
        ).columns.tolist()

        datetime = X.select_dtypes(
            include=["datetime64[ns]"]
        ).columns.tolist()

        return {

            "numerical_columns": numerical,

            "categorical_columns": categorical,

            "boolean_columns": boolean,

            "datetime_columns": datetime,
        }

    # =======================================================
    # Datetime Conversion
    # =======================================================

    @staticmethod
    def convert_datetime_columns(
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        X = X.copy()

        for column in X.columns:

            if X[column].dtype != object:
                continue

            try:

                converted = pd.to_datetime(
                    X[column],
                    errors="raise",
                )

                if converted.notna().sum() > len(X) * 0.8:
                    X[column] = converted

            except Exception:
                continue

        return X

    # =======================================================
    # Detect ID Columns
    # =======================================================

    @staticmethod
    def detect_id_columns(
        X: pd.DataFrame,
    ) -> list[str]:

        ids = []

        for column in X.columns:

            name = column.lower()

            if name in ID_COLUMN_HINTS:
                ids.append(column)
                continue

            unique_ratio = (
                X[column].nunique(dropna=True)
                / len(X)
            )

            if unique_ratio >= 0.98:
                ids.append(column)

        return ids

    # =======================================================
    # Class Imbalance Detection
    # =======================================================

    @staticmethod
    def detect_class_imbalance(
        y: pd.Series,
    ) -> bool:

        distribution = (
            y.value_counts(normalize=True)
        )

        return distribution.max() > 0.95
    
    # =======================================================
    # Build Preprocessing Pipeline
    # =======================================================

    def build_preprocessor(
        self,
        numerical_columns: list[str],
        categorical_columns: list[str],
    ) -> ColumnTransformer:
        """
        Build sklearn preprocessing pipeline.
        """

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    numerical_columns,
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

        return preprocessor

    # =======================================================
    # Train Test Split
    # =======================================================

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ):

        stratify = (
            y
            if (
                problem_type == "classification"
                and y.nunique() > 1
            )
            else None
        )

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    # =======================================================
    # Complete Dataset Preparation
    # =======================================================

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        user_target: str | None = None,
    ) -> dict[str, Any]:
        """
        Complete preprocessing workflow.
        """

        logger.info("Starting preprocessing...")

        self.validate_dataset(df)

        target = self.detect_target_column(
            df,
            user_target,
        )

        self.validate_target(df, target)

        problem_type = self.detect_problem_type(
            df,
            target,
        )

        logger.info(
            "Detected problem type: %s",
            problem_type,
        )

        X, y = self.split_features_target(
            df,
            target,
        )

        X = self.convert_datetime_columns(X)

        feature_info = self.detect_feature_types(X)

        id_columns = self.detect_id_columns(X)

        if id_columns:

            logger.info(
                "Dropping ID columns: %s",
                id_columns,
            )

            X = X.drop(columns=id_columns)

            feature_info = self.detect_feature_types(X)

        preprocessor = self.build_preprocessor(
            feature_info["numerical_columns"],
            feature_info["categorical_columns"],
        )

        X_train, X_test, y_train, y_test = self.split_data(
            X,
            y,
            problem_type,
        )

        is_imbalanced = False

        if problem_type == "classification":
            is_imbalanced = (
                self.detect_class_imbalance(
                    y_train,
                )
            )

        logger.info(
            "Training samples: %d",
            len(X_train),
        )

        logger.info(
            "Testing samples: %d",
            len(X_test),
        )

        logger.info("Preprocessing completed.")

        return {

            # Split data
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,

            # Original
            "X": X,
            "y": y,

            # Metadata
            "target": target,
            "problem_type": problem_type,

            "numerical_columns":
                feature_info["numerical_columns"],

            "categorical_columns":
                feature_info["categorical_columns"],

            "boolean_columns":
                feature_info["boolean_columns"],

            "datetime_columns":
                feature_info["datetime_columns"],

            "id_columns": id_columns,

            "is_imbalanced": is_imbalanced,

            # Pipeline
            "preprocessor": preprocessor,
        }


# ============================================================
# Convenience Function
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
    target: str | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper around DataPreprocessor.
    """

    processor = DataPreprocessor()

    return processor.prepare_dataset(
        df=df,
        user_target=target,
    )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    sample = pd.DataFrame(
        {
            "age": [22, 34, 45, 28, 39],
            "salary": [25000, 54000, 76000, 42000, 62000],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
            ],
            "target": [0, 1, 1, 0, 1],
        }
    )

    data = prepare_dataset(sample)

    print(data["problem_type"])
    print(data["target"])
    print(data["numerical_columns"])
    print(data["categorical_columns"])