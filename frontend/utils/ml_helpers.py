"""
Frontend ML Helpers
===================
Shared utilities for ML-frontend integration
"""

import os
import sys
import logging
import pandas as pd
import streamlit as st
from typing import Dict, Any, Tuple, Optional

# Setup logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_session_state():
    """Initialize all required session state variables"""
    
    defaults = {
        # Data
        "df": None,
        "original_df": None,
        "cleaned_df": None,
        "metadata": None,
        
        # ML Pipeline
        "target_column": None,
        "models": None,
        "best_model": None,
        "pipeline_data": None,
        
        # Results
        "predictions": None,
        "evaluation_report": None,
        "explainability_results": None,
        
        # UI State
        "current_page": "home",
        "processing": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# DATA MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, name: str, location: str = "uploads/processed") -> str:
    """Save processed dataframe to disk"""
    
    os.makedirs(location, exist_ok=True)
    file_path = os.path.join(location, f"{name}.csv")
    
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved dataframe: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save dataframe: {e}")
        raise


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load dataframe from disk"""
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataframe: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataframe: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# ML SERVICE WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Wrapper for run_ml_pipeline with error handling
    
    Returns:
        (results_dict, error_message) - One will be None
    """
    
    try:
        from services.ml_service import run_ml_pipeline
        
        results = run_ml_pipeline(
            df=df,
            user_target=target_col,
            test_ratio=test_size,
            random_state=random_state,
        )
        
        return results, None
    
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def safe_make_predictions(
    model,
    data: pd.DataFrame,
    batch: bool = True
) -> Tuple[Optional[list], Optional[str]]:
    """
    Wrapper for prediction functions with error handling
    
    Returns:
        (predictions, error_message) - One will be None
    """
    
    try:
        if batch:
            from services.prediction import predict_batch
            predictions = predict_batch(model, data)
        else:
            from services.prediction import predict_single
            predictions = predict_single(model, data)
        
        return predictions, None
    
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def display_metrics(metrics_dict: Dict[str, Any], ncols: int = 4):
    """Display metrics in columns"""
    
    if not metrics_dict:
        st.warning("No metrics to display")
        return
    
    cols = st.columns(ncols)
    
    for col, (metric_name, metric_val) in zip(
        cols * (len(metrics_dict) // ncols + 1),
        metrics_dict.items()
    ):
        with col:
            if isinstance(metric_val, float):
                st.metric(metric_name, f"{metric_val:.4f}")
            else:
                st.metric(metric_name, metric_val)


def display_dataframe_comparison(df1: pd.DataFrame, df2: pd.DataFrame, labels: Tuple[str, str]):
    """Display two dataframes side-by-side"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{labels[0]}**")
        st.dataframe(df1, use_container_width=True)
    
    with col2:
        st.write(f"**{labels[1]}**")
        st.dataframe(df2, use_container_width=True)


def display_model_card(model_info: Dict[str, Any]):
    """Display model information in a card"""
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model",
                model_info.get("name", "Unknown"),
                help="Model type"
            )
        
        with col2:
            st.metric(
                "Score",
                f"{model_info.get('score', 0):.4f}",
                help="Performance metric"
            )
        
        with col3:
            st.metric(
                "Time",
                f"{model_info.get('time', 0):.2f}s",
                help="Training time"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def handle_error(error: Exception, context: str = "", show_traceback: bool = True):
    """Display error to user with optional traceback"""
    
    error_msg = f"❌ {context}: {str(error)}" if context else f"❌ {str(error)}"
    st.error(error_msg)
    
    if show_traceback:
        with st.expander("🔍 Debug Info"):
            import traceback
            st.code(traceback.format_exc())
    
    logger.error(f"{context}: {str(error)}")


# ─────────────────────────────────────────────────────────────────────────────
# FILE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def list_saved_models(model_dir: str = "models") -> list:
    """List all saved models"""
    
    if not os.path.exists(model_dir):
        return []
    
    return [f for f in os.listdir(model_dir) if f.endswith((".pkl", ".joblib"))]


def list_saved_reports(report_dir: str = "reports") -> list:
    """List all saved reports"""
    
    if not os.path.exists(report_dir):
        return []
    
    return [f for f in os.listdir(report_dir) if f.endswith((".html", ".pdf", ".json"))]


def list_processed_data(data_dir: str = "uploads/processed") -> list:
    """List all processed datasets"""
    
    if not os.path.exists(data_dir):
        return []
    
    return [f for f in os.listdir(data_dir) if f.endswith(".csv")]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate dataframe for ML pipeline"""
    
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if df.shape[0] < 10:
        return False, "Need at least 10 rows"
    
    if df.shape[1] < 2:
        return False, "Need at least 2 columns"
    
    return True, "Valid"


def validate_target_column(df: pd.DataFrame, target: str) -> Tuple[bool, str]:
    """Validate target column selection"""
    
    if target not in df.columns:
        return False, f"Target column '{target}' not in dataframe"
    
    if df[target].isnull().sum() > df.shape[0] * 0.5:
        return False, "Target column has >50% missing values"
    
    if df[target].nunique() < 2:
        return False, "Target column has <2 unique values"
    
    return True, "Valid"


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_number(num: float, decimals: int = 4) -> str:
    """Format number with specified decimals"""
    return f"{num:.{decimals}f}"


def format_dataframe(df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    """Format dataframe for display"""
    return df.head(max_rows)


def get_dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for dataframe"""
    
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
