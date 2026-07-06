"""
ML Training Page
----------------
Frontend page for training Machine Learning models.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from frontend.utils.api import (
    APIError,
    check_backend,
    train_model,
)

# ==========================================================
# Helper Functions
# ==========================================================


def _show_backend_status():

    if check_backend():

        st.success("🟢 Backend Connected")

    else:

        st.error("🔴 Backend Offline")

        st.stop()


def _validate_dataset():

    if "cleaned_df" not in st.session_state:

        st.warning(
            "Please clean a dataset before training."
        )

        st.stop()

    return st.session_state.cleaned_df


def _dataset_summary(df: pd.DataFrame):

    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Rows",
            len(df),
        )

    with c2:

        st.metric(
            "Columns",
            len(df.columns),
        )

    with c3:

        memory = round(
            df.memory_usage(deep=True).sum() / 1024,
            2,
        )

        st.metric(
            "Memory",
            f"{memory} KB",
        )

    st.dataframe(
        df.head(),
        use_container_width=True,
    )


def _training_options(df: pd.DataFrame):

    st.subheader("Training Configuration")

    c1, c2 = st.columns(2)

    with c1:

        auto_target = st.checkbox(
            "Automatically detect target column",
            value=True,
        )

        target = st.selectbox(
            "Target Column",
            options=df.columns,
            disabled=auto_target,
        )

    with c2:

        problem_type = st.selectbox(
            "Problem Type",
            [
                "Auto Detect",
                "Classification",
                "Regression",
            ],
        )

        cv = st.slider(
            "Cross Validation Folds",
            3,
            10,
            5,
        )

    advanced = st.expander(
        "Advanced Options",
        expanded=False,
    )

    with advanced:

        random_state = st.number_input(
            "Random State",
            value=42,
        )

        test_size = st.slider(
            "Test Size",
            0.1,
            0.4,
            0.2,
            step=0.05,
        )

    return {
        "auto_target": auto_target,
        "target": target,
        "problem_type": problem_type,
        "cv": cv,
        "random_state": random_state,
        "test_size": test_size,
    }


# ==========================================================
# Main Render Function
# ==========================================================


def render():

    st.title("🤖 Machine Learning Engine")

    st.caption(
        "Automatically train multiple machine learning models "
        "and choose the best performer."
    )

    _show_backend_status()

    df = _validate_dataset()

    _dataset_summary(df)

    options = _training_options(df)

    st.divider()

    train = st.button(
        "🚀 Train Models",
        use_container_width=True,
        type="primary",
    )

    if not train:
        return

    target = None

    if not options["auto_target"]:
        target = options["target"]

    progress = st.progress(
    0,
    text="Preparing dataset...",
)

    try:

        progress.progress(
        20,
        text="Sending cleaned dataset...",
    )

        with st.spinner(
         "Training Machine Learning models..."
    ):

            result = train_model(
                df,
                target_column=target,
        )

        progress.progress(
          100,
          text="Training Completed",
        )

        st.session_state["ml_result"] = result

    except APIError as e:
        progress.empty()
        st.error(str(e))
        return

    except Exception as e:
        progress.empty()
        st.exception(e)
        return
    st.success(
    "✅ Training Finished Successfully!"
    )
    # ==========================================================
    # Training Results
    # ==========================================================

    result = st.session_state.get("ml_result")

    if result is None:
        return

    st.divider()

    st.header("🏆 Training Results")

    # ----------------------------------------------------------
    # Best Model Summary
    # ----------------------------------------------------------

    best_model = result.get("best_model", "Unknown")
    best_score = result.get("best_score", None)
    problem_type = result.get("problem_type", "Unknown")
    target_column = result.get("target_column", "Unknown")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Best Model",
            best_model,
        )

    with c2:

        if best_score is not None:
            st.metric(
                "Best Score",
                round(float(best_score), 4),
            )
        else:
            st.metric(
                "Best Score",
                "N/A",
            )

    with c3:
        st.metric(
            "Problem",
            problem_type,
        )

    with c4:
        st.metric(
            "Target",
            target_column,
        )

    # ----------------------------------------------------------
    # Leaderboard
    # ----------------------------------------------------------

    leaderboard = result.get("leaderboard")

    if leaderboard:

        st.subheader("📊 Model Leaderboard")

        board = pd.DataFrame(leaderboard)

        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    metrics = result.get("metrics")

    if metrics:

        st.subheader("📈 Evaluation Metrics")

        cols = st.columns(min(4, len(metrics)))

        for i, (name, value) in enumerate(metrics.items()):

            with cols[i % len(cols)]:

                try:

                    value = round(float(value), 4)

                except Exception:

                    pass

                st.metric(
                    name.replace("_", " ").title(),
                    value,
                )

    # ----------------------------------------------------------
    # Training Information
    # ----------------------------------------------------------

    st.subheader("📋 Training Information")

    info = {
        "Problem Type": problem_type,
        "Target Column": target_column,
        "Rows": len(df),
        "Columns": len(df.columns),
    }

    st.json(info)

    # ----------------------------------------------------------
    # Feature Importance Preview
    # ----------------------------------------------------------

    importance = result.get("feature_importance")

    if importance:

        st.subheader("⭐ Top Features")
        # st.write("DEBUG")
        # st.write(type(importance))
        # st.write(importance)
        fi = (
            pd.DataFrame(
                importance.items(),
                columns=["feature", "importance"],
            )
            .sort_values("importance", ascending=False)
        )

        if "importance" in fi.columns:

            fi = fi.sort_values(
                "importance",
                ascending=False,
            )

        st.dataframe(
            fi,
            use_container_width=True,
            hide_index=True,
        )

        if {"feature", "importance"}.issubset(fi.columns):

            st.bar_chart(
                fi.set_index("feature")["importance"]
            )
    else:
        
        st.info(
            "Feature importance is not available for this model type."
        )
    # ----------------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------------

    confusion = result.get("confusion_matrix")

    if confusion:

        st.subheader("🎯 Confusion Matrix")

        st.image(
            confusion,
            use_container_width=True,
        )

    # ----------------------------------------------------------
    # ROC Curve
    # ----------------------------------------------------------

    roc = result.get("roc_curve")

    if roc:

        st.subheader("📉 ROC Curve")

        st.image(
            roc,
            use_container_width=True,
        )

    # ----------------------------------------------------------
    # Regression Plot
    # ----------------------------------------------------------

    regression = result.get("regression_plot")

    if regression:

        st.subheader("📈 Regression Plot")

        st.image(
            regression,
            use_container_width=True,
        )

# """
# frontend/pages/ml_models.py
# ML Models page — Phase 7 (not yet built). Extracted verbatim from the
# upload.py PLACEHOLDER PAGES "ML Models" entry.
# """

# from frontend.pages._placeholder import render_placeholder


# def render():
#     render_placeholder(
#         "🤖", "Phase 7", "Machine Learning Engine",
#         "AutoML — trains and compares regression, classification, and clustering models.",
#     )
