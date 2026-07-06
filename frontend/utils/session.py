from __future__ import annotations

from typing import Any

import streamlit as st


DEFAULT_SESSION = {
    "raw_df": None,
    "clean_df": None,
    "metadata": None,
    "eda": None,
    "visualizations": None,
    "ml_result": None,
    "prediction": None,
    "leaderboard": None,
    "selected_model": None,
    "training_complete": False,
    "prediction_complete": False,
}


def initialize() -> None:
    for key, value in DEFAULT_SESSION.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:
    st.session_state[key] = value


def exists(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None


def remove(key: str) -> None:
    if key in st.session_state:
        del st.session_state[key]


def clear() -> None:
    for key in DEFAULT_SESSION:
        if key in st.session_state:
            del st.session_state[key]
    initialize()


def set_raw_dataframe(df):
    set("raw_df", df)


def get_raw_dataframe():
    return get("raw_df")


def set_clean_dataframe(df):
    set("clean_df", df)


def get_clean_dataframe():
    return get("clean_df")


def set_metadata(metadata):
    set("metadata", metadata)


def get_metadata():
    return get("metadata")


def set_eda(result):
    set("eda", result)


def get_eda():
    return get("eda")


def set_visualizations(charts):
    set("visualizations", charts)


def get_visualizations():
    return get("visualizations")


def set_ml_result(result):
    set("ml_result", result)


def get_ml_result():
    return get("ml_result")


def set_prediction(result):
    set("prediction", result)


def get_prediction():
    return get("prediction")


def set_leaderboard(df):
    set("leaderboard", df)


def get_leaderboard():
    return get("leaderboard")


def set_selected_model(model_name: str):
    set("selected_model", model_name)


def get_selected_model():
    return get("selected_model")


def mark_training_complete():
    set("training_complete", True)


def training_completed():
    return get("training_complete", False)


def mark_prediction_complete():
    set("prediction_complete", True)


def prediction_completed():
    return get("prediction_complete", False)