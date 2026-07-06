from __future__ import annotations

import streamlit as st

from utils import session

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

session.initialize()

st.title("🤖 AI Data Analyst Platform")
st.caption("Upload data • Clean • Analyze • Train • Explain • Predict")

st.divider()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Dataset",
        "Loaded" if session.exists("raw_df") else "Not Loaded",
    )

with c2:
    st.metric(
        "Cleaning",
        "Completed" if session.exists("clean_df") else "Pending",
    )

with c3:
    st.metric(
        "Training",
        "Completed" if session.training_completed() else "Pending",
    )

with c4:
    st.metric(
        "Prediction",
        "Completed" if session.prediction_completed() else "Pending",
    )

st.divider()

left, right = st.columns([2, 1])

with left:

    st.subheader("Workflow")

    steps = [
        ("📂 Upload Dataset", session.exists("raw_df")),
        ("🧹 Data Cleaning", session.exists("clean_df")),
        ("📊 Metadata", session.exists("metadata")),
        ("📈 EDA", session.exists("eda")),
        ("📉 Visualization", session.exists("visualizations")),
        ("🤖 ML Training", session.training_completed()),
        ("🎯 Prediction", session.prediction_completed()),
    ]

    for title, done in steps:

        if done:
            st.success(f"✔ {title}")

        else:
            st.info(f"• {title}")

with right:

    st.subheader("Current Session")

    if session.exists("raw_df"):

        df = session.get_raw_dataframe()

        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))

    else:

        st.warning("No dataset uploaded.")

st.divider()

st.subheader("Project Modules")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        """
### 📂 Data Processing

- Upload CSV
- Metadata
- Cleaning
- Missing Values
- Outlier Detection
"""
    )

with col2:

    st.info(
        """
### 🤖 Machine Learning

- AutoML
- Feature Engineering
- Model Selection
- Evaluation
- Explainability
"""
    )

with col3:

    st.info(
        """
### 📄 Deployment

- Prediction
- Reports
- Model Manager
- Download Artifacts
"""
    )

st.divider()

st.subheader("Application Status")

progress = 0

if session.exists("raw_df"):
    progress += 15

if session.exists("clean_df"):
    progress += 15

if session.exists("metadata"):
    progress += 10

if session.exists("eda"):
    progress += 15

if session.training_completed():
    progress += 25

if session.prediction_completed():
    progress += 20

st.progress(progress / 100)

st.write(f"Overall Progress: **{progress}%**")

st.divider()

st.subheader("Quick Navigation")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.page_link("pages/1_Upload.py", label="📂 Upload")

with nav2:
    st.page_link("pages/2_Cleaning.py", label="🧹 Cleaning")

with nav3:
    st.page_link("pages/6_ML_Training.py", label="🤖 Train")

with nav4:
    st.page_link("pages/7_Prediction.py", label="🎯 Predict")

st.divider()

st.caption("AI Data Analyst Platform • Version 1.0")