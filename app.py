"""
app.py  —  AI Data Analyst Agent (main entry point)

This is the reorganised, modular version of the original monolithic
upload.py. All logic is identical — it has simply been split across
smaller files:

  Data_Analyst_Agent/
  ├── app.py                              ← this file (router only)
  ├── frontend/
  │   ├── styles.py                       ← CSS only
  │   ├── sidebar.py                      ← Sidebar only
  │   ├── session.py                      ← Session state + shared helpers
  │   └── pages/
  │       ├── dashboard.py
  │       ├── dataset.py
  │       ├── cleaning.py
  │       ├── eda.py
  │       ├── ml_models.py
  │       ├── feature_importance.py
  │       ├── ai_insights.py
  │       ├── reports.py
  │       └── chat.py

Run:
  streamlit run Data_Analyst_Agent/app.py

Folder structure expected (for the services/ package, same as before):
  AI_Data_Analyst/
  ├── Data_Analyst_Agent/app.py     ← this file
  ├── services/metadata_service.py
  ├── services/cleaning_service.py
  └── requirements.txt
"""

import sys
import os

import streamlit as st

# ── Path setup so both `services` and `frontend` packages can be imported ──
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SELF_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from frontend.styles import load_css
from frontend.session import init_session_state
from frontend.sidebar import render_sidebar
from frontend.pages import (
    dashboard,
    dataset,
    cleaning,
    eda,
    ml_models,
    feature_importance,
    ai_insights,
    reports,
    chat,
)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Analyst Agent", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
load_css()

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
locked = render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.active_page

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    dashboard.render(locked)

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PAGE  (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    dataset.render()

# ══════════════════════════════════════════════════════════════════════════════
#  CLEANING PAGE  (Phase 4)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Cleaning":
    cleaning.render()

# ══════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER PAGES  (Phases 5–11)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    eda.render()
elif page == "ML Models":
    ml_models.render()
elif page == "Feature Importance":
    feature_importance.render()
elif page == "AI Insights":
    ai_insights.render()
elif page == "Reports":
    reports.render()
elif page == "Chat":
    chat.render()
