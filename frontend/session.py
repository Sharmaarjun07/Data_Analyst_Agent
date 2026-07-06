"""
frontend/session.py
Session-state initialisation + small shared helpers/constants used
across multiple pages. Extracted verbatim from upload.py.
"""

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "df":            None,   # original uploaded df
    "cleaned_df":    None,   # after Phase 4
    "filename":      None,
    "metadata":      None,   # from Phase 3
    "cleaning_issues": None, # from Phase 4 analyze_issues
    "cleaning_report": None, # from Phase 4 apply_cleaning
    "active_page":   "Dashboard",
}


def init_session_state():
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS  (shared across pages)
# ─────────────────────────────────────────────────────────────────────────────
def fmt_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)


def fmt_bytes(b):
    if b < 1024: return f"{b} B"
    if b < 1024**2: return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.1f} MB"


DTYPE_BADGE = {
    "numeric":     '<span class="badge b-num">NUM</span>',
    "categorical": '<span class="badge b-cat">CAT</span>',
    "datetime":    '<span class="badge b-dat">DATE</span>',
    "text":        '<span class="badge b-txt">TEXT</span>',
}
GRADE_COLOR = {"A": "#3FB950", "B": "#2F81F7", "C": "#D29922", "D": "#F85149", "F": "#F85149"}

FIX_LABELS = {
    "fill_median":    "Fill with Median",
    "fill_mean":      "Fill with Mean",
    "fill_mode":      "Fill with Most Frequent",
    "fill_constant":  "Fill with Constant Value",
    "drop_column":    "Drop Entire Column",
    "drop_rows":      "Drop Rows with Missing",
    "winsorize":      "Winsorize (Clip to IQR Fences)",
    "keep":           "Keep As-Is",
    "drop_duplicates": "Remove Duplicate Rows",
    "convert_numeric": "Convert to Numeric",
    "convert_datetime": "Convert to Datetime",
}
