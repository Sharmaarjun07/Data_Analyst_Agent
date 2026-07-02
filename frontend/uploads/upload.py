"""
upload.py  —  AI Data Analyst Agent
Phases completed:
  ✅ Phase 1 — Project Setup
  ✅ Phase 2 — File Upload / Dashboard page
  ✅ Phase 3 — Metadata Engine + Dataset page

Run:
  streamlit run frontend/uploads/upload.py

Make sure metadata_service.py is in   services/metadata_service.py
and the sys.path block below points to your project root.
"""

import sys
import os
import time
import math

import pandas as pd
import numpy as np
import streamlit as st

# ── Allow import from project root ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# If you kept metadata_service.py next to upload.py for now, this also works:
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if SELF_DIR not in sys.path:
    sys.path.insert(0, SELF_DIR)

try:
    from services.metadata_service import extract_metadata
except ImportError:
    # Fallback stub so the file still loads even if service is missing
    def extract_metadata(df, filename="dataset.csv"):
        return {"overview": {}, "quality": {}, "columns": [], "target_candidates": []}


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:          #0D1117;
    --card:        #161B22;
    --card-alt:    #1C2333;
    --border:      #30363D;
    --accent:      #2F81F7;
    --accent-glow: rgba(47,129,247,0.15);
    --accent-dim:  #1A4A9E;
    --green:       #3FB950;
    --yellow:      #D29922;
    --red:         #F85149;
    --muted:       #8B949E;
    --dim:         #484F58;
    --fg:          #E6EDF3;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--fg) !important;
}
[data-testid="stHeader"]                     { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.sidebar-brand        { padding: 20px 16px 8px; border-bottom: 1px solid var(--border); margin-bottom: 10px; }
.sidebar-brand-title  { font-size: 13px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }
.sidebar-brand-sub    { font-size: 11px; color: var(--dim); margin-top: 2px; }
.section-label        { font-size: 10px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
                        color: var(--dim); padding: 14px 14px 4px; }

/* nav items rendered as HTML (locked ones) */
.nav-html { display: flex; align-items: center; gap: 10px; padding: 10px 14px;
            border-radius: 8px; margin: 3px 0; font-size: 14px; font-weight: 500; color: var(--muted); }
.nav-html.active { background: var(--accent-glow); color: var(--accent);
                   border-left: 3px solid var(--accent); padding-left: 11px; }
.nav-html.locked { opacity: .35; }
.lock-badge { font-size: 10px; font-weight: 500; color: var(--yellow);
              background: rgba(210,153,34,.1); border: 1px solid rgba(210,153,34,.25);
              border-radius: 4px; padding: 2px 7px; margin-left: auto; }

/* Streamlit buttons used as nav items */
div.stButton > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    text-align: left !important;
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
    transition: all .2s !important;
}
div.stButton > button:hover {
    background: var(--accent-glow) !important;
    color: var(--accent) !important;
}
div[data-active="true"] > div.stButton > button {
    background: var(--accent-glow) !important;
    color: var(--accent) !important;
    border-left: 3px solid var(--accent) !important;
}

/* ── Cards ── */
.card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; }

/* ── KPI row ── */
.kpi-row  { display: flex; gap: 14px; margin: 24px 0; flex-wrap: wrap; }
.kpi-card { flex: 1; min-width: 140px; background: var(--card); border: 1px solid var(--border);
            border-radius: 12px; padding: 18px 20px; }
.kpi-label { font-size: 11px; font-weight: 500; letter-spacing: .05em; text-transform: uppercase;
             color: var(--muted); margin-bottom: 8px; }
.kpi-value { font-size: 24px; font-weight: 700; color: var(--fg); font-family: 'JetBrains Mono', monospace; }
.kpi-sub   { font-size: 11px; color: var(--dim); margin-top: 4px; }

/* ── Hero ── */
.hero-header  { padding: 36px 0 12px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
.hero-title   { font-size: 32px; font-weight: 700; letter-spacing: -.02em; line-height: 1.2; }
.hero-title span { color: var(--accent); }
.hero-sub     { font-size: 14px; color: var(--muted); margin-top: 6px; }
.status-pill  { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500;
               padding: 4px 12px; border-radius: 20px; margin-top: 14px; }
.status-pill.idle  { background: rgba(139,148,158,.1); color: var(--muted); border: 1px solid var(--border); }
.status-pill.ready { background: rgba(63,185,80,.1);  color: var(--green); border: 1px solid rgba(63,185,80,.3); }
.sdot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* ── Upload card ── */
.upload-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px;
               padding: 36px 40px; text-align: center; position: relative; overflow: hidden; }
.upload-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
                       background: linear-gradient(90deg,transparent,var(--accent),transparent); }
.upload-icon { width:72px; height:72px; margin: 0 auto 18px; background: var(--accent-glow);
               border: 1.5px solid var(--accent-dim); border-radius:16px;
               display:flex; align-items:center; justify-content:center; font-size:30px; }
.upload-title { font-size:18px; font-weight:600; margin-bottom:6px; }
.upload-sub   { font-size:13px; color:var(--muted); margin-bottom:22px; line-height:1.5; }
.format-chips { display:flex; gap:8px; justify-content:center; margin-bottom:22px; flex-wrap:wrap; }
.format-chip  { font-size:11px; font-weight:500; font-family:'JetBrains Mono',monospace;
                color:var(--accent); background:var(--accent-glow); border:1px solid var(--accent-dim);
                border-radius:6px; padding:3px 10px; }

/* ── File uploader overrides ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(47,129,247,.04) !important;
    border: 1.5px dashed var(--accent-dim) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: var(--accent-glow) !important;
    border-color: var(--accent) !important;
}

/* ── Unlock banner ── */
.unlock-banner { background: linear-gradient(135deg,rgba(63,185,80,.08),rgba(47,129,247,.08));
                 border:1px solid rgba(63,185,80,.25); border-radius:12px; padding:16px 20px;
                 display:flex; align-items:center; gap:14px; margin:16px 0; }
.unlock-icon  { font-size:24px; }
.unlock-title { font-size:14px; font-weight:600; color:var(--green); margin-bottom:2px; }
.unlock-sub   { font-size:12px; color:var(--muted); }

/* ── Dataset preview card ── */
.preview-card   { background:var(--card); border:1px solid var(--border); border-radius:14px; overflow:hidden; }
.preview-header { display:flex; align-items:center; justify-content:space-between;
                  padding:14px 20px; border-bottom:1px solid var(--border); background:var(--card-alt); }
.preview-title  { font-size:13px; font-weight:600; }
.preview-meta   { font-size:11px; color:var(--muted); font-family:'JetBrains Mono',monospace; }

/* ── Column type badges ── */
.badge { font-size:10px; font-weight:600; padding:2px 8px; border-radius:5px;
         font-family:'JetBrains Mono',monospace; }
.b-num  { background:rgba(47,129,247,.12); color:var(--accent); }
.b-cat  { background:rgba(63,185,80,.12);  color:var(--green); }
.b-dat  { background:rgba(210,153,34,.12); color:var(--yellow); }
.b-txt  { background:rgba(139,148,158,.1); color:var(--muted); }

/* ── Quality score ring ── */
.quality-wrap { text-align:center; padding:24px 0; }
.quality-score { font-size:56px; font-weight:700; font-family:'JetBrains Mono',monospace; }
.quality-grade { font-size:22px; font-weight:600; margin-top:4px; }
.quality-label { font-size:12px; color:var(--muted); margin-top:4px; }

/* ── Column detail table ── */
.col-table      { width:100%; border-collapse:collapse; font-size:13px; }
.col-table th   { text-align:left; font-size:10px; font-weight:600; letter-spacing:.08em;
                  text-transform:uppercase; color:var(--muted); padding:10px 14px;
                  border-bottom:1px solid var(--border); background:var(--card-alt); }
.col-table td   { padding:10px 14px; border-bottom:1px solid rgba(48,54,61,.5);
                  color:var(--fg); vertical-align:top; }
.col-table tr:last-child td { border-bottom:none; }
.col-table tr:hover td      { background:rgba(47,129,247,.04); }
.col-name-cell  { font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:500; }
.rec-text       { font-size:12px; color:var(--muted); }
.sample-vals    { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--dim); }
.stat-mini      { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted); }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background: var(--accent) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button { color: var(--muted) !important; font-family:'Inter',sans-serif !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important;
    border-bottom-color: var(--accent) !important; }

/* ── Metric ── */
[data-testid="stMetricValue"] { color:var(--fg) !important; font-family:'JetBrains Mono',monospace !important; }
[data-testid="stMetricLabel"] { color:var(--muted) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "df":          None,
        "filename":    None,
        "metadata":    None,
        "active_page": "Dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def fmt_bytes(b):
    if b < 1024:       return f"{b} B"
    if b < 1024**2:    return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.1f} MB"

DTYPE_BADGE = {
    "numeric":     ('<span class="badge b-num">NUM</span>',  "#2F81F7"),
    "categorical": ('<span class="badge b-cat">CAT</span>',  "#3FB950"),
    "datetime":    ('<span class="badge b-dat">DATE</span>', "#D29922"),
    "text":        ('<span class="badge b-txt">TEXT</span>', "#8B949E"),
}

GRADE_COLOR = {"A": "#3FB950", "B": "#2F81F7", "C": "#D29922", "D": "#F85149", "F": "#F85149"}


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
locked = st.session_state.df is None

NAV = [
    ("Dashboard",          "🏠",  False),
    ("Dataset",            "📂",  locked),
    ("Cleaning",           "✂️",  locked),
    ("EDA",                "📊",  locked),
    ("ML Models",          "🤖",  locked),
    ("Feature Importance", "⚡",  locked),
    ("AI Insights",        "💡",  locked),
    ("Reports",            "📄",  locked),
    ("Chat",               "💬",  locked),
]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">⚡ Data Analyst Agent</div>
        <div class="sidebar-brand-sub">AI-Powered Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)

    for label, icon, is_locked in NAV:
        if is_locked:
            st.markdown(f"""
            <div class="nav-html locked">
                <span>{icon}</span><span>{label}</span>
                <span class="lock-badge">🔒 Upload first</span>
            </div>""", unsafe_allow_html=True)
        else:
            is_active = st.session_state.active_page == label
            active_style = (
                "background:var(--accent-glow);color:var(--accent);"
                "border-left:3px solid var(--accent);padding-left:11px;"
            ) if is_active else ""
            # Render label in HTML for styling, but also put an invisible button
            # st.markdown(f"""
            # <div class="nav-html {'active' if is_active else ''}" style="{active_style}">
            #     <span>{icon}</span><span>{label}</span>
            # </div>""", unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"_nav_{label}"):
                st.session_state.active_page = label
                st.rerun()
                

    st.markdown('<div class="section-label" style="margin-top:10px;">System</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="nav-html"><span>⚙️</span><span>Settings</span></div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.active_page

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD (upload + quick overview)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    # Hero
    status_class = "ready" if not locked else "idle"
    status_text  = "Dataset loaded — all modules unlocked" if not locked else "Waiting for dataset upload"
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">Data <span>Analyst</span> Agent</div>
        <div class="hero-sub">Upload a CSV to unlock automated cleaning, EDA,
            machine learning, insights, and report generation.</div>
        <div class="status-pill {status_class}">
            <span class="sdot"></span>{status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── BEFORE UPLOAD ──────────────────────────────────────────────────────
    if locked:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-icon">📁</div>
            <div class="upload-title">Upload your dataset to get started</div>
            <div class="upload-sub">
                Drop any CSV file here and the agent will automatically clean it,<br>
                analyse it, train models, and generate a full report.
            </div>
            <div class="format-chips">
                <span class="format-chip">.csv</span>
                <span class="format-chip">UTF-8</span>
                <span class="format-chip">comma-separated</span>
                <span class="format-chip">up to 200 MB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"],
                                    label_visibility="collapsed")

        if uploaded:
            bar = st.progress(0, text="Reading file…")
            try:
                for pct in range(0, 81, 20):
                    time.sleep(0.06)
                    bar.progress(pct, text=f"Loading dataset… {pct}%")
                df = pd.read_csv(uploaded)
                bar.progress(85, text="Extracting metadata…")
                meta = extract_metadata(df, uploaded.name)
                bar.progress(100, text="✅  Done!")
                st.session_state.df       = df
                st.session_state.filename = uploaded.name
                st.session_state.metadata = meta
                time.sleep(0.35)
                st.rerun()
            except Exception as e:
                bar.empty()
                st.error(f"❌ Could not read file: {e}")

    # ── AFTER UPLOAD ───────────────────────────────────────────────────────
    else:
        df   = st.session_state.df
        meta = st.session_state.metadata
        ov   = meta["overview"]
        ql   = meta["quality"]

        # Unlock banner
        st.markdown(f"""
        <div class="unlock-banner">
            <div class="unlock-icon">🎉</div>
            <div>
                <div class="unlock-title">All modules unlocked</div>
                <div class="unlock-sub"><strong>{ov['filename']}</strong> loaded successfully.
                Use the sidebar to navigate to any module.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # KPI row
        dc    = ov.get("dtype_counts", {})
        n_num = dc.get("numeric", 0)
        n_cat = dc.get("categorical", 0)
        dups  = ov["dup_count"]
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-value">{fmt_num(ov["n_rows"])}</div>
                <div class="kpi-sub">records in dataset</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Columns</div>
                <div class="kpi-value">{ov["n_cols"]}</div>
                <div class="kpi-sub">{n_num} numeric · {n_cat} categorical</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Missing Values</div>
                <div class="kpi-value">{fmt_num(ov["missing_cells"])}</div>
                <div class="kpi-sub">{ov["missing_pct"]}% of all cells</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Duplicate Rows</div>
                <div class="kpi-value">{dups}</div>
                <div class="kpi-sub">{"⚠️ needs cleaning" if dups > 0 else "✅ none found"}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Data Quality</div>
                <div class="kpi-value" style="color:{GRADE_COLOR.get(ql['grade'],'#fff')}">
                    {ql['score']}
                </div>
                <div class="kpi-sub">Grade {ql['grade']} out of 100</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Memory</div>
                <div class="kpi-value">{fmt_bytes(ov["memory_bytes"])}</div>
                <div class="kpi-sub">in-memory footprint</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dataset preview
        st.markdown(f"""
        <div class="preview-card">
            <div class="preview-header">
                <span class="preview-title">📋 Dataset Preview</span>
                <span class="preview-meta">First 10 rows · {ov["n_rows"]} total</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=300)

        # Replace file
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("⬆️  Upload a different file"):
            nf = st.file_uploader("Replace dataset", type=["csv"],
                                  label_visibility="collapsed", key="_replace")
            if nf:
                try:
                    df2  = pd.read_csv(nf)
                    meta2 = extract_metadata(df2, nf.name)
                    st.session_state.df       = df2
                    st.session_state.filename = nf.name
                    st.session_state.metadata = meta2
                    st.success(f"✅ Replaced with **{nf.name}**")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATASET  (Phase 3 — Metadata Engine)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":

    df   = st.session_state.df
    meta = st.session_state.metadata
    ov   = meta["overview"]
    ql   = meta["quality"]
    cols = meta["columns"]

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">Dataset <span>Overview</span></div>
        <div class="hero-sub">
            Deep column-level metadata for <strong>{ov['filename']}</strong> ·
            extracted at {ov['extracted_at']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top row: Quality score + overview stats ─────────────────────────────
    q_col, o_col = st.columns([1, 3], gap="large")

    with q_col:
        grade_color = GRADE_COLOR.get(ql["grade"], "#fff")
        score = ql["score"]
        # Simple circular-looking display using a big number
        st.markdown(f"""
        <div class="card" style="padding:28px 20px; text-align:center;">
            <div style="font-size:11px;font-weight:600;letter-spacing:.08em;
                        text-transform:uppercase;color:var(--muted);margin-bottom:16px;">
                DATA QUALITY SCORE
            </div>
            <div style="font-size:64px;font-weight:700;
                        font-family:'JetBrains Mono',monospace;
                        color:{grade_color};line-height:1;">
                {score}
            </div>
            <div style="font-size:13px;color:var(--muted);margin-top:4px;">out of 100</div>
            <div style="font-size:28px;font-weight:700;color:{grade_color};margin-top:10px;">
                Grade {ql['grade']}
            </div>
            <div style="margin-top:16px;">
        """, unsafe_allow_html=True)

        # Progress bar as visual
        st.progress(int(score) / 100)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Issues list
        issues = ql.get("issues", {})
        issue_items = []
        if issues.get("missing_cells", 0):
            issue_items.append(f"⚠️ {fmt_num(issues['missing_cells'])} missing cells")
        if issues.get("duplicate_rows", 0):
            issue_items.append(f"⚠️ {issues['duplicate_rows']} duplicate rows")
        if issues.get("high_missing_cols"):
            hm = issues["high_missing_cols"]
            issue_items.append(f"⚠️ {len(hm)} column(s) >30% missing")
        if issues.get("id_like_cols"):
            issue_items.append(f"ℹ️ {len(issues['id_like_cols'])} ID-like column(s)")
        if not issue_items:
            issue_items.append("✅ No major issues found")

        for item in issue_items:
            st.markdown(
                f'<div style="font-size:12px;color:var(--muted);padding:4px 0;">{item}</div>',
                unsafe_allow_html=True
            )

    with o_col:
        # Overview cards
        dc = ov.get("dtype_counts", {})
        st.markdown(f"""
        <div class="kpi-row" style="margin-top:0;">
            <div class="kpi-card">
                <div class="kpi-label">Rows</div>
                <div class="kpi-value">{fmt_num(ov['n_rows'])}</div>
                <div class="kpi-sub">total records</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Columns</div>
                <div class="kpi-value">{ov['n_cols']}</div>
                <div class="kpi-sub">total features</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Numeric</div>
                <div class="kpi-value" style="color:var(--accent);">{dc.get('numeric',0)}</div>
                <div class="kpi-sub">number columns</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Categorical</div>
                <div class="kpi-value" style="color:var(--green);">{dc.get('categorical',0)}</div>
                <div class="kpi-sub">text/category columns</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Datetime</div>
                <div class="kpi-value" style="color:var(--yellow);">{dc.get('datetime',0)}</div>
                <div class="kpi-sub">date columns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Missing Cells</div>
                <div class="kpi-value">{fmt_num(ov['missing_cells'])}</div>
                <div class="kpi-sub">{ov['missing_pct']}% of all cells</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Duplicate Rows</div>
                <div class="kpi-value">{ov['dup_count']}</div>
                <div class="kpi-sub">exact row matches</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Memory</div>
                <div class="kpi-value">{fmt_bytes(ov['memory_bytes'])}</div>
                <div class="kpi-sub">in-memory footprint</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Target candidates
        targets = meta.get("target_candidates", [])
        if targets:
            st.markdown("""
            <div style="font-size:12px;font-weight:600;letter-spacing:.06em;
                        text-transform:uppercase;color:var(--muted);margin:16px 0 8px;">
                🎯 Potential Target Columns for ML
            </div>""", unsafe_allow_html=True)
            for t in targets:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:6px 0;
                            border-bottom:1px solid rgba(48,54,61,.4);font-size:13px;">
                    <span style="font-family:'JetBrains Mono',monospace;color:var(--fg);">
                        {t['name']}</span>
                    <span style="font-size:11px;color:var(--muted);">— {t['reason']}</span>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Column detail tabs ──────────────────────────────────────────────────
    tab_all, tab_num, tab_cat, tab_issues = st.tabs([
        f"📋 All Columns ({len(cols)})",
        f"🔢 Numeric ({dc.get('numeric',0)})",
        f"🏷️  Categorical ({dc.get('categorical',0)})",
        "⚠️  Issues",
    ])

    def render_col_table(col_list):
        if not col_list:
            st.markdown('<div style="color:var(--muted);padding:20px;">No columns in this category.</div>',
                        unsafe_allow_html=True)
            return

        rows_html = ""
        for cm in col_list:
            badge_html, _ = DTYPE_BADGE.get(cm["dtype_category"], ("", ""))
            miss_color = "var(--red)" if cm["missing_pct"] > 30 else \
                         "var(--yellow)" if cm["missing_pct"] > 5 else "var(--muted)"
            miss_html  = f'<span style="color:{miss_color};">{cm["missing_pct"]:.1f}%</span>'

            # Stats mini
            stats_html = ""
            if cm["dtype_category"] == "numeric" and "numeric_stats" in cm:
                ns = cm["numeric_stats"]
                stats_html = (f'<div class="stat-mini">'
                              f'min {ns.get("min","—")} · '
                              f'mean {ns.get("mean","—")} · '
                              f'max {ns.get("max","—")}</div>')
                if ns.get("outliers", 0):
                    stats_html += f'<div class="stat-mini" style="color:var(--yellow);">⚠️ {ns["outliers"]} outlier(s)</div>'
            elif cm["dtype_category"] == "categorical" and "categorical_stats" in cm:
                top = cm["categorical_stats"].get("top_values", [])
                if top:
                    vals = ", ".join(f'{v["value"]} ({v["pct"]}%)' for v in top[:2])
                    stats_html = f'<div class="stat-mini">Top: {vals}</div>'

            samples = ", ".join(str(v) for v in cm.get("sample_values", [])[:3])

            rows_html += f"""
            <tr>
              <td class="col-name-cell">{cm['name']}</td>
              <td>{badge_html}</td>
              <td><span style="color:var(--muted);font-size:12px;">{cm['dtype']}</span></td>
              <td>{miss_html}
                  <div style="font-size:11px;color:var(--dim);">{cm['missing_count']} cells</div>
              </td>
              <td><span style="font-family:'JetBrains Mono',monospace;font-size:12px;">
                  {fmt_num(cm['n_unique'])}</span></td>
              <td>
                  <div class="sample-vals">{samples}</div>
                  {stats_html}
              </td>
              <td><div class="rec-text">{cm['recommendation']}</div></td>
            </tr>"""

        st.markdown(f"""
        <div class="card" style="overflow:auto;">
            <table class="col-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Dtype</th>
                        <th>Missing</th>
                        <th>Unique</th>
                        <th>Sample / Stats</th>
                        <th>Recommendation</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>""", unsafe_allow_html=True)

    with tab_all:
        # Search / filter
        search = st.text_input("🔍 Filter columns by name", placeholder="type a column name…",
                               label_visibility="collapsed")
        filtered = [c for c in cols if search.lower() in c["name"].lower()] if search else cols
        render_col_table(filtered)

    with tab_num:
        render_col_table([c for c in cols if c["dtype_category"] == "numeric"])

    with tab_cat:
        render_col_table([c for c in cols if c["dtype_category"] == "categorical"])

    with tab_issues:
        issue_cols = [
            c for c in cols
            if c["missing_pct"] > 0
            or c.get("numeric_stats", {}).get("outliers", 0) > 0
            or (c["dtype_category"] == "categorical" and c["n_unique"] == ov["n_rows"])
        ]
        if issue_cols:
            render_col_table(issue_cols)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:var(--green);font-size:18px;font-weight:600;">
                ✅ No issues detected in this dataset
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER PAGES  (Phases 4–11 — coming soon)
# ══════════════════════════════════════════════════════════════════════════════
else:
    COMING = {
        "Cleaning":           ("✂️",  "Phase 4",  "Data Cleaning Engine",       "Detect and fix missing values, duplicates, wrong types, and outliers automatically."),
        "EDA":                ("📊",  "Phase 5",  "EDA Engine",                 "Statistics, correlation matrix, skewness, kurtosis, and distribution analysis."),
        "ML Models":          ("🤖",  "Phase 7",  "Machine Learning Engine",    "AutoML: regression, classification, clustering — trains and compares multiple models."),
        "Feature Importance": ("⚡",  "Phase 7",  "Feature Importance",         "Which columns matter most for your prediction target and why."),
        "AI Insights":        ("💡",  "Phase 8",  "Insight Generation Engine",  "The LLM reads your analysis and writes plain-English business insights and recommendations."),
        "Reports":            ("📄",  "Phase 11", "Report Generation",          "Auto-generated PDF report and PowerPoint deck ready to hand to a client or manager."),
        "Chat":               ("💬",  "Phase 10", "Chat With Dataset",          "Ask questions in plain language — SQL Agent answers from your data, not from guessing."),
    }
    icon, phase_tag, title, desc = COMING.get(
        page, ("🚧", "Coming Soon", page, "This module is under development.")
    )
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:60vh;text-align:center;padding:60px 40px;">
        <div style="font-size:56px;margin-bottom:20px;">{icon}</div>
        <div style="font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
                    color:var(--accent);margin-bottom:10px;">{phase_tag}</div>
        <div style="font-size:28px;font-weight:700;margin-bottom:12px;">{title}</div>
        <div style="font-size:15px;color:var(--muted);max-width:480px;line-height:1.6;">
            {desc}
        </div>
        <div style="margin-top:28px;padding:10px 24px;border:1px solid var(--border);
                    border-radius:8px;font-size:13px;color:var(--dim);">
            🔨 Coming in the next phase of development
        </div>
    </div>
    """, unsafe_allow_html=True)