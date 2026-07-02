"""
upload.py  —  AI Data Analyst Agent
Phases completed:
  ✅ Phase 1 — Project Setup
  ✅ Phase 2 — File Upload / Dashboard page
  ✅ Phase 3 — Metadata Engine + Dataset page
  ✅ Phase 4 — Data Cleaning Engine + Cleaning page

Run:
  streamlit run frontend/uploads/upload.py

Folder structure expected:
  AI_Data_Analyst/
  ├── frontend/uploads/upload.py    ← this file
  ├── services/metadata_service.py
  ├── services/cleaning_service.py
  └── requirements.txt
"""

import sys, os, time, copy
import pandas as pd
import numpy as np
import streamlit as st

# ── Path setup so both services can be imported ────────────────────────────
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SELF_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from services.metadata_service import extract_metadata
except ImportError:
    def extract_metadata(df, filename="dataset.csv"):
        return {"overview": {}, "quality": {}, "columns": [], "target_candidates": []}

try:
    from services.cleaning_service import analyze_issues, apply_cleaning
except ImportError:
    def analyze_issues(df, metadata):
        return {"summary": {}, "missing": [], "duplicates": {}, "outliers": [], "type_issues": []}
    def apply_cleaning(df, plan):
        return df.copy(), {"steps": [], "rows_before": len(df), "rows_after": len(df),
                           "rows_removed": 0, "cols_before": df.shape[1],
                           "cols_after": df.shape[1], "total_actions": 0,
                           "applied_at": ""}

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Analyst Agent", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0D1117; --card:#161B22; --card-alt:#1C2333; --border:#30363D;
  --accent:#2F81F7; --glow:rgba(47,129,247,.15); --dim-a:#1A4A9E;
  --green:#3FB950; --yellow:#D29922; --red:#F85149;
  --muted:#8B949E; --dim:#484F58; --fg:#E6EDF3;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'Inter',sans-serif!important;color:var(--fg)!important;}
[data-testid="stHeader"]{background:transparent!important;}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"]>div:first-child{padding-top:0!important;}
.sb-brand{padding:20px 16px 8px;border-bottom:1px solid var(--border);margin-bottom:10px;}
.sb-brand-t{font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);}
.sb-brand-s{font-size:11px;color:var(--dim);margin-top:2px;}
.sec-lbl{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);padding:14px 14px 4px;}
.nav-html{display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:8px;margin:3px 0;font-size:14px;font-weight:500;color:var(--muted);}
.nav-html.active{background:var(--glow);color:var(--accent);border-left:3px solid var(--accent);padding-left:11px;}
.nav-html.locked{opacity:.35;}
.lock-badge{font-size:10px;color:var(--yellow);background:rgba(210,153,34,.1);border:1px solid rgba(210,153,34,.25);border-radius:4px;padding:2px 7px;margin-left:auto;}

/* nav buttons */
div.stButton>button{background:transparent!important;color:var(--muted)!important;border:none!important;border-radius:8px!important;font-size:14px!important;font-weight:500!important;padding:10px 14px!important;text-align:left!important;width:100%!important;font-family:'Inter',sans-serif!important;transition:all .2s!important;}
div.stButton>button:hover{background:var(--glow)!important;color:var(--accent)!important;}

/* primary action button */
.stButton.primary>button,div[data-testid="stFormSubmitButton"]>button{background:var(--accent)!important;color:#fff!important;border-radius:8px!important;font-weight:600!important;}

/* cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;}

/* kpi */
.kpi-row{display:flex;gap:14px;margin:24px 0;flex-wrap:wrap;}
.kpi-card{flex:1;min-width:130px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px;}
.kpi-lbl{font-size:11px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.kpi-val{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;}
.kpi-sub{font-size:11px;color:var(--dim);margin-top:3px;}

/* hero */
.hero{padding:36px 0 12px;border-bottom:1px solid var(--border);margin-bottom:28px;}
.hero-t{font-size:32px;font-weight:700;letter-spacing:-.02em;line-height:1.2;}
.hero-t span{color:var(--accent);}
.hero-s{font-size:14px;color:var(--muted);margin-top:6px;}
.spill{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:500;padding:4px 12px;border-radius:20px;margin-top:14px;}
.spill.idle{background:rgba(139,148,158,.1);color:var(--muted);border:1px solid var(--border);}
.spill.ready{background:rgba(63,185,80,.1);color:var(--green);border:1px solid rgba(63,185,80,.3);}
.sdot{width:7px;height:7px;border-radius:50%;background:currentColor;}

/* upload */
.upload-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:36px 40px;text-align:center;position:relative;overflow:hidden;}
.upload-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);}
.upload-icon{width:72px;height:72px;margin:0 auto 18px;background:var(--glow);border:1.5px solid var(--dim-a);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:30px;}
.upload-title{font-size:18px;font-weight:600;margin-bottom:6px;}
.upload-sub{font-size:13px;color:var(--muted);margin-bottom:22px;line-height:1.5;}
.chip-row{display:flex;gap:8px;justify-content:center;margin-bottom:22px;flex-wrap:wrap;}
.chip{font-size:11px;font-weight:500;font-family:'JetBrains Mono',monospace;color:var(--accent);background:var(--glow);border:1px solid var(--dim-a);border-radius:6px;padding:3px 10px;}

/* file uploader */
[data-testid="stFileUploaderDropzone"]{background:rgba(47,129,247,.04)!important;border:1.5px dashed var(--dim-a)!important;border-radius:10px!important;}
[data-testid="stFileUploaderDropzone"]:hover{background:var(--glow)!important;border-color:var(--accent)!important;}

/* unlock banner */
.ubanner{background:linear-gradient(135deg,rgba(63,185,80,.08),rgba(47,129,247,.08));border:1px solid rgba(63,185,80,.25);border-radius:12px;padding:16px 20px;display:flex;align-items:center;gap:14px;margin:16px 0;}

/* preview card */
.prev-card{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden;}
.prev-hdr{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--border);background:var(--card-alt);}
.prev-t{font-size:13px;font-weight:600;}
.prev-m{font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace;}

/* badges */
.badge{font-size:10px;font-weight:600;padding:2px 8px;border-radius:5px;font-family:'JetBrains Mono',monospace;}
.b-num{background:rgba(47,129,247,.12);color:var(--accent);}
.b-cat{background:rgba(63,185,80,.12);color:var(--green);}
.b-dat{background:rgba(210,153,34,.12);color:var(--yellow);}
.b-txt{background:rgba(139,148,158,.1);color:var(--muted);}

/* column table */
.col-table{width:100%;border-collapse:collapse;font-size:13px;}
.col-table th{text-align:left;font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:10px 14px;border-bottom:1px solid var(--border);background:var(--card-alt);}
.col-table td{padding:10px 14px;border-bottom:1px solid rgba(48,54,61,.5);vertical-align:top;}
.col-table tr:last-child td{border-bottom:none;}
.col-table tr:hover td{background:rgba(47,129,247,.04);}
.mono{font-family:'JetBrains Mono',monospace;}

/* ── CLEANING PAGE ── */
.issue-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin-bottom:12px;transition:border-color .2s;}
.issue-card:hover{border-color:var(--accent);}
.issue-header{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.issue-col-name{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--fg);}
.issue-stat{font-size:12px;color:var(--muted);}
.severity-high{color:var(--red);}
.severity-med{color:var(--yellow);}
.severity-low{color:var(--green);}

/* report table */
.report-table{width:100%;border-collapse:collapse;font-size:13px;}
.report-table th{text-align:left;font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:10px 14px;border-bottom:1px solid var(--border);background:var(--card-alt);}
.report-table td{padding:10px 14px;border-bottom:1px solid rgba(48,54,61,.5);vertical-align:middle;}
.report-table tr:last-child td{border-bottom:none;}

/* before/after compare */
.compare-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center;}
.compare-val{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;margin:8px 0;}
.compare-lbl{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
.compare-delta{font-size:13px;margin-top:4px;}

/* progress */
[data-testid="stProgress"]>div>div{background:var(--accent)!important;}

/* tabs */
[data-testid="stTabs"] button{color:var(--muted)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}

/* select box */
[data-testid="stSelectbox"] label{color:var(--muted)!important;font-size:12px!important;}

/* scrollbar */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:10px;}
</style>
""", unsafe_allow_html=True)

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
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
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
GRADE_COLOR = {"A":"#3FB950","B":"#2F81F7","C":"#D29922","D":"#F85149","F":"#F85149"}

FIX_LABELS = {
    "fill_median":    "Fill with Median",
    "fill_mean":      "Fill with Mean",
    "fill_mode":      "Fill with Most Frequent",
    "fill_constant":  "Fill with Constant Value",
    "drop_column":    "Drop Entire Column",
    "drop_rows":      "Drop Rows with Missing",
    "winsorize":      "Winsorize (Clip to IQR Fences)",
    "keep":           "Keep As-Is",
    "drop_duplicates":"Remove Duplicate Rows",
    "convert_numeric":"Convert to Numeric",
    "convert_datetime":"Convert to Datetime",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
locked = st.session_state.df is None
NAV = [
    ("Dashboard",          "🏠", False),
    ("Dataset",            "📂", locked),
    ("Cleaning",           "✂️", locked),
    ("EDA",                "📊", locked),
    ("ML Models",          "🤖", locked),
    ("Feature Importance", "⚡", locked),
    ("AI Insights",        "💡", locked),
    ("Reports",            "📄", locked),
    ("Chat",               "💬", locked),
]

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-t">⚡ Data Analyst Agent</div>
        <div class="sb-brand-s">AI-Powered Analysis Platform</div>
    </div>
    <div class="sec-lbl">Navigation</div>
    """, unsafe_allow_html=True)

    for label, icon, is_locked in NAV:
        if is_locked:
            st.markdown(f"""
            <div class="nav-html locked">
                <span>{icon}</span><span>{label}</span>
                <span class="lock-badge">🔒 Upload first</span>
            </div>""", unsafe_allow_html=True)
        else:
            active = st.session_state.active_page == label
            # st.markdown(f"""
            # <div class="nav-html {'active' if active else ''}">
            #     <span>{icon}</span><span>{label}</span>
            # </div>""", unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"_nav_{label}"):
                st.session_state.active_page = label
                st.rerun()

    st.markdown('<div class="sec-lbl" style="margin-top:10px;">System</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="nav-html"><span>⚙️</span><span>Settings</span></div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.active_page


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    status_cls  = "ready" if not locked else "idle"
    status_text = "Dataset loaded — all modules unlocked" if not locked else "Waiting for dataset upload"
    st.markdown(f"""
    <div class="hero">
        <div class="hero-t">Data <span>Analyst</span> Agent</div>
        <div class="hero-s">Upload a CSV to unlock automated cleaning, EDA, ML, insights, and reports.</div>
        <div class="spill {status_cls}"><span class="sdot"></span>{status_text}</div>
    </div>""", unsafe_allow_html=True)

    if locked:
        st.markdown("""
        <div class="upload-card">
            <div class="upload-icon">📁</div>
            <div class="upload-title">Upload your dataset to get started</div>
            <div class="upload-sub">The agent will automatically clean it, analyse it,
                train models, and generate a full report.</div>
            <div class="chip-row">
                <span class="chip">.csv</span>
                <span class="chip">UTF-8</span>
                <span class="chip">comma-separated</span>
                <span class="chip">up to 200 MB</span>
            </div>
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            bar = st.progress(0, text="Reading file…")
            try:
                for pct in range(0, 81, 20):
                    time.sleep(0.05)
                    bar.progress(pct, text=f"Loading… {pct}%")
                df = pd.read_csv(uploaded)
                bar.progress(85, text="Extracting metadata…")
                meta   = extract_metadata(df, uploaded.name)
                bar.progress(95, text="Analysing cleaning issues…")
                issues = analyze_issues(df, meta)
                bar.progress(100, text="✅ Done!")
                st.session_state.df              = df
                st.session_state.filename        = uploaded.name
                st.session_state.metadata        = meta
                st.session_state.cleaning_issues = issues
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                bar.empty()
                st.error(f"❌ Could not read file: {e}")
    else:
        df   = st.session_state.df
        meta = st.session_state.metadata
        ov   = meta["overview"]
        ql   = meta["quality"]
        dc   = ov.get("dtype_counts", {})

        st.markdown(f"""
        <div class="ubanner">
            <div style="font-size:24px;">🎉</div>
            <div>
                <div style="font-size:14px;font-weight:600;color:var(--green);">All modules unlocked</div>
                <div style="font-size:12px;color:var(--muted);"><strong>{ov['filename']}</strong>
                loaded — use the sidebar to navigate.</div>
            </div>
        </div>""", unsafe_allow_html=True)

        issues = st.session_state.cleaning_issues or {}
        sm     = issues.get("summary", {})
        total_issues = sm.get("total_issues", 0)

        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-lbl">Rows</div>
                <div class="kpi-val">{fmt_num(ov['n_rows'])}</div>
                <div class="kpi-sub">records</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Columns</div>
                <div class="kpi-val">{ov['n_cols']}</div>
                <div class="kpi-sub">{dc.get('numeric',0)} num · {dc.get('categorical',0)} cat</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Missing Cells</div>
                <div class="kpi-val" style="color:{'var(--yellow)' if ov['missing_cells']>0 else 'var(--green)'};">
                    {fmt_num(ov['missing_cells'])}</div>
                <div class="kpi-sub">{ov['missing_pct']}% of all cells</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Duplicate Rows</div>
                <div class="kpi-val" style="color:{'var(--yellow)' if ov['dup_count']>0 else 'var(--green)'};">
                    {ov['dup_count']}</div>
                <div class="kpi-sub">{"needs cleaning" if ov['dup_count']>0 else "✅ none found"}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Data Quality</div>
                <div class="kpi-val" style="color:{GRADE_COLOR.get(ql['grade'],'#fff')};">
                    {ql['score']}</div>
                <div class="kpi-sub">Grade {ql['grade']} / 100</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-lbl">Cleaning Issues</div>
                <div class="kpi-val" style="color:{'var(--red)' if total_issues>5 else 'var(--yellow)' if total_issues>0 else 'var(--green)'};">
                    {total_issues}</div>
                <div class="kpi-sub">{"→ go to Cleaning" if total_issues>0 else "✅ dataset looks clean"}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prev-card">
            <div class="prev-hdr">
                <span class="prev-t">📋 Dataset Preview</span>
                <span class="prev-m">First 10 rows · {ov['n_rows']} total</span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=300)

        with st.expander("⬆️  Upload a different file"):
            nf = st.file_uploader("Replace", type=["csv"], label_visibility="collapsed", key="_rep")
            if nf:
                try:
                    df2 = pd.read_csv(nf)
                    m2  = extract_metadata(df2, nf.name)
                    i2  = analyze_issues(df2, m2)
                    st.session_state.update({"df": df2, "filename": nf.name, "metadata": m2,
                                             "cleaning_issues": i2, "cleaned_df": None,
                                             "cleaning_report": None})
                    st.success(f"✅ Replaced with **{nf.name}**")
                    time.sleep(0.4)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PAGE  (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    meta = st.session_state.metadata
    df   = st.session_state.df
    ov   = meta["overview"]
    ql   = meta["quality"]
    cols = meta["columns"]
    dc   = ov.get("dtype_counts", {})

    st.markdown(f"""
    <div class="hero">
        <div class="hero-t">Dataset <span>Overview</span></div>
        <div class="hero-s">Deep column-level metadata for <strong>{ov['filename']}</strong>
        · extracted at {ov['extracted_at']}</div>
    </div>""", unsafe_allow_html=True)

    q_col, o_col = st.columns([1, 3], gap="large")
    with q_col:
        gc = GRADE_COLOR.get(ql["grade"], "#fff")
        st.markdown(f"""
        <div class="card" style="padding:24px 20px;text-align:center;">
            <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
                        color:var(--muted);margin-bottom:14px;">DATA QUALITY SCORE</div>
            <div style="font-size:60px;font-weight:700;font-family:'JetBrains Mono',monospace;
                        color:{gc};line-height:1;">{ql['score']}</div>
            <div style="font-size:12px;color:var(--muted);margin-top:4px;">out of 100</div>
            <div style="font-size:26px;font-weight:700;color:{gc};margin-top:8px;">Grade {ql['grade']}</div>
        </div>""", unsafe_allow_html=True)
        st.progress(int(ql["score"]) / 100)
        issues_q = ql.get("issues", {})
        items = []
        if issues_q.get("missing_cells", 0):
            items.append(f"⚠️ {fmt_num(issues_q['missing_cells'])} missing cells")
        if issues_q.get("duplicate_rows", 0):
            items.append(f"⚠️ {issues_q['duplicate_rows']} duplicate rows")
        if issues_q.get("high_missing_cols"):
            items.append(f"⚠️ {len(issues_q['high_missing_cols'])} col(s) >30% missing")
        if not items:
            items.append("✅ No major issues found")
        for it in items:
            st.markdown(f'<div style="font-size:12px;color:var(--muted);padding:4px 0;">{it}</div>',
                        unsafe_allow_html=True)

    with o_col:
        st.markdown(f"""
        <div class="kpi-row" style="margin-top:0;">
            <div class="kpi-card"><div class="kpi-lbl">Rows</div><div class="kpi-val">{fmt_num(ov['n_rows'])}</div></div>
            <div class="kpi-card"><div class="kpi-lbl">Columns</div><div class="kpi-val">{ov['n_cols']}</div></div>
            <div class="kpi-card"><div class="kpi-lbl">Numeric</div>
                <div class="kpi-val" style="color:var(--accent);">{dc.get('numeric',0)}</div></div>
            <div class="kpi-card"><div class="kpi-lbl">Categorical</div>
                <div class="kpi-val" style="color:var(--green);">{dc.get('categorical',0)}</div></div>
            <div class="kpi-card"><div class="kpi-lbl">Datetime</div>
                <div class="kpi-val" style="color:var(--yellow);">{dc.get('datetime',0)}</div></div>
        </div>
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-lbl">Missing</div>
                <div class="kpi-val">{fmt_num(ov['missing_cells'])}</div>
                <div class="kpi-sub">{ov['missing_pct']}% of cells</div></div>
            <div class="kpi-card"><div class="kpi-lbl">Duplicates</div>
                <div class="kpi-val">{ov['dup_count']}</div></div>
            <div class="kpi-card"><div class="kpi-lbl">Memory</div>
                <div class="kpi-val">{fmt_bytes(ov['memory_bytes'])}</div></div>
        </div>""", unsafe_allow_html=True)

        targets = meta.get("target_candidates", [])
        if targets:
            st.markdown('<div style="font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:14px 0 8px;">🎯 ML Target Candidates</div>',
                        unsafe_allow_html=True)
            for t in targets:
                st.markdown(f'<div style="font-size:13px;padding:5px 0;border-bottom:1px solid rgba(48,54,61,.4);">'
                            f'<span class="mono">{t["name"]}</span>'
                            f'<span style="font-size:11px;color:var(--muted);margin-left:8px;">— {t["reason"]}</span></div>',
                            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def render_col_table(col_list):
        if not col_list:
            st.markdown('<div style="color:var(--muted);padding:20px;">No columns in this category.</div>',
                        unsafe_allow_html=True)
            return
        rows_html = ""
        for cm in col_list:
            badge = DTYPE_BADGE.get(cm["dtype_category"], "")
            mc    = "var(--red)" if cm["missing_pct"]>30 else "var(--yellow)" if cm["missing_pct"]>5 else "var(--muted)"
            stats = ""
            if cm["dtype_category"] == "numeric" and "numeric_stats" in cm:
                ns = cm["numeric_stats"]
                stats = f'<div class="mono" style="font-size:11px;color:var(--muted);">min {ns.get("min","—")} · mean {ns.get("mean","—")} · max {ns.get("max","—")}</div>'
                if ns.get("outliers",0): stats += f'<div style="font-size:11px;color:var(--yellow);">⚠️ {ns["outliers"]} outlier(s)</div>'
            elif cm["dtype_category"] == "categorical" and "categorical_stats" in cm:
                top = cm["categorical_stats"].get("top_values",[])
                if top: stats = f'<div class="mono" style="font-size:11px;color:var(--muted);">Top: {", ".join(v["value"] for v in top[:2])}</div>'
            samples = ", ".join(str(v) for v in cm.get("sample_values",[])[:3])
            rows_html += f"""<tr>
              <td class="mono" style="font-size:12px;font-weight:500;">{cm['name']}</td>
              <td>{badge}</td>
              <td style="color:var(--muted);font-size:12px;">{cm['dtype']}</td>
              <td><span style="color:{mc};">{cm['missing_pct']:.1f}%</span>
                  <div style="font-size:11px;color:var(--dim);">{cm['missing_count']} cells</div></td>
              <td class="mono" style="font-size:12px;">{fmt_num(cm['n_unique'])}</td>
              <td><div class="mono" style="font-size:11px;color:var(--dim);">{samples}</div>{stats}</td>
              <td style="font-size:12px;color:var(--muted);">{cm['recommendation']}</td>
            </tr>"""
        st.markdown(f"""
        <div class="card" style="overflow:auto;">
          <table class="col-table"><thead><tr>
            <th>Column</th><th>Type</th><th>Dtype</th><th>Missing</th>
            <th>Unique</th><th>Sample / Stats</th><th>Recommendation</th>
          </tr></thead><tbody>{rows_html}</tbody></table>
        </div>""", unsafe_allow_html=True)

    t_all, t_num, t_cat, t_iss = st.tabs([
        f"📋 All ({len(cols)})",
        f"🔢 Numeric ({dc.get('numeric',0)})",
        f"🏷️ Categorical ({dc.get('categorical',0)})",
        "⚠️ Issues",
    ])
    with t_all:
        s = st.text_input("🔍 Filter by column name", placeholder="type to filter…", label_visibility="collapsed")
        render_col_table([c for c in cols if s.lower() in c["name"].lower()] if s else cols)
    with t_num:
        render_col_table([c for c in cols if c["dtype_category"]=="numeric"])
    with t_cat:
        render_col_table([c for c in cols if c["dtype_category"]=="categorical"])
    with t_iss:
        iss_cols = [c for c in cols if c["missing_pct"]>0
                    or c.get("numeric_stats",{}).get("outliers",0)>0
                    or (c["dtype_category"]=="categorical" and c["n_unique"]==ov["n_rows"])]
        if iss_cols: render_col_table(iss_cols)
        else: st.markdown('<div style="text-align:center;padding:60px;color:var(--green);font-size:18px;font-weight:600;">✅ No issues detected</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANING PAGE  (Phase 4)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Cleaning":
    df      = st.session_state.df
    meta    = st.session_state.metadata
    issues  = st.session_state.cleaning_issues
    ov      = meta["overview"]

    st.markdown(f"""
    <div class="hero">
        <div class="hero-t">Data <span>Cleaning</span> Engine</div>
        <div class="hero-s">Automatically detected issues in <strong>{ov['filename']}</strong>.
        Review the recommended fixes, adjust if needed, then apply.</div>
    </div>""", unsafe_allow_html=True)

    sm = issues.get("summary", {})

    # ── Summary bar ──────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-lbl">Total Issues</div>
            <div class="kpi-val" style="color:{'var(--red)' if sm.get('total_issues',0)>5 else 'var(--yellow)'};">
                {sm.get('total_issues',0)}</div>
            <div class="kpi-sub">detected automatically</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Missing Value Issues</div>
            <div class="kpi-val" style="color:var(--yellow);">{sm.get('missing_issues',0)}</div>
            <div class="kpi-sub">columns affected</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Duplicate Rows</div>
            <div class="kpi-val" style="color:{'var(--yellow)' if issues['duplicates']['count']>0 else 'var(--green)'};">
                {issues['duplicates']['count']}</div>
            <div class="kpi-sub">{issues['duplicates']['pct']}% of rows</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Outlier Issues</div>
            <div class="kpi-val" style="color:var(--yellow);">{sm.get('outlier_issues',0)}</div>
            <div class="kpi-sub">numeric columns</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Type Issues</div>
            <div class="kpi-val" style="color:var(--yellow);">{sm.get('type_issues',0)}</div>
            <div class="kpi-sub">columns to convert</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Already cleaned?
    if st.session_state.cleaned_df is not None:
        rep = st.session_state.cleaning_report
        st.markdown("""
        <div style="background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.3);
                    border-radius:12px;padding:16px 20px;margin-bottom:24px;
                    display:flex;align-items:center;gap:12px;">
            <span style="font-size:22px;">✅</span>
            <div>
                <div style="font-size:14px;font-weight:600;color:var(--green);">
                    Dataset already cleaned</div>
                <div style="font-size:12px;color:var(--muted);">
                    Applied at {applied_at} · {total_actions} action(s) taken.
                    Scroll down to see the report or re-apply with different settings.
                </div>
            </div>
        </div>""".replace("{applied_at}", rep.get("applied_at","—"))
                   .replace("{total_actions}", str(rep.get("total_actions",0))),
                   unsafe_allow_html=True)

    if sm.get("total_issues", 0) == 0:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:var(--green);font-size:18px;font-weight:600;">
            ✅ No cleaning issues detected — dataset is ready for EDA!
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Build the cleaning plan UI ────────────────────────────
    st.markdown("""
    <div style="font-size:16px;font-weight:600;margin:24px 0 16px;">
        ⚙️ Configure Cleaning Plan
        <span style="font-size:12px;font-weight:400;color:var(--muted);margin-left:10px;">
        — pre-filled with recommended fixes. Adjust anything before applying.
        </span>
    </div>""", unsafe_allow_html=True)

    plan = {"missing": {}, "duplicates": {}, "outliers": {}, "types": {}}

    tabs = st.tabs(["🔴 Missing Values", "🟡 Duplicates", "🟠 Outliers", "🔵 Type Issues"])

    # ── Tab 1: Missing Values ─────────────────────────────────
    with tabs[0]:
        missing_list = issues.get("missing", [])
        if not missing_list:
            st.markdown('<div style="color:var(--green);padding:20px;">✅ No missing values found.</div>',
                        unsafe_allow_html=True)
        else:
            for item in missing_list:
                col_name = item["col"]
                sev_color = "var(--red)" if item["missing_pct"]>30 else "var(--yellow)" if item["missing_pct"]>10 else "var(--muted)"
                severity  = "HIGH" if item["missing_pct"]>30 else "MEDIUM" if item["missing_pct"]>10 else "LOW"

                st.markdown(f"""
                <div class="issue-card">
                    <div class="issue-header">
                        <span class="issue-col-name">{col_name}</span>
                        <span class="badge b-{'num' if item['dtype_category']=='numeric' else 'cat'}">{item['dtype_category'].upper()}</span>
                        <span style="font-size:11px;font-weight:600;color:{sev_color};background:rgba(0,0,0,.2);
                                     border:1px solid {sev_color};border-radius:4px;padding:2px 7px;">{severity}</span>
                        <span class="issue-stat" style="margin-left:auto;">
                            {item['missing_count']} missing · {item['missing_pct']:.1f}%
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

                c1, c2 = st.columns([2, 1])
                with c1:
                    opts     = item["fix_options"]
                    opt_lbls = [FIX_LABELS.get(o, o) for o in opts]
                    rec_idx  = opts.index(item["recommended_fix"]) if item["recommended_fix"] in opts else 0
                    chosen   = st.selectbox(
                        f"Fix for **{col_name}**",
                        options=opt_lbls,
                        index=rec_idx,
                        key=f"miss_{col_name}",
                        label_visibility="collapsed",
                        help=f"Recommended: {FIX_LABELS.get(item['recommended_fix'], item['recommended_fix'])}"
                    )
                    action   = opts[opt_lbls.index(chosen)]

                with c2:
                    const_val = str(item.get("fill_value","Unknown"))
                    if action == "fill_constant":
                        const_val = st.text_input(f"Constant for {col_name}",
                                                  value=const_val,
                                                  key=f"const_{col_name}",
                                                  label_visibility="collapsed")

                plan["missing"][col_name] = {"action": action, "constant": const_val}
                st.markdown("<hr style='border-color:var(--border);margin:4px 0 12px;'>",
                            unsafe_allow_html=True)

    # ── Tab 2: Duplicates ─────────────────────────────────────
    with tabs[1]:
        dup = issues["duplicates"]
        if dup["count"] == 0:
            st.markdown('<div style="color:var(--green);padding:20px;">✅ No duplicate rows found.</div>',
                        unsafe_allow_html=True)
            plan["duplicates"]["action"] = "keep"
        else:
            st.markdown(f"""
            <div class="issue-card">
                <div class="issue-header">
                    <span class="issue-col-name">Duplicate Rows</span>
                    <span style="font-size:12px;color:var(--muted);margin-left:auto;">
                        {dup['count']} duplicates · {dup['pct']:.2f}% of dataset
                    </span>
                </div>
                <div style="font-size:13px;color:var(--muted);">
                    These are rows where every column value is identical to another row.
                    Keeping them inflates statistics and model performance metrics.
                </div>
            </div>""", unsafe_allow_html=True)

            dup_choice = st.radio(
                "Duplicate action",
                options=["Remove Duplicate Rows", "Keep As-Is"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )
            plan["duplicates"]["action"] = "drop_duplicates" if "Remove" in dup_choice else "keep"

    # ── Tab 3: Outliers ───────────────────────────────────────
    with tabs[2]:
        outlier_list = issues.get("outliers", [])
        if not outlier_list:
            st.markdown('<div style="color:var(--green);padding:20px;">✅ No outliers detected.</div>',
                        unsafe_allow_html=True)
        else:
            for item in outlier_list:
                col_name = item["col"]
                st.markdown(f"""
                <div class="issue-card">
                    <div class="issue-header">
                        <span class="issue-col-name">{col_name}</span>
                        <span class="badge b-num">NUMERIC</span>
                        <span style="font-size:12px;color:var(--muted);margin-left:auto;">
                            {item['count']} outlier(s) · {item['pct']:.1f}% of values
                        </span>
                    </div>
                    <div style="font-size:12px;color:var(--muted);">
                        IQR fences: lower <span class="mono">{item['lower_fence']}</span>
                        · upper <span class="mono">{item['upper_fence']}</span>
                        &nbsp;|&nbsp; Q1 <span class="mono">{item['q1']}</span>
                        · Q3 <span class="mono">{item['q3']}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                opts     = item["fix_options"]
                opt_lbls = [FIX_LABELS.get(o, o) for o in opts]
                rec_idx  = opts.index(item["recommended_fix"]) if item["recommended_fix"] in opts else 0
                chosen   = st.selectbox(
                    f"Outlier fix for {col_name}",
                    options=opt_lbls,
                    index=rec_idx,
                    key=f"out_{col_name}",
                    label_visibility="collapsed",
                )
                plan["outliers"][col_name] = {"action": opts[opt_lbls.index(chosen)]}
                st.markdown("<hr style='border-color:var(--border);margin:4px 0 12px;'>",
                            unsafe_allow_html=True)

    # ── Tab 4: Type Issues ────────────────────────────────────
    with tabs[3]:
        type_list = issues.get("type_issues", [])
        if not type_list:
            st.markdown('<div style="color:var(--green);padding:20px;">✅ No data type issues detected.</div>',
                        unsafe_allow_html=True)
        else:
            for item in type_list:
                col_name = item["col"]
                st.markdown(f"""
                <div class="issue-card">
                    <div class="issue-header">
                        <span class="issue-col-name">{col_name}</span>
                        <span class="badge b-txt">OBJECT</span>
                        <span style="font-size:12px;color:var(--muted);margin-left:auto;">
                            Suggested → <strong>{item['suggested_dtype']}</strong>
                        </span>
                    </div>
                    <div style="font-size:12px;color:var(--muted);">{item['reason']}</div>
                </div>""", unsafe_allow_html=True)

                opts     = [item["action"], "keep"]
                opt_lbls = [FIX_LABELS.get(o, o) for o in opts]
                chosen   = st.selectbox(
                    f"Type fix for {col_name}",
                    options=opt_lbls,
                    index=0,
                    key=f"type_{col_name}",
                    label_visibility="collapsed",
                )
                plan["types"][col_name] = {"action": opts[opt_lbls.index(chosen)]}
                st.markdown("<hr style='border-color:var(--border);margin:4px 0 12px;'>",
                            unsafe_allow_html=True)

    # ── Apply button ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        apply_clicked = st.button("🧹  Apply Cleaning Plan", type="primary",
                                  use_container_width=True)

    if apply_clicked:
        with st.spinner("Applying cleaning plan…"):
            try:
                cleaned_df, report = apply_cleaning(df, plan)
                st.session_state.cleaned_df      = cleaned_df
                st.session_state.cleaning_report = report
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Cleaning failed: {e}")

    # ── Cleaning Report (shown after apply) ───────────────────
    if st.session_state.cleaning_report:
        rep = st.session_state.cleaning_report

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:16px;font-weight:600;margin-bottom:16px;">
            📋 Cleaning Report
        </div>""", unsafe_allow_html=True)

        # Before / After comparison
        b_col, a_col, r_col, c_col = st.columns(4)
        metrics = [
            (b_col, "Rows Before",    rep["rows_before"],  "—",               "var(--muted)"),
            (a_col, "Rows After",     rep["rows_after"],   "—",               "var(--green)"),
            (r_col, "Rows Removed",   rep["rows_removed"],
             f"-{rep['rows_removed']}" if rep["rows_removed"] else "none",    "var(--yellow)"),
            (c_col, "Actions Taken",  rep["total_actions"],"—",               "var(--accent)"),
        ]
        for col_w, lbl, val, delta, color in metrics:
            with col_w:
                st.markdown(f"""
                <div class="compare-box">
                    <div class="compare-lbl">{lbl}</div>
                    <div class="compare-val" style="color:{color};">{fmt_num(val)}</div>
                    <div class="compare-delta" style="color:var(--dim);">{delta}</div>
                </div>""", unsafe_allow_html=True)

        # Step-by-step log
        st.markdown("<br>", unsafe_allow_html=True)
        steps = rep.get("steps", [])
        if steps:
            rows_html = "".join(
                f"""<tr>
                  <td style="color:var(--accent);">{s['step']}</td>
                  <td class="mono" style="font-size:12px;">{s['column']}</td>
                  <td style="color:var(--muted);">{s['action']}</td>
                  <td style="color:var(--muted);">{s['impact']}</td>
                  <td style="font-size:18px;">{s['status']}</td>
                </tr>"""
                for s in steps
            )
            st.markdown(f"""
            <div class="card" style="overflow:auto;">
              <table class="report-table">
                <thead><tr>
                  <th>Step</th><th>Column</th><th>Action Taken</th>
                  <th>Impact</th><th>Status</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

        # Preview of cleaned dataset
        st.markdown("<br>", unsafe_allow_html=True)
        cleaned_df = st.session_state.cleaned_df
        st.markdown(f"""
        <div class="prev-card">
            <div class="prev-hdr">
                <span class="prev-t">✅ Cleaned Dataset Preview</span>
                <span class="prev-m">First 10 rows · {len(cleaned_df)} rows · {cleaned_df.shape[1]} columns</span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(cleaned_df.head(10), use_container_width=True, height=280)

        # Download cleaned CSV
        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        dl_name   = (st.session_state.filename or "dataset").replace(".csv","") + "_cleaned.csv"
        st.download_button(
            label="⬇️  Download Cleaned CSV",
            data=csv_bytes,
            file_name=dl_name,
            mime="text/csv",
            use_container_width=False,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER PAGES  (Phases 5–11)
# ══════════════════════════════════════════════════════════════════════════════
else:
    COMING = {
        "EDA":                ("📊", "Phase 5", "EDA Engine",
                               "Statistics, correlation matrix, skewness, kurtosis, and distribution analysis."),
        "ML Models":          ("🤖", "Phase 7", "Machine Learning Engine",
                               "AutoML — trains and compares regression, classification, and clustering models."),
        "Feature Importance": ("⚡", "Phase 7", "Feature Importance",
                               "Which columns matter most for your prediction target and why."),
        "AI Insights":        ("💡", "Phase 8", "Insight Generation Engine",
                               "The LLM reads your analysis and writes plain-English business insights."),
        "Reports":            ("📄", "Phase 11","Report Generation",
                               "Auto-generated PDF and PowerPoint ready to hand to a client or manager."),
        "Chat":               ("💬", "Phase 10","Chat With Dataset",
                               "Ask questions in plain language — SQL Agent answers from your data."),
    }
    icon, phase_tag, title, desc = COMING.get(
        page, ("🚧", "Coming Soon", page, "This module is under development."))
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:60vh;text-align:center;padding:60px 40px;">
        <div style="font-size:56px;margin-bottom:20px;">{icon}</div>
        <div style="font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
                    color:var(--accent);margin-bottom:10px;">{phase_tag}</div>
        <div style="font-size:28px;font-weight:700;margin-bottom:12px;">{title}</div>
        <div style="font-size:15px;color:var(--muted);max-width:480px;line-height:1.6;">{desc}</div>
        <div style="margin-top:28px;padding:10px 24px;border:1px solid var(--border);
                    border-radius:8px;font-size:13px;color:var(--dim);">
            🔨 Coming in the next phase of development
        </div>
    </div>""", unsafe_allow_html=True)