"""
frontend/pages/dashboard.py
Dashboard page (Phase 2 — File Upload / Dashboard). Extracted verbatim
from upload.py.
"""

import time
import pandas as pd
import streamlit as st

from frontend.session import fmt_num, GRADE_COLOR

try:
    from services.metadata_service import extract_metadata
except ImportError:
    def extract_metadata(df, filename="dataset.csv"):
        return {"overview": {}, "quality": {}, "columns": [], "target_candidates": []}

try:
    from services.cleaning_service import analyze_issues
except ImportError:
    def analyze_issues(df, metadata):
        return {"summary": {}, "missing": [], "duplicates": {}, "outliers": [], "type_issues": []}


def render(locked: bool):
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
