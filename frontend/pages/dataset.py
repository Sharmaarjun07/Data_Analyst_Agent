"""
frontend/pages/dataset.py
Dataset page (Phase 3 — Metadata Engine + Dataset page). Extracted
verbatim from upload.py.
"""

import streamlit as st

from frontend.session import fmt_num, fmt_bytes, DTYPE_BADGE, GRADE_COLOR


def render():
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
