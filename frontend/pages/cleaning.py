"""
frontend/pages/cleaning.py
Cleaning page (Phase 4 — Data Cleaning Engine + Cleaning page).
Extracted verbatim from upload.py.
"""

import time
import streamlit as st

from frontend.session import fmt_num, FIX_LABELS

try:
    from services.cleaning_service import apply_cleaning
except ImportError:
    def apply_cleaning(df, plan):
        return df.copy(), {"steps": [], "rows_before": len(df), "rows_after": len(df),
                           "rows_removed": 0, "cols_before": df.shape[1],
                           "cols_after": df.shape[1], "total_actions": 0,
                           "applied_at": ""}


def render():
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
