"""
frontend/styles.py
Global CSS for the Data Analyst Agent app.
Extracted verbatim from the original upload.py GLOBAL CSS block.
"""

import streamlit as st


def load_css():
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
