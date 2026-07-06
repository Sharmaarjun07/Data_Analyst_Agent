"""
frontend/pages/_placeholder.py
Shared "coming soon" renderer used by pages for phases that have not
been built yet (Phases 5-11). Extracted verbatim from upload.py's
PLACEHOLDER PAGES block, split so each page file can supply its own
icon/phase/title/description.
"""

import streamlit as st


def render_placeholder(icon: str, phase_tag: str, title: str, desc: str):
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
