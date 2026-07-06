"""
frontend/sidebar.py
Sidebar navigation. Extracted verbatim from upload.py.
"""

import streamlit as st


def render_sidebar():
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

    return locked
