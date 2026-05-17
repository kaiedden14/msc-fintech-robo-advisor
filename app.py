"""Hybrid Robo-Advisor — Streamlit entry point.

Loads theme + session state, then defines the navigation and delegates
page rendering to the selected st.Page. The auto-generated nav widget is
hidden so we can render a custom sidebar matching the design mockup.
"""

import streamlit as st

from lib.state import init_state
from lib.theme import inject_theme
from lib.sidebar import render_sidebar
from lib.logger import log_event


st.set_page_config(
    page_title="Hybrid Robo-Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
init_state()

# Log session_start exactly once per session (the boolean flag in state
# survives reruns but is reset on a browser reload, which is when a new
# session_id is also minted — so the pairing stays consistent).
if not st.session_state["session_start_logged"]:
    log_event("session_start")
    st.session_state["session_start_logged"] = True

pages = [
    st.Page("pages/1_landing.py",            title="Landing",             url_path="landing",       default=True),
    st.Page("pages/2_risk_profile.py",       title="Risk Profile",        url_path="risk-profile"),
    st.Page("pages/3_asset_selection.py",    title="Asset Selection",     url_path="asset-selection"),
    st.Page("pages/4_diversification.py",    title="Diversification",     url_path="diversification"),
    st.Page("pages/5_optimised_portfolio.py", title="Optimised Portfolio", url_path="optimised"),
    st.Page("pages/6_backtest.py",           title="Forward Projection",  url_path="projection"),
    st.Page("pages/7_rebalancing.py",        title="Rebalancing",         url_path="rebalancing"),
]

nav = st.navigation(pages, position="hidden")

render_sidebar(current_page=nav)

nav.run()
