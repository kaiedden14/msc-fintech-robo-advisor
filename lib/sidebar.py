"""Persistent sidebar + per-page footer for the dashboard chrome.

Phase 1 implementation: functional with locked colour tokens. Polish
(pixel-perfect spacing, transitions, refined typography) deferred to
Phase 8.
"""

import streamlit as st


_STEPS: list[tuple[str, str]] = [
    ("Landing",             "pages/1_landing.py"),
    ("Risk Profile",        "pages/2_risk_profile.py"),
    ("Asset Selection",     "pages/3_asset_selection.py"),
    ("Diversification",     "pages/4_diversification.py"),
    ("Optimised Portfolio", "pages/5_optimised_portfolio.py"),
    ("Forward Projection",  "pages/6_backtest.py"),
    ("Rebalancing",         "pages/7_rebalancing.py"),
]


def render_sidebar(current_page) -> None:
    """Render the persistent sidebar.

    Parameters
    ----------
    current_page : st.navigation result (StreamlitPage)
        Used to highlight the active step.
    """
    with st.sidebar:
        # Brand
        st.markdown(
            "<div class='ra-brand'>Hybrid<br/>Robo-Advisor</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='ra-section-label'>Steps</div>", unsafe_allow_html=True)
        for i, (label, path) in enumerate(_STEPS, start=1):
            is_active = (current_page.title == label)
            clicked = st.button(
                f"{i}.  {label}",
                key=f"nav_step_{i}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )
            if clicked and not is_active:
                st.switch_page(path)

        st.markdown("<div class='ra-section-label'>Session</div>", unsafe_allow_html=True)
        _render_session_card()

        # Footer disclaimer (sidebar copy; per-page disclaimer rendered separately)
        st.markdown(
            "<div class='ra-footer-sidebar'>Academic prototype — not financial advice</div>",
            unsafe_allow_html=True,
        )


def _render_session_card() -> None:
    rp = st.session_state.get("risk_profile") or "—"
    amt = st.session_state.get("investment_amount")
    amt_str = f"£{amt:,.0f}" if amt else "—"
    n_sel = len(st.session_state.get("selected_tickers") or [])

    # Colour the selection count amber when outside the 5-15 range
    if n_sel == 0:
        sel_str = "—"
    elif 5 <= n_sel <= 15:
        sel_str = f"{n_sel}"
    else:
        sel_str = f"<span style='color:#C97A1F'>{n_sel}</span>"

    with st.container(border=True):
        st.markdown(
            f"**Risk profile** &nbsp;&nbsp;&nbsp;&nbsp; {rp}<br/>"
            f"**Investment** &nbsp;&nbsp;&nbsp;&nbsp; {amt_str}<br/>"
            f"**Selected** &nbsp;&nbsp;&nbsp;&nbsp; {sel_str}",
            unsafe_allow_html=True,
        )


def render_page_footer() -> None:
    """Render the per-page footer disclaimer. Call at the bottom of every page."""
    st.markdown(
        "<div class='ra-footer-page'>Academic prototype — not financial advice</div>",
        unsafe_allow_html=True,
    )
