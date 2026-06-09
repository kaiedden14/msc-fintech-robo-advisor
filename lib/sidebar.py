"""Persistent sidebar + per-page footer for the dashboard chrome.

Phase 1 implementation: functional with locked colour tokens. Polish
(pixel-perfect spacing, transitions, refined typography) deferred to
Phase 8.
"""

from pathlib import Path

import streamlit as st


_STEPS: list[tuple[str, str]] = [
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

        # Home, discreet link back to the landing page from anywhere in the flow
        if st.button(
            "Home",
            key="nav_home",
            use_container_width=True,
            type="secondary",
        ):
            st.switch_page("pages/1_landing.py")

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

        st.markdown("<div class='ra-section-label'>Data</div>", unsafe_allow_html=True)
        _render_data_section()

        # Footer disclaimer (sidebar copy; per-page disclaimer rendered separately)
        st.markdown(
            "<div class='ra-footer-sidebar'>Simulated allocations, not financial advice</div>",
            unsafe_allow_html=True,
        )


_VALID_PARTICIPANT_IDS = ["Test"] + [f"P{i:02d}" for i in range(1, 11)]  # Test + P01..P10


def _render_session_card() -> None:
    """Render the Session card in the sidebar.

    Shows the participant selector, hydrates session state from a saved
    portfolio if one exists for the selected participant, and displays a
    summary card with the current risk profile, investment amount, and
    number of selected stocks. Participant identity lives in the sidebar
    rather than on the landing page so the dashboard reads as a product
    rather than a research instrument.
    """
    from lib.logger import log_event

    # Participant ID, set by the researcher at the start of each session.
    # Lives in the sidebar rather than the participant-facing landing page
    # so the dashboard reads as a product, not a research instrument.
    current_pid = st.session_state.get("participant_id")
    selected_pid = st.selectbox(
        "Participant",
        options=[""] + _VALID_PARTICIPANT_IDS,
        index=(_VALID_PARTICIPANT_IDS.index(current_pid) + 1)
        if current_pid in _VALID_PARTICIPANT_IDS
        else 0,
        placeholder="Select…",
        key="sidebar_participant_pid",
    )
    if selected_pid and selected_pid != current_pid:
        st.session_state["participant_id"] = selected_pid
        log_event("participant_id_set", previous=current_pid, new=selected_pid)

        # If this participant has a saved portfolio on disk, hydrate the
        # session state from it so they can navigate directly to Rebalancing
        # and view their existing allocation + accumulated performance.
        from lib.persistence import load_portfolio
        saved = load_portfolio(selected_pid)
        if saved is not None:
            st.session_state["risk_profile"] = saved.get("risk_band")
            st.session_state["investment_amount"] = float(
                saved.get("investment_amount") or 0.0
            )
            st.session_state["selected_tickers"] = list(
                saved.get("selected_tickers", [])
            )
            st.session_state["optimised_weights"] = {
                k: float(v) for k, v in saved.get("weights", {}).items()
            }
            st.session_state["user_modified_weights"] = None
            st.session_state["decision"] = saved.get("decision")
            # Each participant load is a fresh view, clear any prior
            # session's confirmation state so the Confirm button reappears.
            st.session_state["portfolio_confirmed"] = False
            log_event(
                "saved_portfolio_loaded",
                participant_id=selected_pid,
                investment_date=saved.get("investment_date"),
            )
            st.toast(
                f"Loaded {selected_pid}'s saved portfolio.",
                icon="✅",
            )
            st.rerun()

    rp = st.session_state.get("risk_profile") or "–"
    amt = st.session_state.get("investment_amount")
    amt_str = f"£{amt:,.0f}" if amt else "–"
    n_sel = len(st.session_state.get("selected_tickers") or [])

    # Colour the selection count amber when outside the 5-15 range
    if n_sel == 0:
        sel_str = "–"
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
        "<div class='ra-footer-page'>Simulated allocations, not financial advice</div>",
        unsafe_allow_html=True,
    )


def _render_data_section() -> None:
    """Render the Data card (Prices + Recommendation dates + Refresh button)."""
    from lib.data import load_prices_clean, load_snapshot_metadata

    try:
        prices_date = load_prices_clean().index.max().date().isoformat()
    except Exception:
        prices_date = "–"
    try:
        rec_date = load_snapshot_metadata().get("snapshot_date", "–")
    except Exception:
        rec_date = "–"

    with st.container(border=True):
        st.markdown(
            f"**Prices** &nbsp;&nbsp;&nbsp;&nbsp; {prices_date}<br/>"
            f"**Recommendation** &nbsp;&nbsp;&nbsp;&nbsp; {rec_date}",
            unsafe_allow_html=True,
        )

    if st.button("Refresh data", use_container_width=True, key="refresh_data_btn"):
        _handle_refresh_data()


def _handle_refresh_data() -> None:
    """Run the daily pipeline (prices -> features -> snapshot) on click.

    Wrapped in try/except so a yfinance failure leaves the dashboard with
    its cached state intact rather than crashing. st.rerun is called only
    after the success path completes so the failure branch's error toast
    survives.
    """
    from src.data.ingest import refresh_prices, refresh_features
    from src.models.snapshot import build_snapshot
    from lib.data import (
        load_prices_clean, load_snapshot_metadata, invalidate_caches,
    )
    from lib.logger import log_event

    # Capture before-state for the log payload
    try:
        prices_before = load_prices_clean().index.max().date().isoformat()
    except Exception:
        prices_before = None
    try:
        snap_before = load_snapshot_metadata().get("snapshot_date")
    except Exception:
        snap_before = None

    raw_path = Path("data/raw/prices.parquet")
    clean_path = Path("data/processed/prices_clean.parquet")
    features_path = Path("data/processed/features.parquet")
    return_model_path = Path("models/rf_return_v6.joblib")
    vol_model_path = Path("models/rf_volatility_v6.joblib")
    output_dir = Path("data/processed")

    success = False
    snap_meta_after: dict = {}
    price_status: dict = {}
    try:
        with st.spinner("Fetching latest prices from yfinance…"):
            price_status = refresh_prices(raw_path, clean_path)
        with st.spinner("Rebuilding feature panel…"):
            refresh_features(clean_path, features_path)
        with st.spinner("Generating recommendations…"):
            snap_meta_after = build_snapshot(
                return_model_path=return_model_path,
                vol_model_path=vol_model_path,
                features_path=features_path,
                output_dir=output_dir,
            )
        invalidate_caches()
        log_event(
            "data_refreshed",
            prices_before=prices_before,
            prices_after=str(price_status.get("new_max_date", "")),
            snapshot_before=snap_before,
            snapshot_after=snap_meta_after.get("snapshot_date"),
            new_rows=int(price_status.get("new_rows", 0)),
        )
        success = True
    except Exception as e:
        log_event("data_refresh_failed", error=str(e))
        st.error(f"Refresh failed: {e}")

    if success:
        if price_status.get("no_new_data"):
            st.toast(
                f"No new prices since {prices_before}. Recommendation rebuilt "
                f"for {snap_meta_after.get('snapshot_date')}."
            )
        else:
            st.toast(
                f"Refreshed: {price_status.get('new_rows', 0)} new trading day(s). "
                f"Recommendation as of {snap_meta_after.get('snapshot_date')}."
            )
        st.rerun()
