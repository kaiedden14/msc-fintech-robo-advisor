"""Asset Selection — Phase 3.

Universe display with search, sector filter, selection table, and a
per-stock detail panel showing price chart + plain-language SHAP card.
Enforces 5-15 selection constraint via Continue-button gating.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.copy import shap_reasons
from lib.data import (
    load_predictions,
    load_prices_clean,
    load_shap_return,
    load_universe_metadata,
)
from lib.logger import log_event
from lib.sidebar import render_page_footer
from lib.state import clear_downstream_of


_MIN_SEL, _MAX_SEL = 5, 15


# ---------- Guard: prior pages must be complete ----------

if not st.session_state["risk_profile"]:
    st.title("Asset Selection")
    st.warning("Please complete the Risk Profile first.")
    if st.button("Back to Risk Profile"):
        st.switch_page("pages/2_risk_profile.py")
    render_page_footer()
    st.stop()


# ---------- Header + intro ----------

st.title("Asset Selection")
st.caption(
    f"Pick {_MIN_SEL}-{_MAX_SEL} stocks from the FTSE 100. "
    "Predictions and explanations refresh whenever you press "
    "**Refresh data** in the sidebar."
)


# ---------- Load data ----------

universe = load_universe_metadata()
predictions = load_predictions()
prices_close = load_prices_clean()["Close"]
shap_return_df = load_shap_return()

# The selectable universe is whatever the snapshot covers — 93 stock tickers
universe_tickers: list[str] = sorted(predictions.index.tolist())

# Latest close + 1-day change (handle ^FTSE / ^VIX presence gracefully)
latest_date = prices_close.index.max()
# Find the previous trading day before latest_date for each ticker
prev_date = prices_close.index[-2]
last_close_all = prices_close.loc[latest_date]
prev_close_all = prices_close.loc[prev_date]

last_close = last_close_all.reindex(universe_tickers)
prev_close = prev_close_all.reindex(universe_tickers)
daily_change_pct = (last_close - prev_close) / prev_close * 100


# ---------- Build display table ----------

selected_set = set(st.session_state["selected_tickers"])

table = pd.DataFrame(
    {
        "Selected":         [t in selected_set for t in universe_tickers],
        "Ticker":           universe_tickers,
        "Company":          [
            universe.loc[t, "name"] if t in universe.index else t
            for t in universe_tickers
        ],
        "Sector":           [
            universe.loc[t, "sector"] if t in universe.index else "—"
            for t in universe_tickers
        ],
        "Last close (GBp)": last_close.values,
        "1-day change %":   daily_change_pct.values,
        "Pred. return %":   predictions.loc[universe_tickers, "predicted_return"].values * 100,
        "Pred. vol %":      predictions.loc[universe_tickers, "predicted_vol"].values * 100,
    }
)


# ---------- Filters ----------

filt_col_a, filt_col_b = st.columns([2, 3], gap="medium")
with filt_col_a:
    search = st.text_input(
        "Search ticker or company",
        placeholder="HSBA or HSBC",
        key="asset_search",
    )
with filt_col_b:
    sectors_available = sorted(
        s for s in table["Sector"].dropna().unique() if s and s != "—"
    )
    sector_filter = st.multiselect(
        "Filter by sector",
        options=sectors_available,
        key="asset_sector_filter",
    )

filtered = table.copy()
if search:
    s = search.strip().lower()
    mask = (
        filtered["Ticker"].str.lower().str.contains(s, na=False)
        | filtered["Company"].str.lower().str.contains(s, na=False)
    )
    filtered = filtered[mask]
if sector_filter:
    filtered = filtered[filtered["Sector"].isin(sector_filter)]


# ---------- Selection table ----------

edited = st.data_editor(
    filtered,
    column_config={
        "Selected": st.column_config.CheckboxColumn(default=False, width="small"),
        "Ticker": st.column_config.TextColumn(width="small"),
        "Company": st.column_config.TextColumn(width="medium"),
        "Sector": st.column_config.TextColumn(width="medium"),
        "Last close (GBp)": st.column_config.NumberColumn(format="%.2f", width="small"),
        "1-day change %":   st.column_config.NumberColumn(format="%+.2f", width="small"),
        "Pred. return %":   st.column_config.NumberColumn(format="%+.2f", width="small"),
        "Pred. vol %":      st.column_config.NumberColumn(format="%.2f", width="small"),
    },
    hide_index=True,
    use_container_width=True,
    disabled=["Ticker", "Company", "Sector", "Last close (GBp)",
              "1-day change %", "Pred. return %", "Pred. vol %"],
    key="asset_table_editor",
    height=420,
)

# Diff against state — only among VISIBLE tickers — and update
visible_tickers = set(filtered["Ticker"])
old_visible_sel = selected_set & visible_tickers
new_visible_sel = set(edited.loc[edited["Selected"], "Ticker"])

added = new_visible_sel - old_visible_sel
removed = old_visible_sel - new_visible_sel

if added or removed:
    new_full = sorted((selected_set - removed) | added)
    for t in added:
        log_event("ticker_selected", ticker=t, current_selection_count=len(new_full))
    for t in removed:
        log_event("ticker_deselected", ticker=t, current_selection_count=len(new_full))
    st.session_state["selected_tickers"] = new_full
    clear_downstream_of("selected_tickers")
    st.rerun()


# ---------- Selection counter + info ----------

n = len(st.session_state["selected_tickers"])
counter_col, info_col = st.columns([1, 2], gap="medium")
with counter_col:
    if n == 0:
        st.markdown(f"**0** of {_MIN_SEL}-{_MAX_SEL} selected")
    elif _MIN_SEL <= n <= _MAX_SEL:
        st.markdown(f"**{n}** of {_MIN_SEL}-{_MAX_SEL} selected")
    elif n < _MIN_SEL:
        st.markdown(
            f"<span style='color:#C97A1F'>"
            f"<b>{n}</b> of {_MIN_SEL}-{_MAX_SEL} selected — pick at least {_MIN_SEL - n} more"
            f"</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span style='color:#C97A1F'>"
            f"<b>{n}</b> of {_MIN_SEL}-{_MAX_SEL} selected — remove at least {n - _MAX_SEL}"
            f"</span>",
            unsafe_allow_html=True,
        )
with info_col:
    with st.container(border=True):
        st.caption(
            "Tip: click on any stock below to see its price history and the "
            "reasons behind its forecast."
        )


# ---------- Per-stock detail panel toggle + render ----------

st.markdown("&nbsp;")
detail_options = ["(none)"] + universe_tickers
default_idx = (
    detail_options.index(st.session_state["active_detail_ticker"])
    if st.session_state["active_detail_ticker"] in detail_options
    else 0
)
detail_choice = st.selectbox(
    "View detail for…",
    options=detail_options,
    index=default_idx,
    format_func=lambda t: (
        "(select a stock)"
        if t == "(none)"
        else f"{t} — {universe.loc[t, 'name']}"
        if t in universe.index
        else t
    ),
    key="detail_picker",
)
new_detail = None if detail_choice == "(none)" else detail_choice
if new_detail != st.session_state["active_detail_ticker"]:
    st.session_state["active_detail_ticker"] = new_detail
    if new_detail:
        log_event("stock_clicked_for_shap", ticker=new_detail)


def _render_detail_panel(ticker: str) -> None:
    company = universe.loc[ticker, "name"] if ticker in universe.index else ticker
    sector = universe.loc[ticker, "sector"] if ticker in universe.index else "—"

    with st.container(border=True):
        st.markdown(f"### {company} ({ticker})")
        st.caption(f"Sector: {sector}")

        # Full-history price chart with Plotly rangeselector buttons.
        # Default view = last 5 years; user can zoom to 1y or All via the
        # buttons above the chart.
        series = prices_close[ticker].dropna()
        if len(series) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(color="#0F2540", width=2),
            ))
            latest = series.index.max()
            default_start = latest - pd.DateOffset(years=5)
            fig.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis=dict(
                    showgrid=False,
                    color="#5A5A5A",
                    range=[default_start, latest],
                    rangeselector=dict(
                        buttons=[
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all", label="All"),
                        ],
                        bgcolor="#F7F6F2",
                        activecolor="#0F2540",
                        font=dict(color="#1A1A1A"),
                    ),
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#E4E2DC", color="#5A5A5A",
                    title="Price (GBp)",
                    autorange=True,
                ),
                showlegend=False,
            )
            fig.update_yaxes(fixedrange=False, autorange=True)
            st.plotly_chart(fig, use_container_width=True)

        # Prediction metrics
        mu = float(predictions.loc[ticker, "predicted_return"])
        sigma = float(predictions.loc[ticker, "predicted_vol"])
        m_a, m_b = st.columns(2)
        with m_a:
            st.metric("Predicted return (next month)", f"{mu*100:+.2f}%")
        with m_b:
            st.metric("Predicted annualised volatility", f"{sigma*100:.2f}%")

        # SHAP plain-language reasons
        st.markdown("**Why this prediction?**")
        try:
            shap_row = shap_return_df.xs(ticker, level="ticker").iloc[0]
        except KeyError:
            st.caption("No SHAP data available for this ticker.")
            return
        reasons = shap_reasons(shap_row, top_n_positive=3, top_n_negative=1)

        if reasons["in_favour"]:
            st.markdown("*In favour*")
            for r in reasons["in_favour"]:
                st.markdown(
                    f"- **{r['label']}** ({r['value_desc']}) "
                    f"— pushing the predicted return **up** by "
                    f"**{r['contribution_pp']:.2f}pp**"
                )
        if reasons["against"]:
            st.markdown("*Against*")
            for r in reasons["against"]:
                st.markdown(
                    f"- **{r['label']}** ({r['value_desc']}) "
                    f"— pulling the predicted return **down** by "
                    f"**{r['contribution_pp']:.2f}pp**"
                )


if st.session_state["active_detail_ticker"]:
    _render_detail_panel(st.session_state["active_detail_ticker"])


# ---------- Back / Continue ----------

st.markdown("&nbsp;")
back_col, cont_col = st.columns(2, gap="small")
with back_col:
    if st.button("Back", key="as_back", use_container_width=True):
        st.switch_page("pages/2_risk_profile.py")
with cont_col:
    valid = _MIN_SEL <= n <= _MAX_SEL
    if st.button(
        "Continue",
        type="primary",
        disabled=not valid,
        use_container_width=True,
        key="as_continue",
    ):
        log_event(
            "page_navigation", from_page="asset_selection", to_page="diversification"
        )
        st.switch_page("pages/4_diversification.py")
if not valid:
    if n < _MIN_SEL:
        st.caption(f"Pick at least {_MIN_SEL - n} more stock(s) to continue.")
    elif n > _MAX_SEL:
        st.caption(f"Remove at least {n - _MAX_SEL} stock(s) to continue.")


render_page_footer()
