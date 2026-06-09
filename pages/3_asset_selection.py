"""Asset Selection, Phase 3.

Click-to-view interactive table. Click any row to load that stock's
chart, predictions, and SHAP explanation into the detail panel below.
Selection (add to / remove from your portfolio) happens via a button
in the detail panel, keeps the table itself clean (no checkbox column,
no separate dropdown). Enforces 5–15 selection constraint via the
Continue-button gating.
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


# ---------- Guard ----------

if not st.session_state["risk_profile"]:
    st.title("Asset Selection")
    st.warning("Please complete the Risk Profile first.")
    if st.button("Back to Risk Profile"):
        st.switch_page("pages/2_risk_profile.py")
    render_page_footer()
    st.stop()


# ---------- Header ----------

st.title("Asset Selection")
st.caption(
    f"Pick {_MIN_SEL}-{_MAX_SEL} stocks from the FTSE 100. "
    "Click a row to see a stock's chart, predictions, and the reasons "
    "behind them, then add it to your selection from the detail panel."
)


# ---------- Load data ----------

universe = load_universe_metadata()
predictions = load_predictions()
prices_close = load_prices_clean()["Close"]
shap_return_df = load_shap_return()

universe_tickers: list[str] = sorted(predictions.index.tolist())

latest_date = prices_close.index.max()
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
        "✓":         ["✓" if t in selected_set else "" for t in universe_tickers],
        "Ticker":    universe_tickers,
        "Company":   [
            universe.loc[t, "name"] if t in universe.index else t
            for t in universe_tickers
        ],
        "Sector":    [
            universe.loc[t, "sector"] if t in universe.index else "–"
            for t in universe_tickers
        ],
        "Last close":  last_close.values,
        "1d %":        daily_change_pct.values,
        "Return %":    predictions.loc[universe_tickers, "predicted_return"].values * 100,
        "Vol %":       predictions.loc[universe_tickers, "predicted_vol"].values * 100,
    }
)


# ---------- Filters + counter row ----------

filt_col_a, filt_col_b, counter_col = st.columns([2, 3, 2], gap="medium")
with filt_col_a:
    search = st.text_input(
        "Search ticker or company",
        placeholder="HSBA or HSBC",
        key="asset_search",
    )
with filt_col_b:
    sectors_available = sorted(
        s for s in table["Sector"].dropna().unique() if s and s != "–"
    )
    sector_filter = st.multiselect(
        "Filter by sector",
        options=sectors_available,
        key="asset_sector_filter",
    )
with counter_col:
    st.markdown("&nbsp;")  # vertical alignment with the inputs
    n = len(st.session_state["selected_tickers"])
    if n == 0:
        st.markdown(f"**0** of {_MIN_SEL}-{_MAX_SEL} selected")
    elif _MIN_SEL <= n <= _MAX_SEL:
        st.markdown(f"**{n}** of {_MIN_SEL}-{_MAX_SEL} selected")
    elif n < _MIN_SEL:
        st.markdown(
            f"<span style='color:#C97A1F'>"
            f"<b>{n}</b> of {_MIN_SEL}-{_MAX_SEL}, pick {_MIN_SEL - n} more"
            f"</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span style='color:#C97A1F'>"
            f"<b>{n}</b> of {_MIN_SEL}-{_MAX_SEL}, remove {n - _MAX_SEL}"
            f"</span>",
            unsafe_allow_html=True,
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


# ---------- Row-click table ----------

event = st.dataframe(
    filtered,
    column_config={
        "✓":          st.column_config.TextColumn(width="small"),
        "Ticker":     st.column_config.TextColumn(width="small"),
        "Company":    st.column_config.TextColumn(width="medium"),
        "Sector":     st.column_config.TextColumn(width="medium"),
        "Last close": st.column_config.NumberColumn(format="%.2f", width="small"),
        "1d %":       st.column_config.NumberColumn(format="%+.2f", width="small"),
        "Return %":   st.column_config.NumberColumn(format="%+.2f", width="small"),
        "Vol %":      st.column_config.NumberColumn(format="%.2f", width="small"),
    },
    hide_index=True,
    use_container_width=True,
    height=420,
    on_select="rerun",
    selection_mode="single-row",
    key="asset_table",
)

# Row click → set active_detail_ticker
if event.selection.rows:
    picked = filtered.iloc[event.selection.rows[0]]["Ticker"]
    if picked != st.session_state["active_detail_ticker"]:
        st.session_state["active_detail_ticker"] = picked
        log_event(
            "stock_clicked_for_shap",
            ticker=picked,
            source="asset_selection_table",
        )
        st.rerun()


# ---------- Detail panel (row-clicked stock) ----------

def _toggle_selection(ticker: str, currently_selected: bool) -> None:
    """Button on_click handler, add to or remove from selected_tickers."""
    selected = set(st.session_state["selected_tickers"])
    if currently_selected:
        new_full = sorted(selected - {ticker})
        log_event(
            "ticker_deselected",
            ticker=ticker,
            current_selection_count=len(new_full),
        )
    else:
        new_full = sorted(selected | {ticker})
        log_event(
            "ticker_selected",
            ticker=ticker,
            current_selection_count=len(new_full),
        )
    st.session_state["selected_tickers"] = new_full
    clear_downstream_of("selected_tickers")


def _render_detail_panel(ticker: str) -> None:
    company = universe.loc[ticker, "name"] if ticker in universe.index else ticker
    sector = universe.loc[ticker, "sector"] if ticker in universe.index else "–"
    is_selected = ticker in set(st.session_state["selected_tickers"])
    n_current = len(st.session_state["selected_tickers"])

    with st.container(border=True):
        # Header row: stock name on the left, Add/Remove button on the right
        hdr_col, btn_col = st.columns([3, 1], gap="medium")
        with hdr_col:
            st.markdown(f"### {company}")
            st.caption(f"{ticker} · {sector}")
        with btn_col:
            if is_selected:
                st.button(
                    "Remove from selection",
                    key="detail_remove_btn",
                    use_container_width=True,
                    on_click=_toggle_selection,
                    args=(ticker, True),
                )
            else:
                at_max = n_current >= _MAX_SEL
                st.button(
                    "Add to selection",
                    key="detail_add_btn",
                    type="primary",
                    use_container_width=True,
                    disabled=at_max,
                    on_click=_toggle_selection,
                    args=(ticker, False),
                )
                if at_max:
                    st.caption(
                        f"At max ({_MAX_SEL}). Remove one first."
                    )

        # 4 metric tiles: Last close, 1d change, Pred. return, Pred. vol
        mu = float(predictions.loc[ticker, "predicted_return"])
        sigma = float(predictions.loc[ticker, "predicted_vol"])
        lc = float(last_close[ticker])
        dc = float(daily_change_pct[ticker])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Last close", f"{lc:,.2f}p")
        with m2:
            st.metric("1d change", f"{dc:+.2f}%", delta_color="off")
        with m3:
            st.metric("Predicted return (1q)", f"{mu*100:+.2f}%")
        with m4:
            st.metric("Predicted vol (annual)", f"{sigma*100:.2f}%")

        # Price chart, full history with rangeselector
        series = prices_close[ticker].dropna()
        if len(series) > 0:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    line=dict(color="#0F2540", width=2),
                )
            )
            latest = series.index.max()
            default_start = latest - pd.DateOffset(years=5)
            fig.update_layout(
                height=300,
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
                        bgcolor="#F2EAD5",
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

        # SHAP plain-language reasons (in-favour / against bullet list)
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
                    f", pushing the predicted return **up** by "
                    f"**{r['contribution_pp']:.2f}pp**"
                )
        if reasons["against"]:
            st.markdown("*Against*")
            for r in reasons["against"]:
                st.markdown(
                    f"- **{r['label']}** ({r['value_desc']}) "
                    f", pulling the predicted return **down** by "
                    f"**{r['contribution_pp']:.2f}pp**"
                )


active = st.session_state["active_detail_ticker"]
if active and active in universe_tickers:
    st.markdown("&nbsp;")
    _render_detail_panel(active)
else:
    st.markdown("&nbsp;")
    with st.container(border=True):
        st.caption(
            "Click any row in the table above to see that stock's chart, "
            "predictions, and SHAP explanation here."
        )


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
            "page_navigation",
            from_page="asset_selection",
            to_page="diversification",
        )
        st.switch_page("pages/4_diversification.py")
if not valid:
    if n < _MIN_SEL:
        st.caption(f"Pick at least {_MIN_SEL - n} more stock(s) to continue.")
    elif n > _MAX_SEL:
        st.caption(f"Remove at least {n - _MAX_SEL} stock(s) to continue.")


render_page_footer()
