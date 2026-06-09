"""Rebalancing.

Session-closure + practical product page. Shows the final allocation,
realised performance since the decision date, and a one-click
re-check that runs the optimiser against the latest data so the user
can see whether the AI's view has shifted.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.bands import BANDS, effective_cap
from lib.data import (
    load_predictions,
    load_prices_clean,
    load_universe_metadata,
)
from lib.logger import log_event
from lib.performance import compute_realised_performance
from lib.persistence import delete_portfolio, load_portfolio, save_portfolio
from lib.sidebar import render_page_footer
from lib.state import reset_for_restart
from src.models.optimise import optimise_portfolio


# ---------- Guards ----------

if not st.session_state.get("decision"):
    st.title("Rebalancing")
    st.warning("Please complete the recommendation flow first.")
    if st.button("Back to Optimised Portfolio"):
        st.switch_page("pages/5_optimised_portfolio.py")
    render_page_footer()
    st.stop()


# ---------- Header ----------

st.title("Rebalancing")
st.caption(
    "Review how your allocation has performed and check whether the AI's "
    "view has shifted since you set it up."
)


# ---------- Determine final accepted weights ----------

decision = st.session_state["decision"]
ai_weights = st.session_state["optimised_weights"]
umw = st.session_state["user_modified_weights"]

if decision == "modify" and umw is not None:
    final_weights = dict(umw)
    weights_source = "your modified allocation"
else:
    final_weights = dict(ai_weights)
    weights_source = "the AI recommendation"

amount = st.session_state["investment_amount"] or 10_000.0
risk_band = st.session_state["risk_profile"]
selected = st.session_state["selected_tickers"]


# ---------- Section 1: Recap card ----------

universe = load_universe_metadata()

weight_series = pd.Series(final_weights)
held = weight_series[weight_series > 0.001].sort_values(ascending=False)

st.markdown("&nbsp;")
recap_col, summary_col = st.columns([2, 3], gap="medium")

with recap_col:
    with st.container(border=True):
        st.markdown("**Your final allocation**")

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=held.index.tolist(),
                    values=held.values,
                    hole=0.55,
                    sort=False,
                    marker=dict(line=dict(color="#FFFFFF", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=10),
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Weight: %{percent}<br>"
                        "Allocation: £%{customdata:,.0f}<extra></extra>"
                    ),
                    customdata=held.values * amount,
                )
            ]
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF",
            showlegend=False,
            annotations=[
                dict(
                    text=f"<b>£{amount:,.0f}</b>",
                    showarrow=False,
                    font=dict(size=16, color="#0F2540"),
                    x=0.5, y=0.55,
                ),
                dict(
                    text=f"{len(held)} stocks",
                    showarrow=False,
                    font=dict(size=10, color="#5A5A5A"),
                    x=0.5, y=0.42,
                ),
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

with summary_col:
    with st.container(border=True):
        st.markdown("**Allocation summary**")

        total_pp_deviation = 0.0
        if decision == "modify" and umw is not None:
            total_pp_deviation = sum(
                abs(umw[t] - ai_weights[t]) for t in umw
            ) * 100.0

        st.markdown(
            f"- **Risk band:** {risk_band}\n"
            f"- **Investment amount:** £{amount:,.0f}\n"
            f"- **Stocks selected:** {len(selected)}\n"
            f"- **Stocks held:** {len(held)}\n"
            f"- **Decision:** {weights_source}"
            + (
                f" (total adjustment: {total_pp_deviation:.1f}pp)"
                if decision == "modify"
                else ""
            )
        )


# ---------- Section 2: Per-stock performance since investment ----------

st.markdown("&nbsp;")
st.markdown("### How each stock has performed")

prices_close = load_prices_clean()["Close"]
pid = st.session_state.get("participant_id")
saved = load_portfolio(pid) if pid else None

# Anchor the performance window at the saved investment_date if present;
# otherwise fall back to today (which will show the "too new" message).
if saved is not None:
    investment_date = pd.Timestamp(saved["investment_date"])
else:
    investment_date = pd.Timestamp(datetime.now().date())

st.caption(
    f"Performance since you invested on **{investment_date.strftime('%d %b %Y')}**. "
    "The table updates each trading day as new prices arrive."
)

perf = compute_realised_performance(
    weights=final_weights,
    prices_close=prices_close,
    start_date=investment_date,
    benchmark_col="^FTSE",
)

if perf.get("insufficient") or perf["n_days"] < 5:
    with st.container(border=True):
        st.markdown(
            "🌱 **Your portfolio is brand new.**  \n"
            "Performance figures will start appearing here once a few "
            "trading days have passed. Check back tomorrow to see the "
            "first day's movement."
        )
else:
    per_stock = perf["per_stock"]

    # Build the display table: only stocks actually held (weight > 0.001)
    rows = []
    for ticker, row in per_stock.iterrows():
        w = float(row["weight"])
        if w < 0.001:
            continue
        ret = float(row["stock_return"])
        invested = w * amount
        current_value = invested * (1 + ret)
        company = (
            universe.loc[ticker, "name"]
            if ticker in universe.index
            else ticker
        )
        rows.append({
            "Stock":         ticker,
            "Company":       company,
            "Position":      f"{w*100:.1f}%",
            "Invested":      f"£{invested:,.0f}",
            "Current value": f"£{current_value:,.0f}",
            "Return":        f"{ret*100:+.2f}%",
            "_ret_raw":      ret,  # used for sorting + styling
        })
    table_df = pd.DataFrame(rows).sort_values("_ret_raw", ascending=False)

    # Aggregate summary (single line above the table)
    total_invested = amount
    total_current = sum(
        float(per_stock.loc[t, "weight"]) * amount
        * (1 + float(per_stock.loc[t, "stock_return"]))
        for t in per_stock.index
        if float(per_stock.loc[t, "weight"]) >= 0.001
    )
    overall_pnl = total_current - total_invested
    overall_pct = overall_pnl / total_invested * 100

    with st.container(border=True):
        sum_col_a, sum_col_b, sum_col_c = st.columns(3)
        with sum_col_a:
            st.metric("Total invested", f"£{total_invested:,.0f}")
        with sum_col_b:
            st.metric(
                "Current value",
                f"£{total_current:,.0f}",
                f"{'+' if overall_pnl >= 0 else ''}£{overall_pnl:,.0f}",
            )
        with sum_col_c:
            st.metric("Overall return", f"{overall_pct:+.2f}%")

        st.caption(
            f"Based on prices from **{perf['start_date'].strftime('%d %b %Y')}** "
            f"to **{perf['end_date'].strftime('%d %b %Y')}** "
            f"({perf['n_days']} trading days)."
        )

        # Colour-code the Return column: teal positive, amber negative
        def _color_return(val: str) -> str:
            if isinstance(val, str):
                if val.startswith("+"):
                    return "color: #0E8E8E; font-weight: 600;"
                if val.startswith("-"):
                    return "color: #C97A1F; font-weight: 600;"
            return ""

        styled = (
            table_df.drop(columns=["_ret_raw"])
            .style.map(_color_return, subset=["Return"])
        )

        st.dataframe(
            styled,
            hide_index=True,
            use_container_width=True,
        )


# ---------- Section 3: Check for an updated recommendation ----------

st.markdown("&nbsp;")
st.markdown("### Check for an updated recommendation")

with st.container(border=True):
    st.caption(
        "Re-runs the AI on your current selection with whatever data is "
        "loaded. To pull the latest prices first, click **Refresh data** "
        "in the sidebar — then re-check below."
    )

    check_col, _ = st.columns([1, 3])
    with check_col:
        if st.button(
            "Re-check this allocation",
            type="primary",
            key="rb_recheck",
            use_container_width=True,
        ):
            # Re-run the optimiser inline (does NOT overwrite session state)
            predictions_full = load_predictions()
            mu = predictions_full.loc[selected, "predicted_return"]
            sigma = predictions_full.loc[selected, "predicted_vol"]
            daily_returns = prices_close[selected].pct_change(
                fill_method=None
            ).dropna(how="all")

            config = BANDS[risk_band]
            cap = effective_cap(len(selected))

            fresh = optimise_portfolio(
                predicted_returns=mu,
                predicted_vols=sigma,
                historical_returns=daily_returns,
                risk_aversion=config.risk_aversion,
                shrinkage_alpha=config.shrinkage_alpha,
                max_weight=cap,
            )
            fresh_weights = {k: float(v) for k, v in fresh["weights"].items()}
            st.session_state["fresh_recommendation"] = {
                "weights":         fresh_weights,
                "expected_return": float(fresh["expected_return"]),
                "expected_vol":    float(fresh["expected_vol"]),
            }
            # Compare against the user's current final weights
            max_delta_pp = max(
                abs(fresh_weights.get(t, 0.0) - final_weights.get(t, 0.0))
                for t in set(fresh_weights) | set(final_weights)
            ) * 100.0
            log_event(
                "rebalance_check_run",
                fresh_weights=fresh_weights,
                current_weights=final_weights,
                max_delta_pp=float(max_delta_pp),
            )

    fresh_rec = st.session_state.get("fresh_recommendation")
    if fresh_rec is not None:
        st.markdown("&nbsp;")
        fw = fresh_rec["weights"]
        cw = final_weights

        tickers_all = sorted(set(cw) | set(fw))
        rows = []
        for t in tickers_all:
            current = cw.get(t, 0.0)
            new = fw.get(t, 0.0)
            delta = new - current
            rows.append({
                "Ticker":          t,
                "Your allocation": current,
                "Latest":          new,
                "Change":          delta,
            })
        comp_df = pd.DataFrame(rows)
        comp_df = comp_df.sort_values("Change", key=lambda s: s.abs(), ascending=False)

        max_abs_delta_pp = comp_df["Change"].abs().max() * 100.0

        if max_abs_delta_pp < 0.5:
            st.success(
                "**No material changes.** The AI's recommendation hasn't shifted "
                "meaningfully since your session — your current allocation still "
                "matches the latest view."
            )
        else:
            st.markdown(
                f"**Largest single-stock change: {max_abs_delta_pp:.1f}pp.** "
                "Review the table below."
            )

            display_df = comp_df.copy()
            display_df["Your allocation"] = (display_df["Your allocation"] * 100).map("{:.1f}%".format)
            display_df["Latest"] = (display_df["Latest"] * 100).map("{:.1f}%".format)
            display_df["Change"] = (display_df["Change"] * 100).map(
                lambda x: f"{x:+.1f}pp"
            )
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
            )

            adopt_col, discard_col, _ = st.columns([1, 1, 2])
            with adopt_col:
                if st.button(
                    "Adopt the latest",
                    type="primary",
                    key="rb_adopt",
                    use_container_width=True,
                ):
                    st.session_state["optimised_weights"] = dict(fw)
                    st.session_state["user_modified_weights"] = None
                    st.session_state["decision"] = "accept"
                    st.session_state["expected_return"] = fresh_rec["expected_return"]
                    st.session_state["expected_vol"] = fresh_rec["expected_vol"]
                    st.session_state["fresh_recommendation"] = None
                    # Persist the new allocation with today's date — adopting
                    # is a fresh investment, so the performance clock resets.
                    if pid:
                        save_portfolio(
                            participant_id=pid,
                            weights=fw,
                            selected_tickers=list(st.session_state["selected_tickers"]),
                            investment_amount=float(amount),
                            risk_band=risk_band,
                            decision="accept",
                        )
                    log_event(
                        "rebalance_adopted",
                        new_weights=fw,
                    )
                    st.rerun()
            with discard_col:
                if st.button(
                    "Keep current",
                    key="rb_discard",
                    use_container_width=True,
                ):
                    st.session_state["fresh_recommendation"] = None
                    log_event("rebalance_discarded")
                    st.rerun()


# ---------- Section 4: Rebalancing guidance ----------

st.markdown("&nbsp;")
with st.expander("About rebalancing"):
    st.markdown(
        "- **The recommendation refreshes when you click *Refresh data*** in "
        "the sidebar. New prices flow into the model; the optimiser re-runs "
        "with up-to-date forecasts when you check again.\n"
        "- **Suggested cadence: once a quarter.** The features the model uses "
        "move slowly, so day-to-day shifts in the recommendation are small. "
        "Quarterly rebalancing captures meaningful drift without acting on noise.\n"
        "- **You decide whether to adopt** any updated recommendation, modify "
        "it further, or stay with what you have."
    )


# ---------- Footer: Confirm portfolio / Start over ----------

st.markdown("&nbsp;")

confirmed = st.session_state.get("portfolio_confirmed", False)

if confirmed:
    st.success("Portfolio confirmed — your session has been saved.")
else:
    st.caption("If you are happy with the portfolio, click **Confirm portfolio**.")

start_over_col, confirm_col, _ = st.columns([1, 1, 2], gap="medium")

with start_over_col:
    if st.button(
        "Start over",
        key="rb_start_over",
        use_container_width=True,
    ):
        log_event("session_restart_from_rebalancing")
        # Wipe this participant's saved portfolio so the restart is a
        # genuine fresh slate (otherwise the next session would still
        # see the old portfolio's investment_date and weights).
        if pid:
            delete_portfolio(pid)
        reset_for_restart()
        st.switch_page("pages/1_landing.py")

with confirm_col:
    if confirmed:
        # Disabled badge — clearly shows the action has been completed.
        st.button(
            "✓ Portfolio confirmed",
            key="rb_confirm_done",
            use_container_width=True,
            disabled=True,
        )
    else:
        if st.button(
            "Confirm portfolio",
            type="primary",
            key="rb_confirm",
            use_container_width=True,
        ):
            try:
                started = datetime.fromisoformat(
                    st.session_state["session_started_at"]
                )
                duration_s = int((datetime.now() - started).total_seconds())
            except Exception:
                duration_s = None
            log_event(
                "session_end",
                total_duration_s=duration_s,
                final_decision=st.session_state.get("decision"),
                weights_source=weights_source,
                final_weights=final_weights,
            )
            st.session_state["portfolio_confirmed"] = True
            st.toast("Portfolio confirmed — session saved.", icon="✓")
            st.rerun()


render_page_footer()
