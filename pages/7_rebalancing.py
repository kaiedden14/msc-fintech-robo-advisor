"""Rebalancing — Phase 7.

Session-closure page. Recaps the participant's final allocation, explains
the quarterly rebalance cadence, and offers Start Over (clear state) or
Finish (log session_end + show a thank-you card). No optimiser computation
— just reads from state.
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.data import load_universe_metadata
from lib.logger import log_event
from lib.sidebar import render_page_footer
from lib.state import reset_for_restart


# ---------- Finished session — show thank-you and stop ----------

if st.session_state.get("session_finished"):
    st.title("Session complete")
    with st.container(border=True):
        st.markdown(
            f"### Thank you for testing the Hybrid Robo-Advisor.\n\n"
            f"Please return to the researcher to complete the "
            f"post-session questionnaire.\n\n"
            f"**Participant ID:** {st.session_state.get('participant_id') or '—'}  \n"
            f"**Session ID:** `{st.session_state['session_id'][:12]}…`  \n"
            f"**Started:** {st.session_state['session_started_at']}"
        )
    render_page_footer()
    st.stop()


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
    "Markets move daily; the model's view updates as new data arrives. "
    "Re-running the recommendation periodically keeps your allocation "
    "aligned with the latest forecasts."
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


# ---------- Recap card ----------

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
        st.markdown("**Session recap**")

        total_pp_deviation = 0.0
        if decision == "modify" and umw is not None:
            total_pp_deviation = sum(
                abs(umw[t] - ai_weights[t]) for t in umw
            ) * 100.0

        st.markdown(
            f"- **Risk band:** {risk_band}\n"
            f"- **Investment amount:** £{amount:,.0f}\n"
            f"- **Stocks selected:** {len(st.session_state['selected_tickers'])}\n"
            f"- **Stocks held in final allocation:** {len(held)}\n"
            f"- **Decision:** {weights_source}"
            + (
                f" (total weight adjustment: {total_pp_deviation:.1f}pp)"
                if decision == "modify"
                else ""
            )
        )


# ---------- Rebalancing guidance ----------

st.markdown("&nbsp;")
with st.container(border=True):
    st.markdown("**How and when to rebalance**")
    st.markdown(
        "- **The recommendation refreshes daily** when you click **Refresh data** "
        "in the sidebar. New prices flow into the features; the optimiser re-runs "
        "with up-to-date forecasts.\n"
        "- **Suggested cadence: once a quarter.** The features the model uses move "
        "slowly (12-month momentum, 252-day beta, 21-day volatility), so day-to-day "
        "shifts in the recommendation are small. Quarterly rebalancing captures "
        "meaningful drift without forcing you to act on noise.\n"
        "- **What changes between refreshes**: predicted returns and volatilities "
        "shift as new price data arrives; the SHAP attributions update accordingly; "
        "the optimiser re-runs with the latest forecasts and may suggest different "
        "weights.\n"
        "- **What you decide**: whether to accept the new recommendation, modify "
        "it further (within ±5 percentage points per stock), or stay with what "
        "you have."
    )


# ---------- Methodology footer ----------

st.markdown("&nbsp;")
st.caption(
    "**Reminder.** This is an academic prototype for a Master's research project. "
    "Allocations shown are simulated and do not constitute financial advice. "
    "The model is trained on a survivors-only FTSE 100 sample (Brown, Goetzmann "
    "& Ross, 1995), which causes return predictions to be biased upward and "
    "underestimate downside risk. Real markets have fatter tails than the "
    "Gaussian projection cone assumes (Cont, 2001)."
)


# ---------- Action buttons ----------

st.markdown("&nbsp;")


left_col, right_col = st.columns(2, gap="medium")
with left_col:
    if st.button(
        "Start Over",
        key="rb_start_over",
        use_container_width=True,
    ):
        log_event("session_restart_from_rebalancing")
        reset_for_restart()
        st.switch_page("pages/1_landing.py")

with right_col:
    if st.button(
        "Finish",
        type="primary",
        key="rb_finish",
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
        )
        st.session_state["session_finished"] = True
        st.rerun()


render_page_footer()
