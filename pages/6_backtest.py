"""Forward Projection — Phase 6.

Monte Carlo cone over a 12-month horizon for the participant's accepted
portfolio, with the FTSE 100 median as a benchmark line. Reads the final
weights (AI or user-modified) from state, uses the optimiser's cov matrix
to compute portfolio variance, and renders an overlay chart + summary card.

Methodology note rendered explicitly: Gaussian assumption understates
fat-tail risk (Cont, 2001). Disclosed in-product so the trust intervention
remains honest.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.data import load_prices_clean
from lib.logger import log_event
from lib.sidebar import render_page_footer
from src.models.project import benchmark_params_from_history, project_forward


# ---------- Guards ----------

if not st.session_state.get("decision"):
    st.title("Forward Projection")
    st.warning("Please accept or modify your portfolio on the previous page first.")
    if st.button("Back to Optimised Portfolio"):
        st.switch_page("pages/5_optimised_portfolio.py")
    render_page_footer()
    st.stop()

if st.session_state.get("optimised_weights") is None or st.session_state.get("cov_matrix") is None:
    st.title("Forward Projection")
    st.warning("Portfolio data is missing — please rebuild your recommendation.")
    if st.button("Back to Optimised Portfolio"):
        st.switch_page("pages/5_optimised_portfolio.py")
    render_page_footer()
    st.stop()


# ---------- Determine the FINAL weights to project ----------

decision = st.session_state["decision"]
ai_weights = st.session_state["optimised_weights"]
umw = st.session_state["user_modified_weights"]

if decision == "modify" and umw is not None:
    final_weights = dict(umw)
    weights_source = "modified"
else:
    final_weights = dict(ai_weights)
    weights_source = "ai"

predictions = st.session_state["predictions"]
cov_matrix = st.session_state["cov_matrix"]
cov_tickers = st.session_state["cov_matrix_tickers"]
amount = st.session_state["investment_amount"] or 10_000.0


# ---------- Compute portfolio E[r] and E[vol] for the final weights ----------

# Vector form aligned to cov_matrix_tickers
w_vec = np.array([final_weights[t] for t in cov_tickers])
mu_vec = predictions.loc[cov_tickers, "predicted_return"].values
final_er = float(w_vec @ mu_vec)
final_var = float(w_vec @ cov_matrix @ w_vec)
final_vol = float(np.sqrt(final_var))

# Shrinkage: optimise_portfolio applies shrinkage internally to its mu before
# the SLSQP objective. For the projection's expected return we use the raw
# RF predictions to give participants an honest forward view of what the
# model thinks. The risk side (cov_matrix) is already RF-vol on diagonal +
# Ledoit-Wolf correlation off-diagonal — directly usable.


# ---------- FTSE 100 benchmark parameters ----------

prices_close = load_prices_clean()["Close"]
ftse_params = benchmark_params_from_history(prices_close["^FTSE"], lookback_years=5)


# ---------- Run the projections ----------

portfolio_proj = project_forward(
    monthly_return=final_er,
    annual_vol=final_vol,
    horizon_months=12,
    n_paths=1000,
    seed=42,
)
ftse_proj = project_forward(
    monthly_return=ftse_params["monthly_return"],
    annual_vol=ftse_params["annual_vol"],
    horizon_months=12,
    n_paths=1000,
    seed=42,
)

# Scale all cumulative multipliers by the investment amount
months = np.arange(13)  # 0..12
p_bands = portfolio_proj["percentile_bands"]
f_bands = ftse_proj["percentile_bands"]

p5  = p_bands["p5"].values  * amount
p25 = p_bands["p25"].values * amount
p50 = p_bands["p50"].values * amount
p75 = p_bands["p75"].values * amount
p95 = p_bands["p95"].values * amount
f50 = f_bands["p50"].values * amount


# ---------- Persist + log ----------

st.session_state["projection_results"] = {
    "portfolio": portfolio_proj["terminal"],
    "ftse": ftse_proj["terminal"],
    "weights_source": weights_source,
}

log_event(
    "projection_viewed",
    weights_source=weights_source,
    investment_amount=float(amount),
    portfolio_median_terminal=float(p50[-1]),
    portfolio_p5_terminal=float(p5[-1]),
    portfolio_p95_terminal=float(p95[-1]),
    ftse_median_terminal=float(f50[-1]),
    delta_vs_ftse=float(p50[-1] - f50[-1]),
)


# ---------- Header ----------

st.title("Forward Projection")
st.caption(
    "Where your portfolio could be in 12 months. Simulated 1,000 times. "
    "FTSE 100 included as a benchmark for comparison."
)


# ---------- Chart ----------

fig = go.Figure()

# Subtle horizontal reference at the starting capital — anchors the eye
# so the user can read "above the line = gain, below = loss".
fig.add_hline(
    y=amount,
    line=dict(color="#5A5A5A", width=1, dash="dot"),
    opacity=0.5,
)

# Portfolio downside (5th percentile) — amber, the "what could go wrong" line
fig.add_trace(go.Scatter(
    x=months, y=p5, mode="lines",
    line=dict(color="#C97A1F", width=2, dash="dot"),
    name="Portfolio downside (5th %ile)",
    hovertemplate="Month %{x}<br>Downside: £%{y:,.0f}<extra></extra>",
))
# Portfolio upside (95th percentile) — teal, the "what could go right" line
fig.add_trace(go.Scatter(
    x=months, y=p95, mode="lines",
    line=dict(color="#0E8E8E", width=2, dash="dot"),
    name="Portfolio upside (95th %ile)",
    hovertemplate="Month %{x}<br>Upside: £%{y:,.0f}<extra></extra>",
))
# Portfolio median — navy, prominent, with start + end markers for visual anchor
fig.add_trace(go.Scatter(
    x=months, y=p50, mode="lines+markers",
    line=dict(color="#0F2540", width=3.2),
    marker=dict(
        size=[8] + [0] * (len(months) - 2) + [10],  # start dot + end dot only
        color="#0F2540",
        line=dict(color="#FFFFFF", width=1),
    ),
    name="Portfolio median",
    hovertemplate="Month %{x}<br>Median: £%{y:,.0f}<extra></extra>",
))
# FTSE 100 median — grey, comparison benchmark
fig.add_trace(go.Scatter(
    x=months, y=f50, mode="lines",
    line=dict(color="#5A5A5A", width=2, dash="dash"),
    name="FTSE 100 median",
    hovertemplate="Month %{x}<br>FTSE: £%{y:,.0f}<extra></extra>",
))

# End-of-line value annotations at month 12 — the interpretive moments.
# Each line's terminal value labelled in its line's colour. The starting
# capital reference label sits alongside (in grey, slightly smaller) where
# the lines have spread out by month 12.
for value, colour in [
    (p95[-1],  "#0E8E8E"),
    (p50[-1],  "#0F2540"),
    (f50[-1],  "#5A5A5A"),
    (p5[-1],   "#C97A1F"),
]:
    fig.add_annotation(
        x=12, y=value,
        text=f"<b>£{value:,.0f}</b>",
        showarrow=False, xanchor="left", yanchor="middle",
        xshift=8,
        font=dict(size=11, color=colour, family="Inter, sans-serif"),
    )

# Starting capital label at the right edge, on the dotted reference line.
fig.add_annotation(
    x=12, y=amount,
    text=f"Start · £{amount:,.0f}",
    showarrow=False, xanchor="left", yanchor="middle",
    xshift=8,
    font=dict(size=10, color="#5A5A5A"),
)

# Extend x-axis a touch so the end-of-line labels don't clip
fig.update_layout(
    height=480,
    margin=dict(l=20, r=120, t=30, b=40),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    xaxis=dict(
        title="Months ahead",
        tickmode="linear", tick0=0, dtick=1,
        range=[-0.2, 12.8],
        color="#5A5A5A",
        showgrid=False,
    ),
    yaxis=dict(
        title="Portfolio value (£)",
        color="#5A5A5A",
        gridcolor="#E4E2DC",
        tickprefix="£",
        separatethousands=True,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(255,255,255,0)",
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)


# ---------- Summary card ----------

delta_vs_ftse = p50[-1] - f50[-1]
median_return_pct = (p50[-1] / amount - 1) * 100
downside_return_pct = (p5[-1] / amount - 1) * 100
upside_return_pct = (p95[-1] / amount - 1) * 100

with st.container(border=True):
    st.markdown("**12-month projection summary**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Median outcome",
            f"£{p50[-1]:,.0f}",
            delta=f"{median_return_pct:+.1f}%",
            delta_color="off",
        )
    with c2:
        st.metric(
            "Downside (5th %ile)",
            f"£{p5[-1]:,.0f}",
            delta=f"{downside_return_pct:+.1f}%",
            delta_color="off",
        )
    with c3:
        st.metric(
            "Upside (95th %ile)",
            f"£{p95[-1]:,.0f}",
            delta=f"{upside_return_pct:+.1f}%",
            delta_color="off",
        )
    with c4:
        st.metric(
            "FTSE 100 median",
            f"£{f50[-1]:,.0f}",
        )

    direction = "above" if delta_vs_ftse >= 0 else "below"
    st.markdown(
        f"On the median outcome, your portfolio finishes "
        f"**£{abs(delta_vs_ftse):,.0f} {direction}** the FTSE 100 over the next 12 months."
    )


# ---------- Methodology disclosure ----------

st.markdown("&nbsp;")
with st.container(border=True):
    st.caption(
        "**Methodology note.** Projections assume monthly returns are "
        "normally distributed (Gaussian Monte Carlo, 1,000 paths, fixed seed). "
        "Real markets have fatter tails than this assumes — both severe "
        "drawdowns and rallies are more likely than the cone suggests "
        "(Cont, 2001). Treat these as broad scenarios, not precise forecasts. "
        f"FTSE 100 parameters estimated from the trailing 5 years of monthly "
        f"returns ({ftse_params['window_start'].date()} to "
        f"{ftse_params['window_end'].date()})."
    )


# ---------- Back / Continue ----------

st.markdown("&nbsp;")
back_col, cont_col = st.columns(2, gap="medium")
with back_col:
    if st.button(
        "Back to Optimised Portfolio",
        key="proj_back",
        use_container_width=True,
    ):
        st.switch_page("pages/5_optimised_portfolio.py")
with cont_col:
    if st.button(
        "Continue",
        type="primary",
        key="proj_continue",
        use_container_width=True,
    ):
        log_event(
            "page_navigation",
            from_page="forward_projection",
            to_page="rebalancing",
        )
        st.switch_page("pages/7_rebalancing.py")


render_page_footer()
