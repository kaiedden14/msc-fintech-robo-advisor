"""Diversification Check — Phase 4.

Pairwise correlation heatmap over the user's selected tickers, plus a
side card listing pairs with |ρ| > 0.6. Informational and non-blocking
— Continue is enabled regardless of flag count.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.data import load_prices_clean
from lib.logger import log_event
from lib.sidebar import render_page_footer


_THRESHOLD = 0.6
_LOOKBACK = 252  # trading days — matches the optimiser's Ledoit-Wolf window


# ---------- Guards ----------

if not st.session_state["risk_profile"]:
    st.title("Diversification Check")
    st.warning("Please complete the Risk Profile first.")
    if st.button("Back to Risk Profile"):
        st.switch_page("pages/2_risk_profile.py")
    render_page_footer()
    st.stop()

selected = st.session_state["selected_tickers"]
if not selected or len(selected) < 5:
    st.title("Diversification Check")
    st.warning("Please select 5–15 stocks first.")
    if st.button("Back to Asset Selection"):
        st.switch_page("pages/3_asset_selection.py")
    render_page_footer()
    st.stop()


# ---------- Page header ----------

st.title("Diversification Check")
st.caption(
    "How correlated are your chosen stocks with each other? "
    "Highly-correlated pairs (|ρ| above 0.6) tend to move together in market shocks "
    "— diversification benefits are weaker between them."
)


# ---------- Compute correlation ----------

prices_close = load_prices_clean()["Close"]
returns = prices_close[selected].tail(_LOOKBACK).pct_change(fill_method=None).dropna()

corr = returns.corr(method="pearson").reindex(index=selected, columns=selected)

# Build flagged-pairs list (upper triangle only — pairs not double-counted)
flagged: list[tuple[str, str, float]] = []
for i, t_a in enumerate(selected):
    for t_b in selected[i + 1:]:
        rho = float(corr.loc[t_a, t_b])
        if abs(rho) > _THRESHOLD:
            flagged.append((t_a, t_b, rho))
flagged.sort(key=lambda x: -abs(x[2]))

# Write to state and log the page view (only on first entry, not on every rerun)
if st.session_state["correlation_flags"] != flagged:
    st.session_state["correlation_flags"] = flagged
    log_event(
        "correlation_check_viewed",
        n_selected=len(selected),
        n_flagged=len(flagged),
    )


# ---------- Heatmap with two-zone colorscale ----------

# Mask diagonal so self-correlation doesn't read as a flag
z = corr.values.astype(float).copy()
np.fill_diagonal(z, np.nan)

# Use |ρ| for the colorscale (sequential off-white → navy below 0.6, amber above).
# Text annotations show the signed value so sign is still visible.
z_abs = np.abs(z)
text_annot = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(z))

# Two-zone Plotly colorscale: off-white → navy gradient, then a hard jump to
# amber at the 0.6 cutoff so any cell above the threshold is unmistakable.
colorscale = [
    [0.00, "#F7F6F2"],   # off-white (page surface)
    [0.30, "#7F8FA8"],   # mid navy-grey
    [0.5999, "#0F2540"], # deep navy just below threshold
    [0.60, "#C97A1F"],   # amber at threshold
    [1.00, "#C97A1F"],   # amber to top
]

fig = go.Figure(
    data=go.Heatmap(
        z=z_abs,
        x=selected,
        y=selected,
        text=text_annot,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#FFFFFF"},
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        hoverongaps=False,
        hovertemplate="%{y} × %{x}<br>ρ = %{text}<extra></extra>",
        colorbar=dict(
            title="|ρ|",
            tickvals=[0, 0.3, 0.6, 1.0],
            ticktext=["0", "0.30", "0.60", "1.0"],
        ),
    )
)
fig.update_layout(
    height=max(420, 36 * len(selected) + 80),
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    xaxis=dict(side="bottom", color="#5A5A5A"),
    yaxis=dict(autorange="reversed", color="#5A5A5A"),
)


# ---------- Two-column layout: heatmap | flagged pairs ----------

heat_col, list_col = st.columns([3, 2], gap="medium")

with heat_col:
    st.plotly_chart(fig, use_container_width=True)

with list_col:
    with st.container(border=True):
        if flagged:
            st.markdown(f"**{len(flagged)} pair(s) above 0.6**")
            st.caption(
                "Pairs of stocks whose returns are highly correlated. Holding "
                "both gives smaller diversification benefit than holding either "
                "alone — they tend to move together when the market shifts."
            )
            for t_a, t_b, rho in flagged:
                st.markdown(
                    f"- **{t_a} ↔ {t_b}**: ρ = "
                    f"<span style='color:#C97A1F'>{rho:+.2f}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("**No flagged pairs**")
            st.caption(
                "None of your selected stocks have correlation above 0.6. "
                "Your selection is reasonably diversified by this measure."
            )


# ---------- Notes for the user ----------

st.markdown("&nbsp;")
with st.container(border=True):
    st.caption(
        "**What this means.** Two stocks with a correlation close to +1 "
        "rise and fall almost in lockstep. A diversified portfolio benefits "
        "from holding assets that don't all move the same way at the same time. "
        "Flagged pairs aren't a problem on their own — they just reduce the "
        "marginal diversification benefit of holding both. "
        "The optimiser already accounts for this in the recommended weights."
    )


# ---------- Back / Continue (non-blocking) ----------

st.markdown("&nbsp;")
back_col, cont_col = st.columns(2, gap="small")
with back_col:
    if st.button("Back to Asset Selection", key="dc_back", use_container_width=True):
        log_event(
            "page_navigation", from_page="diversification", to_page="asset_selection"
        )
        st.switch_page("pages/3_asset_selection.py")
with cont_col:
    if st.button(
        "Continue",
        type="primary",
        use_container_width=True,
        key="dc_continue",
    ):
        log_event(
            "page_navigation", from_page="diversification", to_page="optimised_portfolio"
        )
        st.switch_page("pages/5_optimised_portfolio.py")


render_page_footer()
