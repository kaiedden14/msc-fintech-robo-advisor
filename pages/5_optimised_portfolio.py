"""Optimised Portfolio — Phase 5a.

Triggers compute_recommendation() on entry (or after selection / risk-band
change) and renders a diagnostic readout of the optimiser output. The
allocation view (5b), per-stock detail panel (5c), and action buttons
(5d) land in subsequent sub-phases.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.bands import effective_cap
from lib.copy import (
    bucket_vol_outlook,
    decomposition_summary,
    shap_one_liner_return,
    shap_one_liner_volatility,
    shap_waterfall_frame,
)
from lib.data import load_predictions, load_universe_metadata
from lib.logger import log_event
from lib.portfolio import compute_recommendation, single_anchor_renorm
from lib.sidebar import render_page_footer
from lib.state import reset_for_restart


# ---------- Guards ----------

if not st.session_state["risk_profile"]:
    st.title("Optimised Portfolio")
    st.warning("Please complete the Risk Profile first.")
    if st.button("Back to Risk Profile"):
        st.switch_page("pages/2_risk_profile.py")
    render_page_footer()
    st.stop()

selected = st.session_state["selected_tickers"]
if not selected or not (5 <= len(selected) <= 15):
    st.title("Optimised Portfolio")
    st.warning("Please select 5-15 stocks first.")
    if st.button("Back to Asset Selection"):
        st.switch_page("pages/3_asset_selection.py")
    render_page_footer()
    st.stop()


# ---------- Header ----------

st.title("Optimised Portfolio")
st.caption(
    f"Recommendation for your {len(selected)}-stock "
    f"**{st.session_state['risk_profile']}** portfolio."
)


# ---------- Compute (idempotent) ----------

cached_before_call = st.session_state["optimised_weights"] is not None
try:
    if cached_before_call:
        # Fast path — no spinner, no work
        compute_recommendation()
    else:
        with st.spinner("Building your recommended portfolio…"):
            compute_recommendation()
except ValueError as e:
    st.error(f"Could not compute recommendation: {e}")
    if st.button("Back to Asset Selection"):
        st.switch_page("pages/3_asset_selection.py")
    render_page_footer()
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while computing recommendation: {e}")
    if st.button("Back to Asset Selection"):
        st.switch_page("pages/3_asset_selection.py")
    render_page_footer()
    st.stop()


# ---------- Phase 5a deliverable: diagnostic summary card ----------

amount = st.session_state["investment_amount"] or 0.0
expected_return = st.session_state["expected_return"]
expected_vol = st.session_state["expected_vol"]
weights = st.session_state["optimised_weights"]
n_active = sum(1 for w in weights.values() if w > 0.001)

with st.container(border=True):
    st.markdown("**Recommendation summary**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Risk band", st.session_state["risk_profile"])
    with c2:
        st.metric("Stocks held", f"{n_active} of {len(selected)}")
    with c3:
        st.metric("Expected return (1 month)", f"{expected_return * 100:+.2f}%")
    with c4:
        st.metric("Expected volatility (annualised)", f"{expected_vol * 100:.2f}%")
    st.caption(
        f"Total to allocate: £{amount:,.0f} · "
        f"Optimiser engine: mean-variance SLSQP with Ledoit-Wolf correlation "
        f"and RF-predicted volatilities."
    )


# ---------- Phase 5b / 5c placeholders ----------

st.markdown("&nbsp;")

# ---------- Phase 5b: allocation view (left column) ----------

universe = load_universe_metadata()
predictions_state = st.session_state["predictions"]  # selected-only slice from 5a
predictions_universe = load_predictions()             # full 93-ticker universe
weights = st.session_state["optimised_weights"]

# Filter to held positions and sort by weight desc
weight_series = pd.Series(weights, name="weight")
held = weight_series[weight_series > 0.001].sort_values(ascending=False)

# Build the allocation table — Return as raw %, Risk as bucket label
vol_outlook_all = bucket_vol_outlook(predictions_universe["predicted_vol"])
table = pd.DataFrame(
    {
        "Ticker":         held.index.tolist(),
        "Company":        [
            universe.loc[t, "name"] if t in universe.index else t
            for t in held.index
        ],
        "Weight %":       (held.values * 100),
        "Allocation £":   (held.values * amount),
        "Pred. return %": [
            float(predictions_state.loc[t, "predicted_return"]) * 100
            for t in held.index
        ],
        "Risk":           [vol_outlook_all.loc[t] for t in held.index],
    }
)

left, right = st.columns([3, 2], gap="medium")

with left:
    with st.container(border=True):
        st.markdown("**Allocation**")
        st.caption(
            "Recommended weights for your portfolio. "
            "Click a row in the table to see the reasoning behind that stock."
        )

        # Donut chart with centre annotation
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=held.index.tolist(),
                    values=held.values,
                    hole=0.55,
                    sort=False,
                    marker=dict(line=dict(color="#FFFFFF", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=11),
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
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF",
            showlegend=False,
            annotations=[
                dict(
                    text=f"<b>£{amount:,.0f}</b>",
                    showarrow=False,
                    font=dict(size=20, color="#0F2540"),
                    x=0.5, y=0.55,
                ),
                dict(
                    text=f"across {len(held)} stocks",
                    showarrow=False,
                    font=dict(size=11, color="#5A5A5A"),
                    x=0.5, y=0.42,
                ),
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Allocation table with native single-row selection
        event = st.dataframe(
            table,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Ticker":         st.column_config.TextColumn(width="small"),
                "Company":        st.column_config.TextColumn(width="medium"),
                "Weight %":       st.column_config.NumberColumn(format="%.1f%%", width="small"),
                "Allocation £":   st.column_config.NumberColumn(format="£%.0f", width="small"),
                "Pred. return %": st.column_config.NumberColumn(format="%+.2f%%", width="small"),
                "Risk":           st.column_config.TextColumn(width="small"),
            },
            hide_index=True,
            use_container_width=True,
            key="alloc_table",
        )

        # Row selection -> active_detail_ticker (drives Phase 5c right column)
        if event.selection.rows:
            picked = table.iloc[event.selection.rows[0]]["Ticker"]
            if picked != st.session_state["active_detail_ticker"]:
                st.session_state["active_detail_ticker"] = picked
                log_event(
                    "stock_clicked_for_shap",
                    ticker=picked,
                    source="screen5_table",
                )
                st.rerun()


# ---------- Phase 5c: per-stock detail panel (right column) ----------


def _build_waterfall(wf, y_title: str):
    """Construct a Plotly Waterfall figure from a shap_waterfall_frame()."""
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=wf["measure"].tolist(),
            x=wf["label"].tolist(),
            y=wf["value"].tolist(),
            text=[f"{v:+.2f}pp" if m == "relative" else f"{v:.2f}pp"
                  for v, m in zip(wf["value"], wf["measure"])],
            textposition="outside",
            connector={"line": {"color": "#E4E2DC"}},
            increasing={"marker": {"color": "#0E8E8E"}},
            decreasing={"marker": {"color": "#C97A1F"}},
            totals={"marker": {"color": "#0F2540"}},
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis=dict(title=y_title, color="#5A5A5A", gridcolor="#E4E2DC"),
        xaxis=dict(color="#5A5A5A", tickangle=-20),
        showlegend=False,
    )
    return fig


def _build_decomp_bars(decomp_row):
    """Construct the horizontal-bar weight-decomposition figure (four bars)."""
    contributions = [
        ("Predicted return",      float(decomp_row["return"])     * 100),
        ("Individual volatility", float(decomp_row["variance"])   * 100),
        ("Diversification",       float(decomp_row["covariance"]) * 100),
        ("Risk profile",          float(decomp_row["risk"])       * 100),
    ]
    labels = [c[0] for c in contributions]
    values = [c[1] for c in contributions]
    colors = ["#0E8E8E" if v >= 0 else "#C97A1F" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.2f}pp" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=20, t=20, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title="Weight contribution (pp)",
            color="#5A5A5A",
            gridcolor="#E4E2DC",
            zeroline=True,
            zerolinecolor="#5A5A5A",
        ),
        yaxis=dict(color="#5A5A5A", autorange="reversed"),
        showlegend=False,
    )
    return fig


with right:
    active = st.session_state["active_detail_ticker"]

    if not active:
        with st.container(border=True):
            st.markdown("**Per-stock detail panel**")
            st.caption(
                "Click a row in the allocation table to load a stock's "
                "detail view here."
            )
    else:
        company = (
            universe.loc[active, "name"] if active in universe.index else active
        )
        st.markdown(f"### {company} ({active})")

        # Pull SHAP rows and decomposition row from state (written by Phase 5a)
        shap_r_state = st.session_state["shap_values_return"]
        shap_v_state = st.session_state["shap_values_volatility"]
        decomp_state = st.session_state["decomposition"]

        try:
            shap_r_row = shap_r_state.xs(active, level="ticker").iloc[0]
            shap_v_row = shap_v_state.xs(active, level="ticker").iloc[0]
        except KeyError:
            shap_r_row = None
            shap_v_row = None

        # --- Sub-card 1: Return SHAP waterfall ---
        with st.container(border=True):
            st.markdown("**Why this return prediction?**")
            if shap_r_row is not None:
                wf = shap_waterfall_frame(shap_r_row, top_n=4)
                st.plotly_chart(
                    _build_waterfall(wf, "Return prediction (pp)"),
                    use_container_width=True,
                )
                st.markdown(shap_one_liner_return(shap_r_row, active))
                st.caption("Top 4 of 7 features shown.")
            else:
                st.caption("SHAP data not available for this stock.")

        # --- Sub-card 2: Volatility SHAP waterfall ---
        with st.container(border=True):
            st.markdown("**Why this volatility prediction?**")
            if shap_v_row is not None:
                wf = shap_waterfall_frame(shap_v_row, top_n=4)
                st.plotly_chart(
                    _build_waterfall(wf, "Volatility prediction (pp)"),
                    use_container_width=True,
                )
                st.markdown(shap_one_liner_volatility(shap_v_row, active))
                st.caption("Top 4 of 7 features shown.")
            else:
                st.caption("SHAP data not available for this stock.")

        # --- Section break: predictions ↑ allocation ↓ ---
        st.markdown(
            "<div class='ra-section-label'>Allocation reasoning</div>",
            unsafe_allow_html=True,
        )

        # --- Sub-card 3: Weight decomposition ---
        with st.container(border=True):
            st.markdown("**Why this weight?**")
            if active in held.index:
                decomp_row = decomp_state.loc[active]
                st.plotly_chart(
                    _build_decomp_bars(decomp_row),
                    use_container_width=True,
                )
                st.markdown(decomposition_summary(decomp_row, active))
                st.caption(
                    "Positive (teal) bars increased this stock's weight vs the "
                    "all-neutralised baseline; negative (amber) bars reduced it."
                )
            else:
                st.caption(
                    f"The optimiser allocated no weight to **{active}**. "
                    f"The decomposition isn't meaningful for an unweighted position."
                )


# ---------- Phase 5d: Modify expander + action buttons ----------


def _on_reset_to_ai_click() -> None:
    """Reset-to-AI button callback.

    Must run as an on_click callback (NOT inline after the button) because
    callbacks fire before the next rerun's widgets render — at inline time,
    every weight_slider_<ticker> widget has already been instantiated and
    Streamlit forbids modifying its session_state key.
    """
    ai = st.session_state["optimised_weights"]
    st.session_state["user_modified_weights"] = dict(ai)
    for t in ai:
        slider_key = f"weight_slider_{t}"
        if slider_key in st.session_state:
            st.session_state[slider_key] = ai[t] * 100.0
    log_event("reset_to_ai_weights", tickers=list(ai.keys()))


def _on_weight_slider_change(ticker: str) -> None:
    """Slider on_change callback: applies single-anchor renormalisation."""
    slider_key = f"weight_slider_{ticker}"
    new_value_pp = float(st.session_state[slider_key])
    new_value = new_value_pp / 100.0

    current = dict(st.session_state["user_modified_weights"])
    old_value = current.get(ticker, new_value)

    if abs(new_value - old_value) < 1e-9:
        return

    ai_w = st.session_state["optimised_weights"]
    cap = effective_cap(len(current))

    result = single_anchor_renorm(current, ai_w, ticker, new_value, cap)

    if result is None:
        # Can't fully redistribute — bounce slider back to its previous position
        st.session_state[slider_key] = old_value * 100.0
        log_event(
            "slider_blocked",
            ticker=ticker,
            attempted_weight=float(new_value),
            from_weight=float(old_value),
        )
        return

    # Commit + sync sibling slider widget keys so they display the new positions
    st.session_state["user_modified_weights"] = result
    for t in result:
        if t != ticker:
            st.session_state[f"weight_slider_{t}"] = result[t] * 100.0

    log_event(
        "slider_moved",
        ticker=ticker,
        from_weight=float(old_value),
        to_weight=float(new_value),
        magnitude_pp=float((new_value - old_value) * 100.0),
    )


st.markdown("&nbsp;")

with st.expander("Modify your allocation (optional)", expanded=False):
    # Initialise user_modified_weights to a copy of AI weights on first open
    if st.session_state["user_modified_weights"] is None:
        st.session_state["user_modified_weights"] = dict(
            st.session_state["optimised_weights"]
        )

    user_weights = st.session_state["user_modified_weights"]
    ai_weights = st.session_state["optimised_weights"]

    st.caption(
        "Adjust each stock's weight by up to ±5 percentage points from the "
        "AI's recommendation. When you raise one weight, the largest other "
        "holding is reduced automatically so the total stays at 100%. "
        "Per academic research (Dietvorst et al., 2018), constrained editing "
        "of algorithm output increases user trust without compromising the "
        "model's recommendation."
    )

    cap = effective_cap(len(user_weights))

    # Sliders for each held stock, in AI-weight-descending order
    for ticker in held.index:
        ai_w = ai_weights[ticker]
        lower = max(ai_w - 0.05, 0.0)
        upper = min(ai_w + 0.05, cap)
        slider_key = f"weight_slider_{ticker}"

        # Initialise the widget key once so the slider starts at the AI weight
        if slider_key not in st.session_state:
            st.session_state[slider_key] = user_weights[ticker] * 100.0

        company_short = (
            universe.loc[ticker, "name"][:22] + "…"
            if ticker in universe.index and len(universe.loc[ticker, "name"]) > 22
            else (universe.loc[ticker, "name"] if ticker in universe.index else ticker)
        )

        delta_pp = (user_weights[ticker] - ai_w) * 100.0
        if abs(delta_pp) < 0.05:
            delta_str = "—"
            delta_colour = "#5A5A5A"
        else:
            delta_str = f"{delta_pp:+.1f}pp"
            delta_colour = "#0E8E8E" if delta_pp > 0 else "#C97A1F"

        cols = st.columns([2, 4, 1])
        with cols[0]:
            st.markdown(f"**{ticker}** · {company_short}")
            st.markdown(
                f"<span style='color:#5A5A5A; font-size:0.85rem;'>"
                f"AI: {ai_w*100:.1f}% &nbsp;·&nbsp; "
                f"Δ: <span style='color:{delta_colour}'>{delta_str}</span>"
                f"</span>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.slider(
                "Weight",
                min_value=float(lower * 100.0),
                max_value=float(upper * 100.0),
                step=0.5,
                format="%.1f%%",
                key=slider_key,
                on_change=_on_weight_slider_change,
                args=(ticker,),
                label_visibility="collapsed",
            )
        with cols[2]:
            st.markdown(
                f"<div style='text-align:right; font-weight:600; "
                f"padding-top:0.4rem;'>{user_weights[ticker]*100:.1f}%</div>",
                unsafe_allow_html=True,
            )

    # Modified-portfolio diagnostic + Reset button
    has_modifications = any(
        abs(user_weights[t] - ai_weights[t]) > 1e-6 for t in user_weights
    )

    if has_modifications:
        st.markdown("---")
        predictions_state = st.session_state["predictions"]
        modified_er = sum(
            user_weights[t] * float(predictions_state.loc[t, "predicted_return"])
            for t in user_weights
        )
        ai_er = float(st.session_state["expected_return"])
        total_pp_deviation = sum(
            abs(user_weights[t] - ai_weights[t]) for t in user_weights
        ) * 100.0
        delta_er_pp = (modified_er - ai_er) * 100.0
        st.markdown(
            f"**Modified portfolio**: expected return = "
            f"**{modified_er * 100:+.2f}%** "
            f"(<span style='color:#5A5A5A'>AI baseline: {ai_er*100:+.2f}%, "
            f"Δ {delta_er_pp:+.2f}pp</span>) &nbsp;·&nbsp; "
            f"Total weight deviation: **{total_pp_deviation:.1f}pp**",
            unsafe_allow_html=True,
        )
        st.caption(
            "Expected volatility for the modified portfolio is approximated "
            "using the AI's covariance matrix and shown more precisely in "
            "the next step."
        )

        st.button(
            "Reset to AI weights",
            key="reset_to_ai",
            on_click=_on_reset_to_ai_click,
        )


# ---------- Action buttons: Reject and Restart | Accept ----------

st.markdown("&nbsp;")
action_left, action_right = st.columns(2, gap="medium")

with action_left:
    if st.button(
        "Reject and Restart",
        key="op_reject",
        use_container_width=True,
    ):
        log_event(
            "decision_made",
            choice="reject",
            final_weights=dict(st.session_state["optimised_weights"]),
        )
        reset_for_restart()
        st.switch_page("pages/1_landing.py")

with action_right:
    if st.button(
        "Accept",
        type="primary",
        key="op_accept",
        use_container_width=True,
    ):
        umw = st.session_state["user_modified_weights"]
        ai = st.session_state["optimised_weights"]
        if umw is not None and any(abs(umw[t] - ai[t]) > 1e-6 for t in umw):
            choice = "modify"
            final = dict(umw)
            total_pp_deviation = sum(
                abs(umw[t] - ai[t]) for t in umw
            ) * 100.0
        else:
            choice = "accept"
            final = dict(ai)
            total_pp_deviation = 0.0

        st.session_state["decision"] = choice
        log_event(
            "decision_made",
            choice=choice,
            final_weights=final,
            total_pp_deviation=float(total_pp_deviation),
        )
        st.switch_page("pages/6_backtest.py")


render_page_footer()
