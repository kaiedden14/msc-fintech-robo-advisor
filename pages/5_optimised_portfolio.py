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
    decomp_bar_card_data,
    decomposition_summary,
    shap_bar_card_data,
    shap_summary_sentence,
)
from lib.data import load_predictions, load_universe_metadata
from lib.logger import log_event
from lib.portfolio import (
    compute_recommendation,
    round_weights_to_integer_pp,
    single_anchor_renorm,
)
from lib.sidebar import render_page_footer
from lib.state import clear_downstream_of, reset_for_restart


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
        st.metric("Expected return (1 quarter)", f"{expected_return * 100:+.2f}%")
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

# ---------- Phase 5d helpers (defined before the columns so the modify
# expander, now rendered inside the left column, can reference them) ----------


def _on_reset_to_ai_click() -> None:
    """Reset-to-AI button callback. Restores the rounded integer-pp baseline
    (matches the slider grid the user is actually editing on).

    Must run as an on_click callback (NOT inline after the button) because
    callbacks fire before the next rerun's widgets render — at inline time,
    every weight_slider_<ticker> widget has already been instantiated and
    Streamlit forbids modifying its session_state key.
    """
    ai = st.session_state["optimised_weights"]
    cap = effective_cap(len(ai))
    ai_baseline = round_weights_to_integer_pp(ai, max_weight=cap)
    st.session_state["user_modified_weights"] = dict(ai_baseline)
    for t in ai_baseline:
        slider_key = f"weight_slider_{t}"
        if slider_key in st.session_state:
            # round() before float-cast: avoids 14.000000000000002 vs 14.0
            # max-bound errors from IEEE 754 multiplication of 0.14 × 100.
            st.session_state[slider_key] = float(round(ai_baseline[t] * 100))
    log_event("reset_to_ai_weights", tickers=list(ai_baseline.keys()))


def _on_remove_zero_weight_click(ticker: str) -> None:
    """Drop a ticker from selected_tickers and force the recommendation to
    re-run on the remaining stocks. Called when the user clicks Remove on a
    stock the optimiser allocated 0% to.

    Runs as an on_click callback so it fires before the next rerun's widgets
    are instantiated — letting us safely delete stale weight_slider_<ticker>
    keys for the removed name.
    """
    current = list(st.session_state["selected_tickers"])
    if ticker not in current:
        return
    current.remove(ticker)
    new_selected = sorted(current)
    log_event(
        "ticker_deselected",
        ticker=ticker,
        current_selection_count=len(new_selected),
        source="optimised_portfolio_exclude_zero",
    )
    st.session_state["selected_tickers"] = new_selected
    # Stale slider widget keys would otherwise persist with the removed
    # ticker's old value; drop them all so the modify section re-initialises
    # against the freshly-computed recommendation.
    for k in list(st.session_state.keys()):
        if k.startswith("weight_slider_"):
            del st.session_state[k]
    clear_downstream_of("selected_tickers")


def _on_weight_slider_change(ticker: str) -> None:
    """Slider on_change callback: applies single-anchor renormalisation."""
    slider_key = f"weight_slider_{ticker}"
    new_value_pp = float(st.session_state[slider_key])
    new_value = new_value_pp / 100.0

    current = dict(st.session_state["user_modified_weights"])
    old_value = current.get(ticker, new_value)

    if abs(new_value - old_value) < 1e-9:
        return

    ai = st.session_state["optimised_weights"]
    cap = effective_cap(len(current))
    # Use the rounded baseline (same grid the sliders sit on) so the renorm
    # bounds align with the slider bounds — otherwise the renorm could
    # spill weight beyond a slider's visible range.
    ai_baseline = round_weights_to_integer_pp(ai, max_weight=cap)

    result = single_anchor_renorm(current, ai_baseline, ticker, new_value, cap)

    if result is None:
        st.session_state[slider_key] = float(round(old_value * 100))
        log_event(
            "slider_blocked",
            ticker=ticker,
            attempted_weight=float(new_value),
            from_weight=float(old_value),
        )
        return

    st.session_state["user_modified_weights"] = result
    for t in result:
        if t != ticker:
            # round() before float-cast: see _on_reset_to_ai_click comment
            st.session_state[f"weight_slider_{t}"] = float(round(result[t] * 100))

    log_event(
        "slider_moved",
        ticker=ticker,
        from_weight=float(old_value),
        to_weight=float(new_value),
        magnitude_pp=float((new_value - old_value) * 100.0),
    )


def _render_modify_section() -> None:
    """Render the Modify-weights expander inside the left column.

    Sliders operate in WHOLE percentage points (step=1, format='%.0f%%').
    The user_modified_weights state is initialised from a *rounded* copy of
    the AI weights so the slider grid is integer-aligned and the displays
    match the slider positions exactly (no off-by-half artefacts).
    """
    with st.expander("Modify your allocation (optional)", expanded=False):
        cap = effective_cap(len(st.session_state["optimised_weights"]))
        # Baseline used for slider bounds + the "has modifications" comparison —
        # AI weights rounded to nearest 1pp so the slider grid is integer-clean.
        ai_baseline = round_weights_to_integer_pp(
            st.session_state["optimised_weights"], max_weight=cap,
        )

        # Initialise user_modified_weights to the rounded baseline on first open
        if st.session_state["user_modified_weights"] is None:
            st.session_state["user_modified_weights"] = dict(ai_baseline)

        user_weights = st.session_state["user_modified_weights"]

        st.caption(
            "Adjust each stock's weight by up to ±5 percentage points from the "
            "AI's recommendation. When you raise one weight, the largest other "
            "holding is reduced automatically so the total stays at 100%. "
            "Per academic research (Dietvorst et al., 2018), constrained editing "
            "of algorithm output increases user trust without compromising the "
            "model's recommendation."
        )

        cap_pp = int(round(cap * 100))

        for ticker in held.index:
            ai_pp = int(round(ai_baseline[ticker] * 100))
            lower_pp = max(ai_pp - 5, 0)
            upper_pp = min(ai_pp + 5, cap_pp)
            slider_key = f"weight_slider_{ticker}"

            if slider_key not in st.session_state:
                # round() before float-cast: avoids slider value > max_value
                # errors from IEEE 754 multiplication (0.14 × 100 = 14.00…02).
                st.session_state[slider_key] = float(round(user_weights[ticker] * 100))

            company_short = (
                universe.loc[ticker, "name"][:22] + "…"
                if ticker in universe.index and len(universe.loc[ticker, "name"]) > 22
                else (universe.loc[ticker, "name"] if ticker in universe.index else ticker)
            )

            delta_pp = round((user_weights[ticker] - ai_baseline[ticker]) * 100)
            if delta_pp == 0:
                delta_str = "—"
                delta_colour = "#5A5A5A"
            else:
                delta_str = f"{delta_pp:+d}pp"
                delta_colour = "#0E8E8E" if delta_pp > 0 else "#C97A1F"

            cols = st.columns([2, 4, 1])
            with cols[0]:
                st.markdown(f"**{ticker}** · {company_short}")
                st.markdown(
                    f"<span style='color:#5A5A5A; font-size:0.85rem;'>"
                    f"AI: {ai_pp}% &nbsp;·&nbsp; "
                    f"Δ: <span style='color:{delta_colour}'>{delta_str}</span>"
                    f"</span>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.slider(
                    "Weight",
                    min_value=float(lower_pp),
                    max_value=float(upper_pp),
                    step=1.0,
                    format="%.0f%%",
                    key=slider_key,
                    on_change=_on_weight_slider_change,
                    args=(ticker,),
                    label_visibility="collapsed",
                )
            with cols[2]:
                st.markdown(
                    f"<div style='text-align:right; font-weight:600; "
                    f"padding-top:0.4rem;'>"
                    f"{int(round(user_weights[ticker] * 100))}%</div>",
                    unsafe_allow_html=True,
                )

        # "Modified" = any weight differs from the rounded baseline by ≥ 0.5pp
        # (i.e. at least one slider has moved a full step)
        has_modifications = any(
            abs(user_weights[t] - ai_baseline[t]) > 0.005 for t in user_weights
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
                abs(user_weights[t] - ai_baseline[t]) for t in user_weights
            ) * 100.0
            delta_er_pp = (modified_er - ai_er) * 100.0
            st.markdown(
                f"**Modified portfolio**: expected return = "
                f"**{modified_er * 100:+.2f}%** "
                f"(<span style='color:#5A5A5A'>AI baseline: {ai_er*100:+.2f}%, "
                f"Δ {delta_er_pp:+.2f}pp</span>) &nbsp;·&nbsp; "
                f"Total weight deviation: **{int(round(total_pp_deviation))}pp**",
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

        # ---- Stocks the optimiser allocated 0% to ----
        # Listed for the user to optionally remove from their selection
        # (the recommendation then re-runs on the remaining stocks).
        ai_raw = st.session_state["optimised_weights"]
        excluded_stocks = [
            t for t in st.session_state["selected_tickers"]
            if t not in ai_raw or ai_raw[t] < 0.005  # < 0.5% rounds to 0%
        ]
        if excluded_stocks:
            st.markdown("---")
            st.markdown("**Excluded by the recommendation**")
            st.caption(
                "These stocks received a 0% allocation. Remove any you no "
                "longer want considered — the recommendation will re-run on "
                "the remaining stocks. Minimum selection size is 5."
            )
            min_selection = 5
            removal_disabled = len(st.session_state["selected_tickers"]) <= min_selection
            for t in excluded_stocks:
                company = (
                    universe.loc[t, "name"] if t in universe.index else t
                )
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"**{t}** · {company}")
                with cols[1]:
                    st.button(
                        "Remove",
                        key=f"exclude_remove_{t}",
                        on_click=_on_remove_zero_weight_click,
                        args=(t,),
                        disabled=removal_disabled,
                        use_container_width=True,
                    )
            if removal_disabled:
                st.caption(
                    f"Can't remove — minimum {min_selection} stocks required. "
                    f"Go back to Asset Selection to swap stocks."
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
                    texttemplate="%{label}<br>%{percent:.0%}",
                    textfont=dict(size=11),
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Weight: %{percent:.0%}<br>"
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
        pie_event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="alloc_pie",
        )

        # Pie wedge click → focus that stock in the right-column detail panel
        # (same target state as clicking the table row below).
        if pie_event and pie_event.selection and pie_event.selection.points:
            clicked_label = pie_event.selection.points[0].get("label")
            if clicked_label and clicked_label != st.session_state["active_detail_ticker"]:
                st.session_state["active_detail_ticker"] = clicked_label
                log_event(
                    "stock_clicked_for_shap",
                    ticker=clicked_label,
                    source="screen5_pie",
                )
                st.rerun()

        # Allocation table with native single-row selection
        event = st.dataframe(
            table,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Ticker":         st.column_config.TextColumn(width="small"),
                "Company":        st.column_config.TextColumn(width="medium"),
                "Weight %":       st.column_config.NumberColumn(format="%.0f%%", width="small"),
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

    # Modify expander sits inside the left column directly below the
    # allocation card, filling the otherwise-dead space below the table
    # when the right-column detail panel is taller.
    _render_modify_section()


# ---------- Phase 5c: per-stock detail panel (right column) ----------


def _render_bar_card(items: list[dict]) -> None:
    """Render the bar-card list (label + value + magnitude bar per item).

    Pure HTML/CSS rendering — Streamlit doesn't have a native widget for
    this design. Colour: teal for positive, amber for negative. Bar widths
    are proportional to |value| / max(|value|) across the items.
    """
    if not items:
        st.caption("No data available.")
        return

    for item in items:
        is_zero = item.get("is_zero", False)
        color = "#0E8E8E" if item["is_positive"] else "#C97A1F"
        if is_zero:
            value_str = "0.0 pp"
            value_color = "#5A5A5A"
            fill_html = (
                "<div style='height:6px; background:#E4E2DC; "
                "border-radius:3px;'></div>"
            )
        else:
            sign = "+" if item["contribution_pp"] >= 0 else ""
            value_str = f"{sign}{item['contribution_pp']:.1f} pp"
            value_color = "#1A1A1A"
            fill_html = (
                "<div style='height:6px; background:#E4E2DC; "
                "border-radius:3px; overflow:hidden;'>"
                f"<div style='height:100%; width:{item['fill_pct']:.1f}%; "
                f"background:{color}; border-radius:3px;'></div>"
                "</div>"
            )
        st.markdown(
            "<div style='display:flex; justify-content:space-between; "
            "align-items:center; margin-top:10px; margin-bottom:6px;'>"
            f"<span style='color:#1A1A1A; font-size:0.95rem;'>{item['label']}</span>"
            f"<span style='color:{value_color}; font-variant-numeric:tabular-nums;"
            f" font-size:0.95rem;'>{value_str}</span>"
            "</div>"
            f"{fill_html}",
            unsafe_allow_html=True,
        )


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
        st.markdown(f"### {company}")
        st.caption(active)

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

        # --- Sub-card 1: Why this return outlook ---
        with st.container(border=True):
            st.markdown("**Why this return outlook**")
            if shap_r_row is not None:
                items = shap_bar_card_data(shap_r_row, top_n=4)
                if items:
                    _render_bar_card(items)
                    st.markdown("&nbsp;")
                    st.markdown(shap_summary_sentence(shap_r_row, active, kind="return"))
                else:
                    st.caption(
                        f"The model has no strong return signal for **{active}** — "
                        f"its forecast sits close to the model's typical value."
                    )
            else:
                st.caption("SHAP data not available for this stock.")

        # --- Sub-card 2: Why this risk outlook ---
        with st.container(border=True):
            st.markdown("**Why this risk outlook**")
            if shap_v_row is not None:
                items = shap_bar_card_data(shap_v_row, top_n=4)
                if items:
                    _render_bar_card(items)
                    st.markdown("&nbsp;")
                    st.markdown(shap_summary_sentence(shap_v_row, active, kind="risk"))
                else:
                    st.caption(
                        f"The model has no strong risk signal for **{active}** — "
                        f"its forecast sits close to the model's typical value."
                    )
            else:
                st.caption("SHAP data not available for this stock.")

        # --- Section divider: predictions ↑ allocation ↓ ---
        st.markdown(
            "<div class='ra-section-label'>Allocation reasoning</div>",
            unsafe_allow_html=True,
        )

        # --- Sub-card 3: Why this weight ---
        with st.container(border=True):
            st.markdown("**Why this weight**")
            if active in held.index:
                decomp_row = decomp_state.loc[active]
                _render_bar_card(decomp_bar_card_data(decomp_row))
                st.markdown("&nbsp;")
                st.markdown(decomposition_summary(decomp_row, active))
            else:
                st.caption(
                    f"The optimiser allocated no weight to **{active}**. "
                    f"The decomposition isn't meaningful for an unweighted position."
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
