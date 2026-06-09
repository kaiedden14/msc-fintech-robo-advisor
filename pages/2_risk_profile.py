"""Risk Profile page, Phase 2.

4-band segmented control + per-band explanation + investment amount.
State writes: risk_profile (with downstream clear), investment_amount.
"""

import streamlit as st

from lib.bands import BANDS
from lib.logger import log_event
from lib.sidebar import render_page_footer
from lib.state import clear_downstream_of


_BAND_NAMES = list(BANDS.keys())  # ["Cautious", "Balanced", "Growth", "Adventurous"]

_BAND_EXPLANATIONS: dict[str, str] = {
    "Cautious": (
        "Prioritises stability. The portfolio leans on stocks the model "
        "considers less volatile, and the return signal is heavily down-"
        "weighted. Lower expected return, lower expected volatility."
    ),
    "Balanced": (
        "Balances expected return against volatility. The default profile "
        "and the reference baseline for the weight-decomposition view."
    ),
    "Growth": (
        "Tilts toward stocks the model expects to outperform. Accepts more "
        "portfolio volatility in exchange for a stronger lean on the "
        "model's return predictions."
    ),
    "Adventurous": (
        "Leans furthest into the model's return predictions. Highest "
        "expected return, highest expected volatility, suitable if you "
        "want the recommendation to reflect the model's directional view."
    ),
}


# Guard: require Landing to be completed first
if not st.session_state["participant_id"]:
    st.title("Risk Profile")
    st.warning("Please complete the Landing page first.")
    if st.button("Back to Landing"):
        st.switch_page("pages/1_landing.py")
    render_page_footer()
    st.stop()


st.title("Risk Profile")
st.caption("Set your investment amount and risk preference. Both can be changed later.")


# ---------- Investment amount ----------

st.markdown("&nbsp;")
st.markdown("**Investment amount**")

current_amount = st.session_state["investment_amount"] or 10_000.0
new_amount = st.number_input(
    "Investment amount (GBP)",
    min_value=100.0,
    max_value=1_000_000.0,
    value=float(current_amount),
    step=500.0,
    format="%.0f",
    label_visibility="collapsed",
)

if new_amount != st.session_state["investment_amount"]:
    prev = st.session_state["investment_amount"]
    st.session_state["investment_amount"] = float(new_amount)
    log_event("investment_amount_set", previous=prev, new=float(new_amount))


# ---------- Risk band ----------

st.markdown("&nbsp;")
st.markdown("**Risk preference**")

current_band = st.session_state["risk_profile"]
selected_band = st.segmented_control(
    "Risk band",
    options=_BAND_NAMES,
    default=current_band if current_band else None,
    label_visibility="collapsed",
)

if selected_band and selected_band != current_band:
    prev = current_band
    st.session_state["risk_profile"] = selected_band
    if prev is None:
        log_event("risk_band_set", band=selected_band)
    else:
        log_event("risk_band_changed", from_band=prev, to_band=selected_band)
    clear_downstream_of("risk_profile")

# Dynamic explanation block
if selected_band:
    cfg = BANDS[selected_band]
    with st.container(border=True):
        st.markdown(f"**{selected_band}**")
        st.markdown(_BAND_EXPLANATIONS[selected_band])
        st.caption(
            f"Optimiser parameters: risk-aversion λ = {cfg.risk_aversion}, "
            f"return shrinkage α = {cfg.shrinkage_alpha}"
        )
else:
    with st.container(border=True):
        st.markdown(
            "<div style='color:#5A5A5A; padding:0.5rem 0;'>"
            "Select a risk band to see its profile."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------- Continue button ----------

st.markdown("&nbsp;")
back_col, cont_col = st.columns([1, 1], gap="small")

with back_col:
    if st.button("Back", key="rp_back", use_container_width=True):
        st.switch_page("pages/1_landing.py")

with cont_col:
    cont_disabled = (
        not st.session_state["risk_profile"]
        or not st.session_state["investment_amount"]
        or st.session_state["investment_amount"] < 100
    )
    if st.button(
        "Continue",
        type="primary",
        disabled=cont_disabled,
        use_container_width=True,
        key="rp_continue",
    ):
        log_event(
            "page_navigation", from_page="risk_profile", to_page="asset_selection"
        )
        st.switch_page("pages/3_asset_selection.py")

if cont_disabled and selected_band is None:
    st.caption("Select a risk band to continue.")


render_page_footer()
