"""Landing page — Phase 2.

Hero, two value bullets, consent banner, participant ID selector, primary
CTA to Risk Profile, secondary methodology stub modal. First state writes
of the session: consent_acknowledged, participant_id.
"""

import streamlit as st

from lib.logger import log_event
from lib.sidebar import render_page_footer


_VALID_PARTICIPANT_IDS = [f"P{i:02d}" for i in range(1, 11)]  # P01..P10


@st.dialog("Methodology")
def show_methodology() -> None:
    st.markdown(
        """
**How the Hybrid Robo-Advisor works**

1. You select 5–15 FTSE 100 stocks from a curated universe.
2. Two machine-learning models forecast each selected stock's
   expected return and realised volatility for the month ahead.
3. A mean-variance optimiser combines those forecasts with a historical
   covariance estimate and your chosen risk profile to recommend a set
   of portfolio weights.
4. The recommendation is explained at two levels:
   - **Per stock**: SHAP attributions show which input features drove
     each prediction.
   - **Per portfolio**: An analytical decomposition shows how much each
     factor (return signal, individual volatility, diversification, and
     risk profile) contributed to the weight on each stock.
5. You can **accept** the recommendation, **modify** any weight by up
   to ±5 percentage points, or **reject** and start over.

The product is an academic prototype for an MSc Financial Technology
project. Full methodology details are in the accompanying write-up.
"""
    )


# ---------- Hero ----------

st.markdown(
    "<div style='font-size:2.5rem; font-weight:700; color:#0F2540; "
    "line-height:1.1; margin-top:0.5rem;'>Hybrid Robo-Advisor</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-size:1.1rem; color:#5A5A5A; margin-bottom:2rem;'>"
    "Machine-learning portfolio recommendations for UK retail investors, "
    "with full transparency over every prediction and every weight."
    "</div>",
    unsafe_allow_html=True,
)


# ---------- Two value bullets ----------

col_a, col_b = st.columns(2, gap="large")
with col_a:
    with st.container(border=True):
        st.markdown("**You choose the stocks**")
        st.markdown(
            "Pick the 5–15 FTSE 100 companies you want to consider. "
            "The product never adds or removes stocks from your shortlist — "
            "it only recommends how to weight them."
        )
with col_b:
    with st.container(border=True):
        st.markdown("**You see why**")
        st.markdown(
            "Every recommendation comes with two layers of explanation: "
            "what drove each stock's forecast, and what drove its weight "
            "in the final portfolio."
        )


# ---------- Consent banner ----------

if not st.session_state["consent_acknowledged"]:
    st.markdown("&nbsp;")
    with st.container(border=True):
        st.markdown("**Research data collection**")
        st.markdown(
            "This session is being recorded for research as part of an MSc "
            "Financial Technology project. Anonymous interaction data only "
            "— no names, contact details, or real financial information are "
            "captured. You may withdraw at any time."
        )
        if st.button("I understand", type="primary", key="consent_ack_btn"):
            st.session_state["consent_acknowledged"] = True
            log_event("consent_acknowledged")
            st.rerun()


# ---------- Participant ID ----------

st.markdown("&nbsp;")
st.markdown("**Session set-up**")
st.caption("The researcher will set the participant ID before you begin.")

selected_pid = st.selectbox(
    "Participant ID",
    options=[""] + _VALID_PARTICIPANT_IDS,
    index=(_VALID_PARTICIPANT_IDS.index(st.session_state["participant_id"]) + 1)
    if st.session_state["participant_id"] in _VALID_PARTICIPANT_IDS
    else 0,
    label_visibility="collapsed",
    placeholder="Select…",
)

if selected_pid and selected_pid != st.session_state["participant_id"]:
    prev = st.session_state["participant_id"]
    st.session_state["participant_id"] = selected_pid
    log_event("participant_id_set", previous=prev, new=selected_pid)


# ---------- CTA + methodology link ----------

st.markdown("&nbsp;")
cta_col, meth_col = st.columns([1, 3])

with cta_col:
    cta_disabled = not (
        st.session_state["consent_acknowledged"]
        and st.session_state["participant_id"]
    )
    if st.button(
        "Start Building",
        type="primary",
        disabled=cta_disabled,
        use_container_width=True,
        key="landing_cta",
    ):
        log_event("page_navigation", from_page="landing", to_page="risk_profile")
        st.switch_page("pages/2_risk_profile.py")

    if cta_disabled:
        missing = []
        if not st.session_state["consent_acknowledged"]:
            missing.append("acknowledge the data-collection notice")
        if not st.session_state["participant_id"]:
            missing.append("select a participant ID")
        st.caption(f"To continue, {' and '.join(missing)}.")

with meth_col:
    if st.button("Methodology", key="meth_btn", type="secondary"):
        log_event("methodology_modal_opened")
        show_methodology()


render_page_footer()
