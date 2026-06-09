"""Landing page.

Hero, three numbered "how it works" cards, three value props, primary CTA
to Risk Profile, and a secondary "How it works" modal for deeper detail.
Participant identity is set by the researcher in the sidebar (out-of-band
from the participant-facing flow).
"""

import streamlit as st

from lib.logger import log_event
from lib.sidebar import render_page_footer


@st.dialog("How it works")
def show_methodology() -> None:
    st.markdown(
        """
**Behind the scenes**

1. You select 5–15 FTSE 100 stocks.
2. Two machine-learning models forecast each stock's expected return
   and volatility for the quarter ahead.
3. A mean-variance optimiser combines those forecasts with a historical
   covariance estimate and your chosen risk profile to recommend portfolio
   weights.
4. The recommendation is explained at two levels:
   - **Per stock**: which inputs drove each prediction.
   - **Per portfolio**: how much each factor (return signal, individual
     volatility, diversification, risk profile) contributed to the weight
     on each stock.
5. You can **accept** the recommendation, **modify** any weight by up
   to ±5 percentage points, or **reject** and start over.
"""
    )


# ---------- Hero ----------

st.markdown(
    "<div style='display:flex; align-items:center; gap:0.9rem; "
    "margin-top:0.5rem;'>"
    "<div style='width:6px; height:48px; background:#0F2540; border-radius:3px;'></div>"
    "<div style='font-size:2.6rem; font-weight:700; color:#0F2540; "
    "line-height:1.05;'>Hybrid Robo-Advisor</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-size:1.15rem; color:#5A5A5A; max-width:720px; "
    "margin:0.75rem 0 2rem 0; line-height:1.55;'>"
    "Smarter portfolios for UK retail investors, built around your choices, "
    "explained at every step, with you in control."
    "</div>",
    unsafe_allow_html=True,
)


# ---------- Three-step "how it works" cards ----------

st.markdown(
    "<div style='font-size:0.85rem; font-weight:600; color:#5A5A5A; "
    "letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.6rem;'>"
    "How it works</div>",
    unsafe_allow_html=True,
)

s1, s2, s3 = st.columns(3, gap="medium")
_STEPS_HOWTO = [
    ("01", "Choose your stocks",
     "Pick 5–15 FTSE 100 companies you want to consider. The product never adds "
     "or removes stocks from your shortlist."),
    ("02", "See the AI's view",
     "Two machine-learning models forecast each stock's expected return and risk "
     "for the next quarter. A mean-variance optimiser turns those into weights."),
    ("03", "Stay in control",
     "Every weight comes with a plain-English explanation. You can accept, "
     "modify within ±5 percentage points, or reject."),
]
for col, (num, title, body) in zip([s1, s2, s3], _STEPS_HOWTO):
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:1.1rem; font-weight:700; color:#0E8E8E; "
                f"margin-bottom:0.3rem;'>{num}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{title}**")
            st.markdown(
                f"<div style='color:#5A5A5A; font-size:0.92rem; "
                f"line-height:1.45;'>{body}</div>",
                unsafe_allow_html=True,
            )


# ---------- Value props ----------

st.markdown("&nbsp;")
st.markdown(
    "<div style='font-size:0.85rem; font-weight:600; color:#5A5A5A; "
    "letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.6rem;'>"
    "Why it's different</div>",
    unsafe_allow_html=True,
)

v1, v2, v3 = st.columns(3, gap="medium")
_VALUE_PROPS = [
    ("Transparent by design",
     "See exactly why each stock was forecast the way it was, and what drove "
     "its weight in your portfolio."),
    ("Built on your shortlist",
     "The algorithm respects your selection. It recommends how to weight the "
     "stocks you've chosen, not which ones to hold."),
    ("Adjustable within limits",
     "Fine-tune individual weights without breaking the optimiser's overall "
     "structure. Control without chaos."),
]
for col, (title, body) in zip([v1, v2, v3], _VALUE_PROPS):
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(
                f"<div style='color:#5A5A5A; font-size:0.92rem; "
                f"line-height:1.45;'>{body}</div>",
                unsafe_allow_html=True,
            )


# ---------- CTA ----------

st.markdown("&nbsp;")
cta_col, link_col = st.columns([1, 4])

with cta_col:
    cta_disabled = not st.session_state["participant_id"]
    if st.button(
        "Start Building",
        type="primary",
        disabled=cta_disabled,
        use_container_width=True,
        key="landing_cta",
    ):
        log_event("page_navigation", from_page="landing", to_page="risk_profile")
        st.switch_page("pages/2_risk_profile.py")

with link_col:
    if st.button(
        "How it works in detail →",
        key="meth_btn",
        type="secondary",
    ):
        log_event("methodology_modal_opened")
        show_methodology()


render_page_footer()
