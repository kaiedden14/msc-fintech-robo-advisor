"""Session-state schema and idempotent initialiser.

Mirrors the schema approved in Phase 0. AppState is documentation-only —
st.session_state is dynamic, but the TypedDict serves as the canonical
reference for every key name, type, and default.
"""

import uuid
from datetime import datetime
from typing import Any, Literal, TypedDict

import streamlit as st


RiskBand = Literal["Cautious", "Balanced", "Growth", "Adventurous"]
Decision = Literal["accept", "modify", "reject"]


class AppState(TypedDict, total=False):
    # User inputs
    risk_profile:           RiskBand | None
    investment_amount:      float | None
    selected_tickers:       list[str]
    # Derived / computed
    correlation_flags:      list[tuple[str, str, float]]
    predictions:            Any  # pd.DataFrame or None
    shap_values_return:     Any  # pd.DataFrame or None
    shap_values_volatility: Any  # pd.DataFrame or None
    optimised_weights:      dict[str, float] | None
    user_modified_weights:  dict[str, float] | None
    decomposition:          Any  # pd.DataFrame or None
    expected_return:        float | None
    expected_vol:           float | None
    cov_matrix:             Any  # np.ndarray or None
    cov_matrix_tickers:     list[str] | None
    projection_results:     dict | None
    # Screen 5 focus
    active_detail_ticker:   str | None
    # Decision
    decision:               Decision | None
    # UI flags
    consent_acknowledged:   bool
    session_start_logged:   bool
    # Event logging plumbing
    session_id:             str
    participant_id:         str | None
    session_started_at:     str


DEFAULTS: dict[str, Any] = {
    "risk_profile":           None,
    "investment_amount":      None,
    "selected_tickers":       [],
    "correlation_flags":      [],
    "predictions":            None,
    "shap_values_return":     None,
    "shap_values_volatility": None,
    "optimised_weights":      None,
    "user_modified_weights":  None,
    "decomposition":          None,
    "expected_return":        None,
    "expected_vol":           None,
    "cov_matrix":             None,
    "cov_matrix_tickers":     None,
    "projection_results":     None,
    "active_detail_ticker":   None,
    "decision":               None,
    # Consent is handled out-of-band via a paper form signed in person before
    # the session. The flag is kept in state for the event-log schema but
    # defaults to True so the in-app flow is not gated on a duplicate capture.
    "consent_acknowledged":   True,
    "session_start_logged":   False,
    "participant_id":         None,
    # Rebalancing — fresh recommendation produced by the re-check action
    "fresh_recommendation":   None,  # dict with 'weights', 'expected_return', 'expected_vol'
}


# Keys downstream of each input. When the input changes, downstream keys
# are invalidated. Single source of truth for the "clear downstream of X"
# pattern that prevents stale-data bugs (see Phase 0 schema lifecycle).
DOWNSTREAM_OF: dict[str, list[str]] = {
    "selected_tickers": [
        "correlation_flags", "predictions",
        "shap_values_return", "shap_values_volatility",
        "optimised_weights", "user_modified_weights",
        "decomposition", "expected_return", "expected_vol",
        "cov_matrix", "cov_matrix_tickers",
        "projection_results", "active_detail_ticker",
    ],
    "risk_profile": [
        "optimised_weights", "user_modified_weights",
        "decomposition", "expected_return", "expected_vol",
        "cov_matrix", "cov_matrix_tickers",
        "projection_results",
    ],
}


def init_state() -> None:
    """Initialise st.session_state with the schema defaults if absent.

    Idempotent: runs on every script rerun but only sets keys that don't
    already exist. session_id and session_started_at are populated exactly
    once, on first run.
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex
        st.session_state["session_started_at"] = datetime.now().isoformat(timespec="seconds")

    for key, default in DEFAULTS.items():
        if key in st.session_state:
            continue
        # Copy mutable defaults so they don't share a reference across sessions
        if isinstance(default, list):
            st.session_state[key] = list(default)
        elif isinstance(default, dict):
            st.session_state[key] = dict(default)
        else:
            st.session_state[key] = default


def clear_downstream_of(key: str) -> None:
    """Reset every state key downstream of the named input.

    Call this from the setter for risk_profile or selected_tickers so a
    change to either input invalidates every dependent computation.
    """
    for downstream_key in DOWNSTREAM_OF.get(key, []):
        st.session_state[downstream_key] = DEFAULTS[downstream_key]
        # Preserve list/dict isolation
        if isinstance(DEFAULTS[downstream_key], list):
            st.session_state[downstream_key] = []
        elif isinstance(DEFAULTS[downstream_key], dict):
            st.session_state[downstream_key] = {}


# Keys preserved across a "Reject and Restart" — the participant identity
# and consent acknowledgement stay so the participant doesn't have to
# re-consent. Everything portfolio-related is wiped.
_RESTART_PRESERVE: set[str] = {
    "session_id", "session_started_at", "session_start_logged",
    "participant_id", "consent_acknowledged",
}


def reset_for_restart() -> None:
    """Clear portfolio-related state and any dynamic slider widget keys.

    Called when the participant clicks "Reject and Restart" on Screen 5.
    Returns the user to Landing with a fresh portfolio slate, while
    keeping the session and participant identity intact.
    """
    for key in list(DEFAULTS.keys()):
        if key in _RESTART_PRESERVE:
            continue
        default = DEFAULTS[key]
        if isinstance(default, list):
            st.session_state[key] = []
        elif isinstance(default, dict):
            st.session_state[key] = {}
        else:
            st.session_state[key] = default

    # Dynamic slider widget keys (weight_slider_<ticker>) are not in DEFAULTS;
    # remove them so any residual values don't bleed into a fresh allocation.
    for k in list(st.session_state.keys()):
        if k.startswith("weight_slider_"):
            del st.session_state[k]
