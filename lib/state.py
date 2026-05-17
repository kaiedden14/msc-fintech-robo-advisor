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
    "projection_results":     None,
    "active_detail_ticker":   None,
    "decision":               None,
    "consent_acknowledged":   False,
    "session_start_logged":   False,
    "participant_id":         None,
}


# Keys downstream of each input. When the input changes, downstream keys
# are invalidated. Single source of truth for the "clear downstream of X"
# pattern that prevents stale-data bugs (see Phase 0 schema lifecycle).
DOWNSTREAM_OF: dict[str, list[str]] = {
    "selected_tickers": [
        "correlation_flags", "predictions",
        "shap_values_return", "shap_values_volatility",
        "optimised_weights", "user_modified_weights",
        "decomposition", "projection_results",
        "active_detail_ticker",
    ],
    "risk_profile": [
        "optimised_weights", "user_modified_weights",
        "decomposition", "projection_results",
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
