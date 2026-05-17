"""Event logging for the user study.

Appends JSON Lines events to a per-session file. Schema per
memory/event_logging_schema.md. Filename keyed by session start time +
short session-id prefix so logs are unambiguous and filesystem-safe;
participant_id is a per-event attribute (nullable until set on Landing).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


_LOG_DIR = Path("data/logs/sessions")


def _log_path() -> Path:
    """Return the JSONL file for this session."""
    started = st.session_state["session_started_at"]
    # Colons are illegal on Windows / awkward on macOS — replace
    started_safe = started.replace(":", "-")
    sid_short = st.session_state["session_id"][:8]
    return _LOG_DIR / f"session_{started_safe}_{sid_short}.jsonl"


def log_event(event_type: str, **payload: Any) -> None:
    """Append one JSON Lines event to the current session's log file.

    Common fields (ts, session_id, participant_id, event_type) are added
    automatically. The caller passes any event-specific kwargs.
    """
    event = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": st.session_state.get("session_id"),
        "participant_id": st.session_state.get("participant_id"),
        "event_type": event_type,
        **payload,
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event) + "\n")
