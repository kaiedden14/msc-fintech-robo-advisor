"""Per-participant portfolio persistence.

Saves each participant's accepted portfolio to a JSON file under
data/portfolios/<participant_id>.json so the rebalancing page can compute
realised performance from the actual investment date forward, and so a
participant returning on a later day sees their portfolio with one more
day of accumulated performance each time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


_PORTFOLIOS_DIR = Path("data/portfolios")


def _portfolio_path(participant_id: str) -> Path:
    """Return the on-disk JSON path for a participant's saved portfolio."""
    return _PORTFOLIOS_DIR / f"{participant_id}.json"


def save_portfolio(
    participant_id: str,
    weights: dict[str, float],
    selected_tickers: list[str],
    investment_amount: float,
    risk_band: str,
    decision: str,
    investment_date: str | None = None,
) -> None:
    """Persist a participant's portfolio to disk.

    Overwrites any existing record for this participant. Called from the
    Optimised Portfolio page on accept/modify, and from the Rebalancing
    page when the participant adopts an updated recommendation.

    Parameters
    ----------
    investment_date : str, optional
        ISO date (YYYY-MM-DD). Defaults to today.
    """
    if investment_date is None:
        investment_date = datetime.now().date().isoformat()

    record = {
        "participant_id":    participant_id,
        "investment_date":   investment_date,
        "weights":           {k: float(v) for k, v in weights.items()},
        "selected_tickers":  list(selected_tickers),
        "investment_amount": float(investment_amount),
        "risk_band":         risk_band,
        "decision":          decision,
        "saved_at":          datetime.now().isoformat(timespec="seconds"),
    }

    _PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
    _portfolio_path(participant_id).write_text(json.dumps(record, indent=2))


def load_portfolio(participant_id: str) -> dict | None:
    """Load a participant's saved portfolio. Returns None if absent."""
    if not participant_id:
        return None
    path = _portfolio_path(participant_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def delete_portfolio(participant_id: str) -> bool:
    """Remove a participant's saved portfolio file. Returns True if a file
    was actually deleted, False if no file existed."""
    if not participant_id:
        return False
    path = _portfolio_path(participant_id)
    if not path.exists():
        return False
    path.unlink()
    return True
