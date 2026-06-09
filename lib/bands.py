"""Risk-band -> optimiser-parameter mapping.

Locked decisions from memory/dashboard_decisions.md. The dashboard
controller maps band name -> (risk_aversion, shrinkage_alpha) here
and never exposes raw parameters to the user.
"""

from typing import NamedTuple


class BandConfig(NamedTuple):
    risk_aversion: float
    shrinkage_alpha: float


BANDS: dict[str, BandConfig] = {
    "Cautious":    BandConfig(risk_aversion=5.0, shrinkage_alpha=0.85),
    "Balanced":    BandConfig(risk_aversion=2.0, shrinkage_alpha=0.75),
    "Growth":      BandConfig(risk_aversion=1.0, shrinkage_alpha=0.55),
    "Adventurous": BandConfig(risk_aversion=0.5, shrinkage_alpha=0.35),
}

# Band used as the decomposition baseline (risk-contribution = 0 here).
MODERATE_BAND = "Balanced"


def effective_cap(n: int) -> float:
    """Headroom-aware per-asset weight cap.

    Returns max(0.10, 1.5/n), guarantees ~50% solver headroom at every
    selection size so risk bands stay differentiated across n in [5, 15].
    See memory/dashboard_decisions.md for the geometric derivation.
    """
    return max(0.10, 1.5 / n)
