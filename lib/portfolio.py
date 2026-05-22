"""Portfolio computation — optimiser + decomposition + state writes.

Single entry point `compute_recommendation()` called by Screen 5 on entry.
Idempotent: skips work when `optimised_weights` and `decomposition` are
already set in state. `clear_downstream_of("selected_tickers" or
"risk_profile")` upstream invalidates the cache by resetting state keys
to None so the next call recomputes.

Also exports `single_anchor_renorm()` — the locked redistribution rule
for user weight modifications (Phase 5d Modify expander).
"""

from typing import Any

import pandas as pd
import streamlit as st

from lib.bands import BANDS, MODERATE_BAND, effective_cap
from lib.data import (
    load_predictions,
    load_prices_clean,
    load_shap_return,
    load_shap_volatility,
)
from lib.logger import log_event
from src.models.decompose import decompose_weights
from src.models.optimise import optimise_portfolio


_DEVIATION_PP = 0.05      # Dietvorst-anchored ±5pp per-asset cap
_EPSILON = 1e-9
_FAIL_THRESHOLD = 1e-6    # If residual delta exceeds this after the loop, reject


def round_weights_to_integer_pp(
    weights: dict[str, float],
    max_weight: float = 1.0,
) -> dict[str, float]:
    """Round each weight to integer percentage points using the
    largest-remainders method, preserving sum=1.0.

    Used to give the Modify-slider UX clean integer-pp displays. Largest-
    remainders is the standard apportionment algorithm: floor every weight,
    then distribute the residual percentage points to the tickers with the
    largest fractional remainders. Unlike naive rounding-and-adjust-the-
    largest, this preserves cap-binding stocks at their cap (a ticker the
    optimiser pushed to exactly 10% stays at 10%) and distributes rounding
    error across mid-weight names instead of penalising the largest holding.

    Tickers already at the cap (cap_pp) are skipped for round-up but can
    still be rounded down if the residual is negative (over-floored due to
    floating-point quirks).

    Returns
    -------
    dict[str, float]
        Weights in decimal form, each an integer multiple of 0.01,
        summing exactly to 1.0.
    """
    cap_pp = round(max_weight * 100)
    floor_pp = {t: int(w * 100) for t, w in weights.items()}
    remainders = {t: (w * 100) - floor_pp[t] for t, w in weights.items()}
    needed = 100 - sum(floor_pp.values())

    rounded_pp = dict(floor_pp)
    if needed > 0:
        # Round up tickers with largest fractional remainders, skipping any
        # already at the cap.
        for t in sorted(remainders, key=lambda t: -remainders[t]):
            if needed == 0:
                break
            if rounded_pp[t] < cap_pp:
                rounded_pp[t] += 1
                needed -= 1
    elif needed < 0:
        # Over-floored (rare; floats > integer multiples) — round down the
        # tickers with the smallest fractional remainders.
        for t in sorted(remainders, key=lambda t: remainders[t]):
            if needed == 0:
                break
            if rounded_pp[t] > 0:
                rounded_pp[t] -= 1
                needed += 1

    return {t: pp / 100 for t, pp in rounded_pp.items()}


def single_anchor_renorm(
    weights: dict[str, float],
    ai_weights: dict[str, float],
    changed_ticker: str,
    new_value: float,
    effective_cap_val: float,
) -> dict[str, float] | None:
    """Single-anchor renormalisation per memory/dashboard_decisions.md.

    When the user moves the slider for ``changed_ticker`` to ``new_value``,
    redistribute the delta across other tickers — largest first, spill to
    next-largest if the current target hits its bound. Each other ticker
    has its own [ai - 5pp, ai + 5pp] range, clamped by [0, effective_cap].

    Parameters
    ----------
    weights
        Current weights dict (before the move).
    ai_weights
        AI's baseline weights (define the ±5pp window centre per ticker).
    changed_ticker
        Ticker whose slider was moved.
    new_value
        The new weight value (decimal) the user wants for ``changed_ticker``.
    effective_cap_val
        Upper cap on any single weight (from ``lib.bands.effective_cap``).

    Returns
    -------
    dict | None
        New weights dict if the move can be fully absorbed; None if not
        (in which case the caller should bounce the slider back).
    """
    old_value = weights[changed_ticker]
    delta = new_value - old_value
    if abs(delta) < _EPSILON:
        return dict(weights)

    result = dict(weights)
    result[changed_ticker] = new_value
    others = [t for t in weights if t != changed_ticker]

    if delta > 0:
        # User raised this ticker — reduce others by the surplus
        remaining = delta
        # Largest other weight first (has the most room to give back to AI baseline)
        others.sort(key=lambda t: -result[t])
        for t in others:
            if remaining < _EPSILON:
                break
            lower_bound = max(ai_weights[t] - _DEVIATION_PP, 0.0)
            absorbable = result[t] - lower_bound
            if absorbable > 0:
                taken = min(remaining, absorbable)
                result[t] -= taken
                remaining -= taken
        if remaining > _FAIL_THRESHOLD:
            return None
    else:
        # User lowered this ticker — give the freed weight to others
        remaining = abs(delta)
        # Largest other weight first (smaller available headroom to upper bound,
        # so the redistribution naturally concentrates on holdings that aren't
        # already at their ceilings)
        others.sort(key=lambda t: -result[t])
        for t in others:
            if remaining < _EPSILON:
                break
            upper_bound = min(ai_weights[t] + _DEVIATION_PP, effective_cap_val)
            absorbable = upper_bound - result[t]
            if absorbable > 0:
                given = min(remaining, absorbable)
                result[t] += given
                remaining -= given
        if remaining > _FAIL_THRESHOLD:
            return None

    return result


def compute_recommendation() -> dict[str, Any] | None:
    """Run the optimiser + decomposition for the current selection and risk band.

    Writes seven state keys atomically (all or nothing):
        predictions, shap_values_return, shap_values_volatility,
        optimised_weights, decomposition, expected_return, expected_vol.

    Returns
    -------
    dict | None
        The raw optimiser result dict on a fresh compute (so the caller can
        inspect it for diagnostics). None on cache hit.
    """
    # Cache hit — both expensive outputs already populated
    if (
        st.session_state["optimised_weights"] is not None
        and st.session_state["decomposition"] is not None
    ):
        return None

    selected = st.session_state["selected_tickers"]
    band_name = st.session_state["risk_profile"]
    config = BANDS[band_name]
    n = len(selected)
    cap = effective_cap(n)

    # Snapshot caches (loaded once per session via @st.cache_data)
    predictions_full = load_predictions()
    shap_r_full = load_shap_return()
    shap_v_full = load_shap_volatility()
    prices_close = load_prices_clean()["Close"]

    # Slice to the user's selection
    mu = predictions_full.loc[selected, "predicted_return"]
    sigma = predictions_full.loc[selected, "predicted_vol"]
    daily_returns = prices_close[selected].pct_change(fill_method=None).dropna(how="all")

    # Optimiser: mean-variance SLSQP with Ledoit-Wolf correlation + RF vols
    result = optimise_portfolio(
        predicted_returns=mu,
        predicted_vols=sigma,
        historical_returns=daily_returns,
        risk_aversion=config.risk_aversion,
        shrinkage_alpha=config.shrinkage_alpha,
        max_weight=cap,
    )

    # Decomposition: 5 counterfactual optimiser runs to attribute each weight
    # to the four contributions (return / variance / covariance / risk profile)
    decomp = decompose_weights(
        predicted_returns=mu,
        predicted_vols=sigma,
        historical_returns=daily_returns,
        full_result=result,
        risk_aversion=config.risk_aversion,
        moderate_risk_aversion=BANDS[MODERATE_BAND].risk_aversion,
        shrinkage_alpha=config.shrinkage_alpha,
        max_weight=cap,
    )

    # Persist atomically — every key written, none half-set
    st.session_state["predictions"] = pd.DataFrame(
        {"predicted_return": mu, "predicted_vol": sigma}
    )
    st.session_state["shap_values_return"] = shap_r_full[
        shap_r_full.index.get_level_values("ticker").isin(selected)
    ]
    st.session_state["shap_values_volatility"] = shap_v_full[
        shap_v_full.index.get_level_values("ticker").isin(selected)
    ]
    st.session_state["optimised_weights"] = {
        k: float(v) for k, v in result["weights"].items()
    }
    st.session_state["decomposition"] = decomp
    st.session_state["expected_return"] = float(result["expected_return"])
    st.session_state["expected_vol"] = float(result["expected_vol"])
    # Persist the optimiser's covariance matrix and ticker ordering so
    # Phase 6 (forward projection) can compute portfolio variance for
    # user-modified weights without recomputing Ledoit-Wolf shrinkage.
    st.session_state["cov_matrix"] = result["cov_matrix"]
    st.session_state["cov_matrix_tickers"] = list(result["weights"].index)

    log_event(
        "recommendation_generated",
        n_tickers=n,
        risk_band=band_name,
        risk_aversion=config.risk_aversion,
        shrinkage_alpha=config.shrinkage_alpha,
        max_weight=float(cap),
        expected_return=float(result["expected_return"]),
        expected_vol=float(result["expected_vol"]),
        converged=bool(result["converged"]),
        weights={k: float(v) for k, v in result["weights"].items()},
    )

    return result
