"""Analytical decomposition of each optimised portfolio weight.

The optimiser is a quadratic programme, not a model, so SHAP does not
apply to the weights it produces. This module instead runs five
counterfactual versions of the optimiser and reads the contributions of
each input from the differences between them.

Given the full optimiser run, four counterfactuals are computed by
neutralising one input at a time:

1. Return signal      : set ``alpha = 1`` so all stocks share the same
                        expected return.
2. Individual vol     : replace predicted vols with their cross-sectional
                        mean.
3. Diversification    : replace the correlation matrix with the identity.
4. Risk preference    : replace the user's lambda with the Balanced band's
                        baseline lambda.

A fifth run applies all four neutralisations simultaneously, producing
the all-neutralised anchor.

For every ticker, each contribution is the difference between the full
weight and the corresponding counterfactual. The interaction term, the
gap between the total deviation from the anchor and the sum of the four
contributions, is reported explicitly rather than forced to zero so
nothing is hidden.

This decomposition layer is what the dashboard surfaces under "Why this
weight". It complements the SHAP cards (which explain the per-stock
predictions) by explaining the per-stock allocations.
"""

import numpy as np
import pandas as pd

from src.models.optimise import optimise_portfolio


def decompose_weights(
    predicted_returns: pd.Series,
    predicted_vols: pd.Series,
    historical_returns: pd.DataFrame,
    full_result: dict,
    risk_aversion: float = 2.0,
    moderate_risk_aversion: float = 2.0,
    shrinkage_alpha: float = 0.75,
    max_weight: float = 0.10,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Decompose portfolio weights into four contributions via five counterfactual
    optimiser runs. Reference anchor is the all-neutralised portfolio (not 1/n),
    which correctly absorbs cap effects.

    Contributions (each = w_full - w_counterfactual):
      return:      return signal's effect, shrinkage_alpha=1 neutralises it
      variance:    individual vol signal, replace predicted vols with mean vol
      covariance:  off-diagonal structure, replace correlation with identity
      risk:        user's risk aversion vs moderate baseline

    Interaction = (w_full - w_eq) - sum(four contributions).
    Reported explicitly; not forced to zero.

    Parameters
    ----------
    predicted_returns : pd.Series
        Original RF return predictions passed to optimise_portfolio.
    predicted_vols : pd.Series
        Original RF volatility predictions passed to optimise_portfolio.
    historical_returns : pd.DataFrame
        Daily returns for correlation estimation.
    full_result : dict
        Output of optimise_portfolio, w_full lifted from here to avoid
        recomputing the sixth run.
    risk_aversion : float
        The lambda used in the full run.
    moderate_risk_aversion : float
        Neutral lambda for risk contribution (default 2.0 = moderate profile).
    shrinkage_alpha : float
        Shrinkage used in the full run.
    max_weight : float
        Per-asset cap used in the full run.
    lookback : int
        Lookback window used in the full run.

    Returns
    -------
    pd.DataFrame
        Index: ticker. Columns: return, variance, covariance, risk,
        interaction, total_deviation.
    """
    shared = dict(
        historical_returns=historical_returns,
        max_weight=max_weight,
        lookback=lookback,
    )

    w_full = full_result["weights"]
    tickers = w_full.index.tolist()
    mean_vol = predicted_vols.mean()
    n = len(tickers)
    identity = np.eye(n)

    # --- Five counterfactual runs ---

    # 1. Neutralise return: collapse return predictions to their mean
    r_ret = optimise_portfolio(
        predicted_returns=predicted_returns,
        predicted_vols=predicted_vols,
        risk_aversion=risk_aversion,
        shrinkage_alpha=1.0,          # all assets get same expected return
        **shared,
    )["weights"].reindex(tickers).fillna(0)

    # 2. Neutralise variance: replace predicted vols with cross-sectional mean
    flat_vols = pd.Series(mean_vol, index=predicted_vols.index)
    r_var = optimise_portfolio(
        predicted_returns=predicted_returns,
        predicted_vols=flat_vols,
        risk_aversion=risk_aversion,
        shrinkage_alpha=shrinkage_alpha,
        **shared,
    )["weights"].reindex(tickers).fillna(0)

    # 3. Neutralise covariance: replace correlation matrix with identity
    r_cov = optimise_portfolio(
        predicted_returns=predicted_returns,
        predicted_vols=predicted_vols,
        risk_aversion=risk_aversion,
        shrinkage_alpha=shrinkage_alpha,
        override_corr=identity,
        **shared,
    )["weights"].reindex(tickers).fillna(0)

    # 4. Neutralise risk profile: replace user's lambda with moderate baseline
    r_risk = optimise_portfolio(
        predicted_returns=predicted_returns,
        predicted_vols=predicted_vols,
        risk_aversion=moderate_risk_aversion,
        shrinkage_alpha=shrinkage_alpha,
        **shared,
    )["weights"].reindex(tickers).fillna(0)

    # 5. All-neutralised reference: anchor portfolio
    r_eq = optimise_portfolio(
        predicted_returns=predicted_returns,
        predicted_vols=flat_vols,
        risk_aversion=moderate_risk_aversion,
        shrinkage_alpha=1.0,
        override_corr=identity,
        **shared,
    )["weights"].reindex(tickers).fillna(0)

    # --- Compute contributions ---
    w = w_full.reindex(tickers).fillna(0)

    c_return     = w - r_ret
    c_variance   = w - r_var
    c_covariance = w - r_cov
    c_risk       = w - r_risk
    total_dev    = w - r_eq
    interaction  = total_dev - (c_return + c_variance + c_covariance + c_risk)

    return pd.DataFrame({
        "return":          c_return,
        "variance":        c_variance,
        "covariance":      c_covariance,
        "risk":            c_risk,
        "interaction":     interaction,
        "total_deviation": total_dev,
    }, index=tickers)
