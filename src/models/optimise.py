"""Mean-variance portfolio optimisation.

This module solves the constrained quadratic programme that turns the
per-stock predictions into a recommended set of portfolio weights. The
objective minimised by SLSQP is

    (lambda / 2) * w' Sigma w  -  w' mu_shrunk

subject to ``sum(w) == 1`` and ``0 <= w_i <= cap`` for every ticker.

The two methodological choices that make this a "hybrid" rather than a
textbook mean-variance run:

- ``Sigma = D Cov D``, where ``D`` is a diagonal matrix of RF-predicted
  forward volatilities and ``Cov`` is a Ledoit-Wolf shrunk correlation
  matrix estimated from 252 days of historical daily returns. The
  volatility model is the empirical backbone of the optimiser; using
  historical vols on the diagonal would discard that signal. Held-out
  Spearman rank correlation for the vol model is 0.46.

- ``mu_shrunk = (1 - alpha) * mu_rf + alpha * mu_bar``, with the
  shrinkage intensity ``alpha`` set by the user's risk band. The return
  model has a modest but positive rank-ordering signal (held-out Spearman
  0.09); shrinkage pulls each stock's expected return toward the
  cross-sectional mean of the user's selection so the optimiser does not
  over-react to a small signal.

A defensive nearest-positive-semi-definite projection absorbs floating
point drift, and an equal-weight fallback is returned on non-convergence
so the dashboard never shows the user a broken recommendation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def build_covariance_matrix(
    predicted_vols: pd.Series,
    historical_returns: pd.DataFrame,
    lookback: int = 252,
    override_corr: np.ndarray = None,
) -> np.ndarray:
    """
    Construct a forward-looking covariance matrix by combining RF-predicted
    volatilities (diagonal) with a Ledoit-Wolf historical correlation matrix
    (off-diagonal).

    Sigma = D @ C @ D

    where D is a diagonal matrix of RF-predicted volatilities (sigma, not
    sigma^2) and C is the Ledoit-Wolf regularised correlation matrix.

    Parameters
    ----------
    predicted_vols : pd.Series
        RF-predicted forward volatilities (sigma units, annualised).
        Index: ticker.
    historical_returns : pd.DataFrame
        Daily returns, columns=tickers, rows=dates. Used for correlation only.
    lookback : int
        Rolling window in trading days for correlation estimation.

    Returns
    -------
    cov_matrix : np.ndarray
        Covariance matrix of shape (n, n). Guaranteed PSD by Ledoit-Wolf;
        nearest-PSD projection applied as defensive backstop.
    corr : np.ndarray
        Ledoit-Wolf regularised correlation matrix (n, n), after diagonal
        correction. Returned as estimated, not re-derived from cov_matrix.
    tickers : list[str]
        Ticker order matching rows/columns of both matrices.
    """
    tickers = predicted_vols.index.tolist()
    ret_window = historical_returns[tickers].tail(lookback).dropna(axis=1)

    # Log any tickers silently dropped due to NaN in the return window
    dropped = set(tickers) - set(ret_window.columns)
    if dropped:
        import warnings
        warnings.warn(
            f"build_covariance_matrix: dropped {len(dropped)} ticker(s) "
            f"with NaN in {lookback}-day return window: {sorted(dropped)}",
            stacklevel=2,
        )

    # Align tickers to those present in the return window
    tickers = ret_window.columns.tolist()

    if not tickers:
        raise ValueError(
            f"build_covariance_matrix: all {len(predicted_vols)} input tickers were "
            f"dropped due to NaN values in the {lookback}-day return window. "
            "Pass cleaned prices, see src.data.ingest.load_or_build_clean_prices."
        )

    sigma = predicted_vols.reindex(tickers).values  # shape (n,)

    # Step 1: correlation matrix, use override if provided, else Ledoit-Wolf
    if override_corr is not None:
        corr = override_corr
    else:
        lw = LedoitWolf().fit(ret_window.values)
        cov_hist = lw.covariance_
        std_hist = np.sqrt(np.diag(cov_hist))
        corr = cov_hist / np.outer(std_hist, std_hist)
        np.fill_diagonal(corr, 1.0)  # correct floating-point drift on diagonal

    # Step 2: Scale correlation by RF-predicted volatilities
    # Sigma = D @ C @ D  where D = diag(sigma)
    D = np.diag(sigma)
    cov_matrix = D @ corr @ D

    # Assertion: diagonal of Sigma must equal sigma^2
    assert np.allclose(np.diag(cov_matrix), sigma ** 2, atol=1e-8), (
        "Covariance diagonal does not match predicted sigma^2. "
        "Check D @ C @ D construction."
    )

    # Defensive backstop: nearest-PSD projection if D @ C @ D introduced
    # small negative eigenvalues through floating-point accumulation
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if eigvals.min() < 0:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.clip(eigvals, 0, None)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov_matrix, corr, tickers


def optimise_portfolio(
    predicted_returns: pd.Series,
    predicted_vols: pd.Series,
    historical_returns: pd.DataFrame,
    risk_aversion: float = 2.0,
    shrinkage_alpha: float = 0.75,
    max_weight: float = 0.10,
    lookback: int = 252,
    override_corr: np.ndarray = None,
) -> dict:
    """
    Mean-variance optimisation via SLSQP.

    Objective (minimise): (risk_aversion / 2) * w'Σw - w'μ_shrunk

    Parameters
    ----------
    predicted_returns : pd.Series
        RF forward return predictions. Index: ticker.
    predicted_vols : pd.Series
        RF forward volatility predictions (sigma, annualised). Index: ticker.
    historical_returns : pd.DataFrame
        Daily returns for correlation estimation. Columns: tickers.
    risk_aversion : float
        Lambda, penalty on portfolio variance. Higher = more conservative.
    shrinkage_alpha : float
        Return shrinkage intensity in [0, 1].
        0 = raw RF predictions (maximum signal exploitation, maximum noise).
        1 = all assets get equal return = cross-sectional mean (pure
            minimum-variance; return model contributes nothing).
        Default 0.75, heavy shrinkage given near-zero return model skill.
    max_weight : float
        Maximum weight per asset. Guards against estimation-error concentration.
    lookback : int
        Trading days of history for correlation estimation.

    Returns
    -------
    dict with keys:
        weights            : pd.Series, final portfolio weights (indexed by ticker)
        expected_return    : float, w'μ_shrunk
        expected_vol       : float, sqrt(w'Σw)
        mu                 : pd.Series, shrunk return vector (for decomposition)
        sigma_diag         : pd.Series, RF predicted vols (for decomposition)
        cov_matrix         : np.ndarray, full covariance matrix (for decomposition)
        correlation_matrix : pd.DataFrame, Ledoit-Wolf correlation matrix,
                             indexed and columned by ticker (for decomposition)
        converged          : bool, False triggers fallback to equal weights
        message            : str, solver message or fallback reason
    """
    # Align tickers across both prediction series
    tickers = predicted_returns.index.intersection(
        predicted_vols.index).tolist()
    mu_rf = predicted_returns.reindex(tickers)
    sigma = predicted_vols.reindex(tickers)

    # Validate dimensions
    if len(tickers) == 0:
        raise ValueError(
            "No common tickers between return and volatility predictions.")

    # Build covariance matrix, may reorder tickers to match return window
    cov_matrix, corr, tickers = build_covariance_matrix(
        sigma, historical_returns, lookback, override_corr=override_corr)
    mu_rf = mu_rf.reindex(tickers)
    sigma = sigma.reindex(tickers)
    n = len(tickers)

    # Apply shrinkage: blend RF predictions toward cross-sectional mean
    mu_bar = mu_rf.mean()
    mu_shrunk = (1 - shrinkage_alpha) * mu_rf + shrinkage_alpha * mu_bar

    # SLSQP setup
    def objective(w):
        """Mean-variance objective: (lambda/2) * w' Sigma w - w' mu_shrunk."""
        port_var = w @ cov_matrix @ w
        port_return = w @ mu_shrunk.values
        return (risk_aversion / 2) * port_var - port_return

    def objective_grad(w):
        """Analytical gradient of the objective wrt w."""
        return risk_aversion * cov_matrix @ w - mu_shrunk.values

    w0 = np.ones(n) / n  # equal weights, always feasible
    # Dynamic cap: ensures feasibility when n < 1/max_weight
    effective_max = max(max_weight, 1.0 / n)
    bounds = [(0.0, effective_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    result = minimize(
        objective,
        w0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if result.success:
        weights = pd.Series(result.x, index=tickers)
        converged = True
        message = result.message
    else:
        # Fallback to equal weights, logged via returned message
        weights = pd.Series(w0, index=tickers)
        converged = False
        message = f"SLSQP failed: {result.message}. Falling back to equal weights."

    w = weights.values
    return {
        "weights":            weights,
        "expected_return":    float(w @ mu_shrunk.values),
        "expected_vol":       float(np.sqrt(w @ cov_matrix @ w)),
        "mu":                 mu_shrunk,
        "sigma_diag":         sigma,
        "cov_matrix":         cov_matrix,
        "correlation_matrix": pd.DataFrame(corr, index=tickers, columns=tickers),
        "converged":          converged,
        "message":            message,
    }
