import numpy as np
import pandas as pd


def project_forward(
    period_return: float,
    annual_vol: float,
    horizon_periods: int = 4,
    n_paths: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo forward projection of a portfolio's cumulative value over a
    fixed horizon under an iid Gaussian return assumption.

    Each path draws horizon_periods returns from N(period_return, period_vol^2)
    where period_vol = annual_vol / sqrt(4) and one period ≈ 63 trading days
    (one calendar quarter). The portfolio value starts at 1.0 and grows by
    (1 + r_t) each quarter.

    The Gaussian assumption understates tail risk for equity portfolios
    (Cont, 2001 — fat tails). This is acknowledged in the methodology.

    Parameters
    ----------
    period_return : float
        Expected per-period (quarterly) portfolio return — e.g.
        optimise_portfolio's expected_return, which under v6 is on a
        quarterly scale.
    annual_vol : float
        Annualised portfolio volatility (e.g. optimise_portfolio's
        expected_vol).
    horizon_periods : int
        Projection horizon in quarters (default 4 = one year).
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        paths             : np.ndarray (n_paths, horizon_periods+1)
                            cumulative portfolio value per path per quarter,
                            starting at 1.0 at quarter 0
        percentile_bands  : pd.DataFrame
                            Index = quarter (0..horizon_periods),
                            columns = ['p5','p25','p50','p75','p95']
        terminal          : dict — p5/p25/p50/p75/p95 of cumulative value
                            at the final quarter
        params            : dict — echo of period_return, period_vol used
    """
    rng = np.random.default_rng(seed)
    period_vol = annual_vol / np.sqrt(4)

    returns = rng.normal(period_return, period_vol, size=(n_paths, horizon_periods))
    cum = np.cumprod(1 + returns, axis=1)
    # prepend quarter-0 value of 1.0
    paths = np.concatenate([np.ones((n_paths, 1)), cum], axis=1)

    pct_levels = [5, 25, 50, 75, 95]
    band_data = {f"p{p}": np.percentile(paths, p, axis=0) for p in pct_levels}
    percentile_bands = pd.DataFrame(band_data, index=range(horizon_periods + 1))
    percentile_bands.index.name = "quarter"

    terminal = {f"p{p}": float(np.percentile(paths[:, -1], p)) for p in pct_levels}

    return {
        "paths": paths,
        "percentile_bands": percentile_bands,
        "terminal": terminal,
        "params": {
            "period_return": period_return,
            "period_vol": period_vol,
            "annual_vol": annual_vol,
        },
    }


def benchmark_params_from_history(
    benchmark_close: pd.Series,
    lookback_years: int = 10,
) -> dict:
    """
    Derive Gaussian projection parameters (per-quarter mean + annualised vol)
    for a benchmark series from its trailing quarterly returns.

    Parameters
    ----------
    benchmark_close : pd.Series
        Daily close prices indexed by date (e.g. clean_close['^FTSE']).
    lookback_years : int
        Trailing window for parameter estimation. Default 10 years
        (40 quarterly observations) — wider window than the monthly v5
        setup to keep the quarterly sample large enough to be stable.

    Returns
    -------
    dict with keys period_return, annual_vol, n_quarters, window_start, window_end.
    """
    # Resample to quarter-end, compute quarterly returns
    quarterly_close = benchmark_close.resample("QE").last().dropna()
    quarterly_returns = quarterly_close.pct_change().dropna()

    window = quarterly_returns.tail(lookback_years * 4)
    period_return = float(window.mean())
    period_vol = float(window.std())
    annual_vol = period_vol * np.sqrt(4)

    return {
        "period_return": period_return,
        "annual_vol": annual_vol,
        "n_quarters": len(window),
        "window_start": window.index[0],
        "window_end": window.index[-1],
    }
