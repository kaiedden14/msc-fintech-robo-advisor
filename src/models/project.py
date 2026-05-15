import numpy as np
import pandas as pd


def project_forward(
    monthly_return: float,
    annual_vol: float,
    horizon_months: int = 12,
    n_paths: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo forward projection of a portfolio's cumulative value over a
    fixed horizon under an iid Gaussian return assumption.

    Each path draws horizon_months returns from N(monthly_return, monthly_vol^2)
    where monthly_vol = annual_vol / sqrt(12). The portfolio value starts at 1.0
    and grows by (1 + r_t) each month.

    The Gaussian assumption understates tail risk for equity portfolios
    (Cont, 2001 — fat tails). This is acknowledged in the methodology.

    Parameters
    ----------
    monthly_return : float
        Expected monthly portfolio return (e.g. optimise_portfolio's
        expected_return, which is already on a monthly scale).
    annual_vol : float
        Annualised portfolio volatility (e.g. optimise_portfolio's
        expected_vol).
    horizon_months : int
        Projection horizon in months.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        paths             : np.ndarray (n_paths, horizon_months+1)
                            cumulative portfolio value per path per month,
                            starting at 1.0 at month 0
        percentile_bands  : pd.DataFrame
                            Index = month (0..horizon_months),
                            columns = ['p5','p25','p50','p75','p95']
        terminal          : dict — p5/p25/p50/p75/p95 of cumulative value
                            at the final month
        params            : dict — echo of monthly_return, monthly_vol used
    """
    rng = np.random.default_rng(seed)
    monthly_vol = annual_vol / np.sqrt(12)

    returns = rng.normal(monthly_return, monthly_vol, size=(n_paths, horizon_months))
    cum = np.cumprod(1 + returns, axis=1)
    # prepend month-0 value of 1.0
    paths = np.concatenate([np.ones((n_paths, 1)), cum], axis=1)

    pct_levels = [5, 25, 50, 75, 95]
    band_data = {f"p{p}": np.percentile(paths, p, axis=0) for p in pct_levels}
    percentile_bands = pd.DataFrame(band_data, index=range(horizon_months + 1))
    percentile_bands.index.name = "month"

    terminal = {f"p{p}": float(np.percentile(paths[:, -1], p)) for p in pct_levels}

    return {
        "paths": paths,
        "percentile_bands": percentile_bands,
        "terminal": terminal,
        "params": {
            "monthly_return": monthly_return,
            "monthly_vol": monthly_vol,
            "annual_vol": annual_vol,
        },
    }


def benchmark_params_from_history(
    benchmark_close: pd.Series,
    lookback_years: int = 5,
) -> dict:
    """
    Derive Gaussian projection parameters (monthly mean + annualised vol) for
    a benchmark series from its trailing monthly returns.

    Parameters
    ----------
    benchmark_close : pd.Series
        Daily close prices indexed by date (e.g. clean_close['^FTSE']).
    lookback_years : int
        Trailing window for parameter estimation.

    Returns
    -------
    dict with keys monthly_return, annual_vol, n_months, window_start, window_end.
    """
    # Resample to month-end, compute monthly returns
    monthly_close = benchmark_close.resample("ME").last().dropna()
    monthly_returns = monthly_close.pct_change().dropna()

    window = monthly_returns.tail(lookback_years * 12)
    monthly_return = float(window.mean())
    monthly_vol = float(window.std())
    annual_vol = monthly_vol * np.sqrt(12)

    return {
        "monthly_return": monthly_return,
        "annual_vol": annual_vol,
        "n_months": len(window),
        "window_start": window.index[0],
        "window_end": window.index[-1],
    }
