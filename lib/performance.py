"""Realised portfolio performance helpers.

Computes how a fixed-weight portfolio has actually performed from a given
start date through to the most recent price data, alongside the FTSE 100
benchmark over the same window. Used by the Rebalancing page to surface
real-world post-decision behaviour of the recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_realised_performance(
    weights: dict[str, float],
    prices_close: pd.DataFrame,
    start_date: pd.Timestamp,
    benchmark_col: str = "^FTSE",
) -> dict:
    """Compute realised portfolio performance from start_date onwards.

    Parameters
    ----------
    weights : dict[str, float]
        Final ticker → weight (must sum to ~1.0).
    prices_close : pd.DataFrame
        Close prices wide-format. Must contain every ticker in `weights`
        plus the benchmark column.
    start_date : pd.Timestamp
        Inclusive lower bound — typically the snapshot_date that the
        optimiser ran on.
    benchmark_col : str
        Column name of the benchmark price series.

    Returns
    -------
    dict with keys:
        start_date            : pd.Timestamp — actual first trading day used
        end_date              : pd.Timestamp — latest available trading day
        n_days                : int          — trading days in the window
        portfolio_value_path  : pd.Series    — £1 invested, indexed by date
        benchmark_value_path  : pd.Series    — £1 invested in FTSE, same dates
        portfolio_return      : float        — terminal cumulative return
        benchmark_return      : float        — terminal cumulative return
        portfolio_ann_vol     : float        — realised annualised vol
        benchmark_ann_vol     : float        — realised annualised vol
        max_drawdown          : float        — worst peak-to-trough on the portfolio
        per_stock             : pd.DataFrame — ticker × [weight, stock_return,
                                                contribution]
    """
    tickers = list(weights.keys())

    # Slice price panel to the relevant tickers + benchmark, from start_date on
    cols_needed = tickers + [benchmark_col]
    missing = [c for c in cols_needed if c not in prices_close.columns]
    if missing:
        raise KeyError(f"Prices missing for: {missing}")

    sub = prices_close[cols_needed].loc[start_date:].dropna(how="all")
    if len(sub) < 2:
        return {
            "start_date":     start_date,
            "end_date":       start_date,
            "n_days":         0,
            "insufficient":   True,
        }

    actual_start = sub.index[0]
    actual_end = sub.index[-1]

    # Daily simple returns, dropping the first NaN row from pct_change
    daily_returns = sub.pct_change(fill_method=None).dropna(how="all")

    # Portfolio daily return = sum(weight_i * daily_return_i)
    weight_vec = pd.Series(weights).reindex(tickers).fillna(0.0)
    portfolio_daily = (daily_returns[tickers] * weight_vec).sum(axis=1)
    portfolio_value = (1 + portfolio_daily).cumprod()
    # Prepend the start-day value of 1.0
    portfolio_value = pd.concat([
        pd.Series([1.0], index=[actual_start]),
        portfolio_value,
    ])

    # Benchmark
    benchmark_daily = daily_returns[benchmark_col]
    benchmark_value = (1 + benchmark_daily).cumprod()
    benchmark_value = pd.concat([
        pd.Series([1.0], index=[actual_start]),
        benchmark_value,
    ])

    # Per-stock attribution
    start_prices = sub[tickers].iloc[0]
    end_prices = sub[tickers].iloc[-1]
    stock_returns = end_prices / start_prices - 1
    contributions = weight_vec * stock_returns
    per_stock = pd.DataFrame({
        "weight":       weight_vec,
        "stock_return": stock_returns,
        "contribution": contributions,
    }).sort_values("contribution", ascending=False)

    # Realised vol (annualised)
    portfolio_ann_vol = float(portfolio_daily.std() * np.sqrt(252))
    benchmark_ann_vol = float(benchmark_daily.std() * np.sqrt(252))

    # Max drawdown on the portfolio path
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value / running_max - 1).min()

    return {
        "start_date":            actual_start,
        "end_date":              actual_end,
        "n_days":                len(portfolio_value),
        "portfolio_value_path":  portfolio_value,
        "benchmark_value_path":  benchmark_value,
        "portfolio_return":      float(portfolio_value.iloc[-1] - 1),
        "benchmark_return":      float(benchmark_value.iloc[-1] - 1),
        "portfolio_ann_vol":     portfolio_ann_vol,
        "benchmark_ann_vol":     benchmark_ann_vol,
        "max_drawdown":          float(drawdown),
        "per_stock":             per_stock,
        "insufficient":          False,
    }
