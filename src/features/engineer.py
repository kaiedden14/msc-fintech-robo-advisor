import pandas as pd
import numpy as np


def compute_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """Compute 12m-minus-1m momentum (Jegadeesh and Titman, 1993)."""
    return_12m = close.pct_change(252)
    return_1m = close.pct_change(21)
    return return_12m - return_1m


def compute_volatility(close: pd.DataFrame) -> pd.DataFrame:
    """Compute 21-day realised volatility, annualised."""
    daily_returns = close.pct_change(fill_method=None)
    return daily_returns.rolling(21).std() * (252 ** 0.5)


def compute_drawdown(close: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown from 52-week (252-day) rolling high. Always <= 0."""
    rolling_high = close.rolling(252).max()
    return (close - rolling_high) / rolling_high


def compute_relative_strength(close: pd.DataFrame,
                              benchmark: pd.Series) -> pd.DataFrame:
    """Compute 21-day stock return minus FTSE 100 benchmark return."""
    stock_return = close.pct_change(21)
    benchmark_return = benchmark.pct_change(21)
    return stock_return.sub(benchmark_return, axis=0)


def compute_volume_ratio(volume: pd.DataFrame) -> pd.DataFrame:
    """Compute volume ratio: today's volume / 20-day average volume."""
    avg_volume = volume.rolling(20).mean()
    return volume / avg_volume


def compute_beta(close: pd.DataFrame, benchmark: pd.Series,
                 window: int = 252) -> pd.DataFrame:
    """
    Compute rolling beta to FTSE 100 benchmark using vectorised covariance.
    Beta = Cov(stock, benchmark) / Var(benchmark) over a 252-day window.
    """
    returns = close.pct_change(fill_method=None)
    bench_returns = benchmark.pct_change(fill_method=None)

    rolling_cov = returns.rolling(window).cov(bench_returns)
    rolling_var = bench_returns.rolling(window).var()

    return rolling_cov.div(rolling_var, axis=0)


def compute_vix(vix_series: pd.Series) -> pd.Series:
    """Return VIX level — broadcast to all tickers in build_features()."""
    return vix_series


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 7 features for every ticker and return a long-format DataFrame.

    Features retained after Spearman correlation audit (dropped pairs with
    median |corr| > 0.7 across tickers):
      - return_1m dropped (0.856 with relative_strength)
      - rsi_14 dropped (0.744 with relative_strength)
      - ma_ratio_200 dropped (0.874 with drawdown_52w)
      - percentile_52w dropped (0.947 with drawdown_52w)

    Returns
    -------
    pd.DataFrame
        Index: (date, ticker). Columns: 7 features + 2 targets.
    """
    close = clean_df["Close"].drop(columns=["^FTSE", "^VIX"])
    volume = clean_df["Volume"].drop(columns=["^FTSE", "^VIX"])
    benchmark = clean_df["Close"]["^FTSE"]
    vix = compute_vix(clean_df["Close"]["^VIX"])

    wide_features = {
        "momentum":          compute_momentum(close),
        "volatility_21d":    compute_volatility(close),
        "drawdown_52w":      compute_drawdown(close),
        "relative_strength": compute_relative_strength(close, benchmark),
        "volume_ratio_20":   compute_volume_ratio(volume),
        "beta_252":          compute_beta(close, benchmark),
    }

    long_frames = []
    for name, df in wide_features.items():
        s = df.stack(future_stack=True)
        s.name = name
        long_frames.append(s)

    feature_df = pd.concat(long_frames, axis=1)
    feature_df.index.names = ["date", "ticker"]

    # Broadcast VIX across all tickers (same value per date)
    feature_df["vix"] = (
        vix.reindex(feature_df.index.get_level_values("date")).values
    )

    # Forward targets (63 trading days ≈ 1 quarter)
    daily_returns = close.pct_change(fill_method=None)
    forward_returns = close.pct_change(63).shift(-63)
    forward_volatility = daily_returns.rolling(
        63).std().shift(-63) * (252 ** 0.5)

    target_return = forward_returns.stack(future_stack=True)
    target_return.index.names = ["date", "ticker"]
    feature_df["forward_return"] = target_return

    target_vol = forward_volatility.stack(future_stack=True)
    target_vol.index.names = ["date", "ticker"]
    feature_df["forward_volatility"] = target_vol

    return feature_df
