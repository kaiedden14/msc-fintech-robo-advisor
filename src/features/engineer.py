import pandas as pd
import numpy as np


def compute_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """Compute 12m-minus-1m momentum"""
    return_12m = close.pct_change(252)
    return_1m = close.pct_change(21)
    return return_12m - return_1m


def compute_short_term_return(close: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-month return (short-term reversal signal)."""
    return close.pct_change(21)


def compute_volatility(close: pd.DataFrame) -> pd.DataFrame:
    """Compute 21-day realised volatility, annualised."""
    daily_returns = close.pct_change(fill_method=None)
    return daily_returns.rolling(21).std() * (252 ** 0.5)


def compute_drawdown(close: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown from 52-week (252-day) rolling high."""
    rolling_high = close.rolling(252).max()
    return (close - rolling_high) / rolling_high


def compute_ma_ratio(close: pd.DataFrame) -> pd.DataFrame:
    """Compute price / 200-day moving average ratio."""
    ma_200 = close.rolling(200).mean()
    return close / ma_200


def compute_rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute RSI(14) using Wilder's smoothing method."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: exponential moving average with alpha = 1/window
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


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

    # Rolling covariance of every stock vs benchmark simultaneously
    rolling_cov = returns.rolling(window).cov(bench_returns)
    # Rolling variance of benchmark
    rolling_var = bench_returns.rolling(window).var()

    # Divide each stock's covariance by benchmark variance
    return rolling_cov.div(rolling_var, axis=0)


def compute_vix(vix_series: pd.Series) -> pd.Series:
    """Return VIX level — broadcast to all tickers in build_features()."""
    return vix_series


def compute_relative_strength(close: pd.DataFrame,
                              benchmark: pd.Series) -> pd.DataFrame:
    """Compute 21-day stock return minus FTSE 100 benchmark return."""
    stock_return = close.pct_change(21)
    benchmark_return = benchmark.pct_change(21)
    return stock_return.sub(benchmark_return, axis=0)


def compute_52w_percentile(close: pd.DataFrame) -> pd.DataFrame:
    """Compute price position within 52-week (252-day) high-low range."""
    rolling_high = close.rolling(252).max()
    rolling_low = close.rolling(252).min()
    return (close - rolling_low) / (rolling_high - rolling_low)


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 9 features for every ticker and return a long-format DataFrame.

    Returns
    -------
    pd.DataFrame
        Index: (date, ticker). Columns: 9 features + forward_return target.
    """
    close = clean_df["Close"].drop(columns=["^FTSE", "^VIX"])
    volume = clean_df["Volume"].drop(columns=["^FTSE", "^VIX"])
    benchmark = clean_df["Close"]["^FTSE"]
    vix = compute_vix(clean_df["Close"]["^VIX"])

    # Compute all features in wide format (rows=dates, columns=tickers)
    wide_features = {
        "momentum":        compute_momentum(close),
        "return_1m":       compute_short_term_return(close),
        "volatility_21d":  compute_volatility(close),
        "drawdown_52w":    compute_drawdown(close),
        "ma_ratio_200":    compute_ma_ratio(close),
        "rsi_14":          compute_rsi(close),
        "volume_ratio_20": compute_volume_ratio(volume),
        "beta_252":        compute_beta(close, benchmark),
        "relative_strength":  compute_relative_strength(close, benchmark),
        "percentile_52w":     compute_52w_percentile(close),

    }

    # Stack each wide DataFrame to long format and concatenate
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

    # Compute forward 1-month return as the target variable
    daily_returns = close.pct_change(fill_method=None)
    forward_returns = close.pct_change(21).shift(-21)
    forward_volatility = daily_returns.rolling(
        21).std().shift(-21) * (252 ** 0.5)

    target_return = forward_returns.stack(future_stack=True)
    target_return.index.names = ["date", "ticker"]
    feature_df["forward_return"] = target_return

    target_vol = forward_volatility.stack(future_stack=True)
    target_vol.index.names = ["date", "ticker"]
    feature_df["forward_volatility"] = target_vol

    return feature_df
