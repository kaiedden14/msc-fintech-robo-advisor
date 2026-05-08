import pandas as pd
import yfinance as yf
from pathlib import Path
import requests
from io import StringIO


def build_universe() -> list[str]:
    """
    build a list of FTSE 100 tickers, scraping 
    from the Wikipedia page. ".L" is appended to each
    ticker to comply with yfinance's format 
    for London Stock Exchange tickers.

    Returns
    -------
    list[str]
        A list of FTSE 100 tickers in the format required by yfinance.
    """

    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))

    # index identified empirically; verify if scrape breaks
    ftse_table = tables[6]
    tickers = ftse_table["Ticker"].tolist()
    tickers = [ticker + ".L" for ticker in tickers]
    tickers = ["BT-A.L" if t == "BT.A.L" else t for t in tickers]

    tickers.extend(["^FTSE", "^VIX"])
    return tickers


def download_prices(tickers: list[str], cache_path: Path) -> pd.DataFrame:
    """
    Download daily adjusted close prices and volume for all tickers
    from yfinance. Results are cached to disk as a parquet file so
    yfinance is only called once.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols returned by build_universe().
    cache_path : Path
        File path for the cached parquet file.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with columns (Close, Volume) per ticker.
    """

    # If cached file exists, load and return it immediately
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    # Otherwise download from yfinance
    print(f"Downloading data for {len(tickers)} tickers...")
    raw = yf.download(
        tickers=tickers,
        start="2015-01-01",
        auto_adjust=True,   # Close column becomes adjusted close
        progress=True,      # shows a progress bar
    )

    # Keep only Close and Volume columns
    data = raw[["Close", "Volume"]]

    # Save to disk so future runs skip the download
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_path)
    print(f"Saved to {cache_path}")

    return data


def clean_prices(df: pd.DataFrame, min_history: int = 1260) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply quality filters to the raw price DataFrame. Drops tickers
    with insufficient history or excessive missingness, replaces
    isolated NaNs via forward-fill, and flags suspicious zero prices.

    Parameters
    ----------
    df : pd.DataFrame
        Raw MultiIndex DataFrame from download_prices().
    min_history : int
        Minimum number of non-NaN rows required per ticker (default 1260 = ~5 years).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        clean_df : cleaned Close+Volume DataFrame
        report_df : quality report showing stats per ticker
    """

    close = df["Close"]
    volume = df["Volume"]

    # --- Build quality report before any dropping ---
    report = pd.DataFrame(index=close.columns)
    report["total_rows"] = len(close)
    report["non_nan_rows"] = close.notna().sum()
    report["pct_missing"] = (close.isna().mean() * 100).round(2)
    report["zero_prices"] = (close == 0).sum()
    report["zero_volume_days"] = (volume == 0).sum()

    # Daily returns for outlier detection
    daily_returns = close.pct_change(fill_method=None)
    report["extreme_returns"] = (
        daily_returns.abs() > 0.5).sum()  # >±50% in a day

    # --- Flag tickers that fail quality thresholds ---
    report["drop_reason"] = ""
    insufficient = report["non_nan_rows"] < min_history
    report.loc[insufficient, "drop_reason"] = "insufficient history"

    excessive_missing = report["pct_missing"] > 20
    report.loc[excessive_missing & (
        report["drop_reason"] == ""), "drop_reason"] = "excessive missingness >20%"

    # --- Drop failing tickers ---
    to_drop = report[report["drop_reason"] != ""].index.tolist()
    if to_drop:
        print(f"Dropping {len(to_drop)} tickers: {to_drop}")
        close = close.drop(columns=to_drop)
        volume = volume.drop(columns=to_drop)

    # --- Forward-fill isolated NaNs (e.g. bank holidays) then back-fill any leading NaNs ---
    close = close.ffill().bfill()
    volume = volume.ffill().bfill()

    # --- Reassemble MultiIndex DataFrame ---
    clean_df = pd.concat({"Close": close, "Volume": volume}, axis=1)

    return clean_df, report
