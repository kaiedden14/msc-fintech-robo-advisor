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


def build_universe_metadata(cache_path: Path) -> pd.DataFrame:
    """
    Scrape FTSE 100 company name + sector from Wikipedia and cache to disk.

    Mirrors the table-index choice in build_universe(). Returns a DataFrame
    indexed by ticker (in yfinance format, e.g. 'HSBA.L'), with columns
    'name' and 'sector'. The dashboard uses this for the asset-selection
    table's Company and Sector columns. Run once; the dashboard never
    re-scrapes at runtime.

    Parameters
    ----------
    cache_path : Path
        Destination parquet (e.g. data/processed/universe_metadata.parquet).

    Returns
    -------
    pd.DataFrame
        Indexed by ticker, with columns ['name', 'sector'].
    """
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    ftse_table = tables[6]  # same index used by build_universe()

    # Identify columns defensively — Wikipedia column names drift over time
    rename = {}
    for col in ftse_table.columns:
        c = str(col).lower()
        if "ticker" in c and "ticker" not in rename.values():
            rename[col] = "ticker"
        elif "company" in c:
            rename[col] = "name"
        elif "industry" in c or "sector" in c or "fics" in c:
            rename[col] = "sector"

    required = {"ticker", "name", "sector"}
    missing = required - set(rename.values())
    if missing:
        raise ValueError(
            f"Wikipedia FTSE 100 table missing expected columns {missing}. "
            f"Saw columns: {list(ftse_table.columns)}"
        )

    meta = ftse_table.rename(columns=rename)[["ticker", "name", "sector"]].copy()
    meta["ticker"] = meta["ticker"].astype(str) + ".L"
    meta["ticker"] = meta["ticker"].replace("BT.A.L", "BT-A.L")
    meta = meta.drop_duplicates(subset="ticker").set_index("ticker")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(cache_path)
    return meta


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


def load_or_build_clean_prices(
    raw_path: Path,
    clean_path: Path,
    min_history: int = 1260,
) -> pd.DataFrame:
    """
    Return cleaned price data, building and caching it from raw on first call.

    Mirrors the download_prices caching pattern. On cache miss, reads the raw
    parquet, applies clean_prices(), and writes the result to clean_path.
    The cached file contains the MultiIndex (Close, Volume) DataFrame with
    failing tickers dropped and remaining NaNs forward/back-filled.

    Parameters
    ----------
    raw_path : Path
        Path to the raw prices parquet (output of download_prices()).
    clean_path : Path
        Destination cache for the cleaned DataFrame.
    min_history : int
        Forwarded to clean_prices().

    Returns
    -------
    pd.DataFrame
        Cleaned MultiIndex (Close, Volume) DataFrame.
    """
    if clean_path.exists():
        return pd.read_parquet(clean_path)

    raw = pd.read_parquet(raw_path)
    clean_df, _ = clean_prices(raw, min_history=min_history)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(clean_path)
    return clean_df


def refresh_prices(
    raw_path: Path,
    clean_path: Path,
    min_history: int = 1260,
) -> dict:
    """
    Incrementally update the cached prices from yfinance.

    Reads the existing raw parquet, downloads only the trading days since
    its max date, appends, writes back to raw_path, and rebuilds the clean
    cache at clean_path. Raises if no raw cache exists or if yfinance fails.

    Parameters
    ----------
    raw_path : Path
        Path to the existing raw prices parquet (output of download_prices()).
    clean_path : Path
        Path to the cleaned prices parquet. Rebuilt from updated raw.
    min_history : int
        Forwarded to clean_prices() when rebuilding.

    Returns
    -------
    dict with keys:
        previous_max_date : datetime.date — newest row before refresh
        new_max_date      : datetime.date — newest row after refresh
        new_rows          : int           — number of trading days appended
        no_new_data       : bool          — True if yfinance returned nothing new
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No cached prices at {raw_path}. "
            "Run download_prices() to build the initial cache first."
        )

    existing = pd.read_parquet(raw_path)
    previous_max_date = existing.index.max()
    tickers = existing["Close"].columns.tolist()

    start = (previous_max_date + pd.Timedelta(days=1)).date()
    new_data = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
    )

    if new_data.empty:
        return {
            "previous_max_date": previous_max_date.date(),
            "new_max_date":      previous_max_date.date(),
            "new_rows":          0,
            "no_new_data":       True,
        }

    new_data = new_data[["Close", "Volume"]]

    combined = pd.concat([existing, new_data]).sort_index()
    # De-dup in case yfinance returns an overlapping row
    combined = combined[~combined.index.duplicated(keep="last")]

    combined.to_parquet(raw_path)

    # Rebuild clean cache from updated raw
    clean_df, _ = clean_prices(combined, min_history=min_history)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(clean_path)

    return {
        "previous_max_date": previous_max_date.date(),
        "new_max_date":      combined.index.max().date(),
        "new_rows":          int((combined.index > previous_max_date).sum()),
        "no_new_data":       False,
    }


def refresh_features(
    clean_prices_path: Path,
    features_path: Path,
) -> dict:
    """
    Rebuild the feature panel from the latest clean prices and overwrite.

    Loads cleaned prices, runs build_features() (7 model features + 2 forward
    targets), drops NaN rows, and writes the result to features_path. Called
    by the dashboard's "Refresh data" handler after refresh_prices().

    Parameters
    ----------
    clean_prices_path : Path
        Path to the cleaned prices parquet (built by load_or_build_clean_prices
        or refresh_prices).
    features_path : Path
        Destination parquet for the feature panel (overwrites existing).

    Returns
    -------
    dict with keys:
        max_date  : datetime.date — newest feature row after rebuild
        n_rows    : int           — total feature rows after dropna
        n_tickers : int           — unique tickers in the rebuilt panel
    """
    # Local import to avoid pulling sklearn / scipy into ingest's import graph
    from src.features.engineer import build_features

    clean_df = pd.read_parquet(clean_prices_path)
    feature_df = build_features(clean_df)
    feature_clean = feature_df.dropna()

    features_path.parent.mkdir(parents=True, exist_ok=True)
    feature_clean.to_parquet(features_path)

    return {
        "max_date": feature_clean.index.get_level_values("date").max().date(),
        "n_rows": int(len(feature_clean)),
        "n_tickers": int(feature_clean.index.get_level_values("ticker").nunique()),
    }


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

    # --- Forward-fill only (carries last known price over halts/holidays). ---
    # Backward-fill is deliberately NOT applied: it would fill leading NaN with
    # a future price, which is look-ahead bias. Late-listing tickers that pass
    # the min_history filter (e.g. post-IPO listings) retain leading NaN here,
    # which feature engineering then propagates through rolling windows so that
    # contaminated rows are dropped naturally rather than silently bfilled.
    close = close.ffill()
    volume = volume.ffill()

    # Log any tickers with residual NaN after ffill (always leading NaN, by
    # construction). These are late listings; their leading rows will become
    # NaN in features.parquet and get dropped during the training-set assembly.
    residual_nan = close.isna().sum()
    report["leading_nan_after_ffill"] = residual_nan
    affected = residual_nan[residual_nan > 0]
    if len(affected) > 0:
        import warnings
        affected_summary = ", ".join(
            f"{t}({n})" for t, n in affected.sort_values(ascending=False).items()
        )
        warnings.warn(
            f"clean_prices: {len(affected)} ticker(s) have leading NaN after ffill "
            f"(post-listing tickers): {affected_summary}. These cells are NOT bfilled "
            f"to avoid look-ahead bias; downstream feature engineering will propagate "
            f"NaN through rolling windows and drop the affected rows.",
            stacklevel=2,
        )

    # --- Reassemble MultiIndex DataFrame ---
    clean_df = pd.concat({"Close": close, "Volume": volume}, axis=1)

    return clean_df, report
