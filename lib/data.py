"""Cached data loaders for the dashboard.

All loaders are @st.cache_data decorated so each Streamlit rerun reads
from in-memory cache rather than disk. Cache lives for the session.

The dashboard never loads RF models or calls SHAP — both are pre-computed
into the snapshot files by src/models/snapshot.py upstream.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


_DATA_DIR = Path("data/processed")


@st.cache_data
def load_universe_metadata() -> pd.DataFrame:
    """Ticker -> (name, sector). Built once by src.data.ingest.build_universe_metadata."""
    return pd.read_parquet(_DATA_DIR / "universe_metadata.parquet")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Per-ticker predicted_return and predicted_vol from the monthly snapshot.

    Index: ticker. Columns: predicted_return (monthly, decimal),
    predicted_vol (annualised, decimal).
    """
    return pd.read_parquet(_DATA_DIR / "snapshot_predictions.parquet")


@st.cache_data
def load_shap_return() -> pd.DataFrame:
    """SHAP attributions for the return model on the snapshot cross-section.

    MultiIndex (date, ticker). 16 columns: 7 shap_* + 7 feature values +
    base_value + prediction.
    """
    return pd.read_parquet(_DATA_DIR / "snapshot_shap_return.parquet")


@st.cache_data
def load_shap_volatility() -> pd.DataFrame:
    """SHAP attributions for the volatility model on the snapshot cross-section."""
    return pd.read_parquet(_DATA_DIR / "snapshot_shap_volatility.parquet")


@st.cache_data
def load_prices_clean() -> pd.DataFrame:
    """Cleaned MultiIndex (Close, Volume) price panel.

    Index: trading dates. Columns: MultiIndex (top: Close|Volume, second:
    ticker). Includes ^FTSE and ^VIX as additional tickers used for
    benchmarking but not as user-selectable assets.
    """
    return pd.read_parquet(_DATA_DIR / "prices_clean.parquet")


@st.cache_data
def load_snapshot_metadata() -> dict:
    """Snapshot metadata JSON (snapshot_date, build time, n_tickers, model paths)."""
    return json.loads((_DATA_DIR / "snapshot_metadata.json").read_text())


def invalidate_caches() -> None:
    """Clear all data-loader caches.

    Called by the "Refresh data" handler after the daily pipeline runs so the
    next read of any loader pulls fresh data from disk. Universe metadata is
    not invalidated because it changes only when build_universe_metadata is
    re-run (not part of the daily refresh).
    """
    load_predictions.clear()
    load_shap_return.clear()
    load_shap_volatility.clear()
    load_prices_clean.clear()
    load_snapshot_metadata.clear()
