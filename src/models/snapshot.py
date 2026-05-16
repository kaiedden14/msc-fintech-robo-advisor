import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.models.explain import compute_shap_values


def build_snapshot(
    return_model_path: Path,
    vol_model_path: Path,
    features_path: Path,
    output_dir: Path,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Build a frozen recommendation snapshot for the user-study window.

    Pre-computes per-ticker predictions and SHAP attributions on a single
    cross-section so the dashboard reads from disk rather than recomputing
    SHAP on every page load. Required for replicable per-participant outputs
    and acceptable dashboard latency (SHAP on ~95 tickers takes ~1-3s live).

    Writes to output_dir:
        snapshot_predictions.parquet      ticker x [predicted_return, predicted_vol]
        snapshot_shap_return.parquet      (date, ticker) x SHAP cols + features
        snapshot_shap_volatility.parquet  same structure
        snapshot_metadata.json            build_date, snapshot_date, n_tickers, model paths

    Parameters
    ----------
    return_model_path : Path
        Fitted return RF (rf_return_v4.joblib).
    vol_model_path : Path
        Fitted volatility RF (rf_volatility_v4.joblib).
    features_path : Path
        Features parquet with (date, ticker) MultiIndex.
    output_dir : Path
        Destination directory for snapshot files.
    snapshot_date : pd.Timestamp, optional
        Cross-section date. Defaults to latest date in features.

    Returns
    -------
    dict
        Metadata also written to snapshot_metadata.json.
    """
    rf_return = joblib.load(return_model_path)
    rf_vol = joblib.load(vol_model_path)

    feature_cols = list(rf_return.feature_names_in_)
    if list(rf_vol.feature_names_in_) != feature_cols:
        raise ValueError(
            "Return and volatility models were trained on different feature sets. "
            f"Return: {feature_cols}. Vol: {list(rf_vol.feature_names_in_)}."
        )

    features = pd.read_parquet(features_path)

    if snapshot_date is None:
        snapshot_date = features.index.get_level_values("date").max()
    snapshot_date = pd.Timestamp(snapshot_date)

    cross = features.xs(snapshot_date, level="date")
    X = cross[feature_cols].dropna()
    n_tickers_before = len(cross)
    n_tickers = len(X)
    n_dropped = n_tickers_before - n_tickers

    if n_tickers == 0:
        raise ValueError(
            f"No tickers have complete features at snapshot_date={snapshot_date.date()}."
        )

    # Predictions
    predictions = pd.DataFrame(
        {
            "predicted_return": rf_return.predict(X),
            "predicted_vol": rf_vol.predict(X),
        },
        index=X.index,
    )
    predictions.index.name = "ticker"

    # SHAP needs a (date, ticker) MultiIndex
    feat_for_shap = X.copy()
    feat_for_shap.index = pd.MultiIndex.from_product(
        [[snapshot_date], X.index], names=["date", "ticker"]
    )
    shap_return_df, dropped_return = compute_shap_values(
        rf_return, feat_for_shap, feature_cols
    )
    shap_vol_df, dropped_vol = compute_shap_values(
        rf_vol, feat_for_shap, feature_cols
    )

    if dropped_return or dropped_vol:
        raise RuntimeError(
            f"Unexpected NaN drops during SHAP: return={dropped_return}, vol={dropped_vol}. "
            "Feature matrix was pre-cleaned; this should not happen."
        )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "snapshot_predictions.parquet")
    shap_return_df.to_parquet(output_dir / "snapshot_shap_return.parquet")
    shap_vol_df.to_parquet(output_dir / "snapshot_shap_volatility.parquet")

    metadata = {
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "built_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "n_tickers": int(n_tickers),
        "n_tickers_dropped_for_nan": int(n_dropped),
        "feature_cols": feature_cols,
        "return_model": str(return_model_path),
        "vol_model": str(vol_model_path),
    }
    (output_dir / "snapshot_metadata.json").write_text(json.dumps(metadata, indent=2))

    return metadata
