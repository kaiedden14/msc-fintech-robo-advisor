"""SHAP attribution for the Random Forest predictions.

This module wraps ``shap.TreeExplainer`` with the choices the dissertation
locked in for the dashboard's per-stock explanation cards.

Two methodological choices are worth noting:

- ``feature_perturbation="tree_path_dependent"`` is used in preference to
  the ``interventional`` alternative. The pipeline runs on a rolling
  quarterly cadence with continuously drifting feature distributions, so
  maintaining a representative interventional background dataset is a
  recurring methodological burden. ``tree_path_dependent`` avoids it at
  the cost of a small attribution-noise floor between strongly correlated
  features, which the pre-committed correlation audit (median absolute
  rank correlation under 0.70) keeps well below the meaningful threshold.

- The additivity property (base value plus the sum of SHAP values equals
  the model's prediction) is asserted to within 1e-5 on every snapshot
  computation, so any silent drift in the explainer is caught at build
  time rather than in the dashboard.

The dashboard renders these attributions through the plain-language
phrasing templates in ``lib/copy.py`` and the bar-card pattern in the
detail panel on the Asset Selection and Optimised Portfolio pages.
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    model,
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Compute SHAP values for the volatility RF using TreeExplainer with
    tree_path_dependent perturbation. Returns an assembled DataFrame
    ready for storage and a log of dropped tickers.

    Parameters
    ----------
    model : fitted RandomForestRegressor
        The frozen volatility RF (rf_volatility_v4).
    feature_matrix : pd.DataFrame
        Feature matrix MultiIndexed on (date, ticker). Expected production
        input is one date's cross-section (~80 rows), but the function
        accepts any (date, ticker)-indexed matrix, the same function is
        used for development-time inspection over larger windows.
    feature_cols : list[str]
        Ordered feature column list matching training order. Passed
        explicitly to prevent silent column reordering bugs.

    Returns
    -------
    shap_df : pd.DataFrame
        MultiIndexed (date, ticker). Columns: shap_<feature> × 7,
        feature values × 7, base_value, prediction.
        Contains only rows with complete features.
    dropped_tickers : list[tuple[str, str]]
        (ticker, reason) for rows dropped due to NaN features.
        Absent tickers are out of scope, handled upstream at ingestion.
    """
    # Select and reorder columns explicitly, do not trust input ordering
    X = feature_matrix[feature_cols].copy()

    # Identify and drop NaN rows, log them
    nan_mask = X.isnull().any(axis=1)
    dropped_tickers = [
        (idx[1], "NaN in features")
        for idx in X[nan_mask].index
    ]
    X_clean = X[~nan_mask]

    if X_clean.empty:
        raise ValueError("No complete-feature rows remain after NaN removal.")

    # Instantiate explainer, tree_path_dependent requires no background dataset.
    # Chosen over interventional because this pipeline runs quarterly on drifting
    # live data; fixing a stable background dataset is a recurring methodological
    # burden that tree_path_dependent avoids. Limitation: moderate correlations
    # among risk-signal features (volatility_21d, beta_252, vix) will cause some
    # attribution noise between those features. This is acknowledged in the
    # methodology.
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )

    shap_values = explainer.shap_values(X_clean)   # shape: (n_rows, n_features)
    base_value  = float(np.asarray(explainer.expected_value).flat[0])  # scalar

    predictions = model.predict(X_clean)

    # Assemble output DataFrame
    shap_cols = {f"shap_{col}": shap_values[:, i]
                 for i, col in enumerate(feature_cols)}
    feat_cols = {col: X_clean[col].values for col in feature_cols}

    shap_df = pd.DataFrame(
        {**shap_cols, **feat_cols},
        index=X_clean.index,
    )
    shap_df["base_value"] = base_value
    shap_df["prediction"] = predictions

    # Additivity assertion: base + sum(shap) ≈ prediction for every row.
    # Tolerance of 1e-5 accounts for floating-point accumulation across
    # 500 tree paths without masking genuine failures.
    reconstructed = base_value + shap_values.sum(axis=1)
    assert np.allclose(reconstructed, predictions, atol=1e-5), (
        "SHAP additivity check failed: base_value + sum(shap_values) "
        "does not match model predictions within tolerance 1e-5."
    )

    return shap_df, dropped_tickers


def store_shap_values(shap_df: pd.DataFrame, path) -> None:
    """
    Write the assembled SHAP DataFrame to Parquet.

    Parameters
    ----------
    shap_df : pd.DataFrame
        Output of compute_shap_values, no reshaping required.
    path : str or Path
        Destination path for the Parquet file.
    """
    shap_df.to_parquet(path)
