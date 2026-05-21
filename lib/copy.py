"""Plain-language templating for SHAP attribution cards.

Translates raw SHAP values into retail-readable sentences. String
formatting only — no LLM calls. Templates are deterministic so the
methodology write-up can show the exact transformation rule.

The display labels and value phrasings were chosen for retail
explainability — e.g. 'Recent volatility' instead of 'volatility_21d',
'Market sensitivity' instead of 'beta_252'.
"""

from typing import Any

import pandas as pd


# One entry per model feature. The 'value_phrasing' is rendered with the
# current feature value substituted in; it goes into the parenthesis.
_FEATURE_COPY: dict[str, dict[str, str]] = {
    "momentum": {
        "label": "Recent momentum",
        "value_phrasing": "12-month minus 1-month return is {value:+.1%}",
    },
    "volatility_21d": {
        "label": "Recent volatility",
        "value_phrasing": "21-day realised volatility is {value:.1%} annualised",
    },
    "drawdown_52w": {
        "label": "Drawdown from 52-week high",
        "value_phrasing": "currently {value:.1%} below its 52-week high",
    },
    "relative_strength": {
        "label": "Performance vs FTSE 100",
        "value_phrasing": "21-day relative return is {value:+.1%}",
    },
    "volume_ratio_20": {
        "label": "Trading volume",
        "value_phrasing": "{value:.1f}× the 20-day average",
    },
    "beta_252": {
        "label": "Market sensitivity",
        "value_phrasing": "252-day beta to FTSE 100 is {value:.2f}",
    },
    "vix": {
        "label": "Market volatility regime",
        "value_phrasing": "VIX level is {value:.1f}",
    },
}


def shap_reasons(
    shap_row: pd.Series,
    top_n_positive: int = 3,
    top_n_negative: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Extract top-N positive and top-N negative SHAP contributions.

    Parameters
    ----------
    shap_row : pd.Series
        One row from a snapshot SHAP DataFrame — contains shap_* columns,
        feature value columns, base_value, and prediction.
    top_n_positive, top_n_negative : int
        Number of features to surface on each side.

    Returns
    -------
    dict with 'in_favour' and 'against' lists of dicts. Each dict has
    keys: label, value_desc, contribution_pp (absolute value, in
    percentage points).
    """
    pairs = []
    for col in shap_row.index:
        if not col.startswith("shap_"):
            continue
        feature = col[len("shap_"):]
        if feature not in _FEATURE_COPY:
            continue
        pairs.append((feature, shap_row[feature], float(shap_row[col])))

    positive = sorted([p for p in pairs if p[2] > 0], key=lambda x: -x[2])
    negative = sorted([p for p in pairs if p[2] < 0], key=lambda x: x[2])

    def _render(feature: str, feat_val: float, shap_val: float) -> dict[str, Any]:
        c = _FEATURE_COPY[feature]
        return {
            "label": c["label"],
            "value_desc": c["value_phrasing"].format(value=feat_val),
            "contribution_pp": abs(shap_val) * 100,
        }

    return {
        "in_favour": [_render(*p) for p in positive[:top_n_positive]],
        "against":   [_render(*p) for p in negative[:top_n_negative]],
    }


# ---------- Phase 5c: waterfall data + plain-language one-liners ----------


# Retail-readable labels for the 4 weight-decomposition contributions
_DECOMP_LABELS: dict[str, str] = {
    "return":     "predicted return",
    "variance":   "individual volatility",
    "covariance": "diversification with your other stocks",
    "risk":       "your risk profile",
}


def shap_waterfall_frame(shap_row: pd.Series, top_n: int = 4) -> pd.DataFrame:
    """Build a frame for a Plotly Waterfall trace from one SHAP row.

    Values are scaled to percentage points (×100) for display. Rows:
    [Base] + top-N features by |SHAP| + [Prediction]. The waterfall sums
    to the prediction only if N == 7; with top-N < 7 the chart is for
    explanation, not accounting.

    Parameters
    ----------
    shap_row : pd.Series
        One row from a snapshot SHAP DataFrame — shap_* cols + feature
        values + base_value + prediction.
    top_n : int
        Number of feature bars between Base and Prediction.

    Returns
    -------
    pd.DataFrame with columns ['label', 'value', 'measure'] suitable for
    direct use with plotly.graph_objects.Waterfall.
    """
    pairs = []
    for col in shap_row.index:
        if not col.startswith("shap_"):
            continue
        feature = col[len("shap_"):]
        if feature not in _FEATURE_COPY:
            continue
        pairs.append((feature, float(shap_row[col])))
    pairs.sort(key=lambda x: -abs(x[1]))
    top = pairs[:top_n]

    scale = 100.0
    base = float(shap_row["base_value"]) * scale
    prediction = float(shap_row["prediction"]) * scale

    rows = [{"label": "Base", "value": base, "measure": "absolute"}]
    for feature, val in top:
        rows.append({
            "label":   _FEATURE_COPY[feature]["label"],
            "value":   val * scale,
            "measure": "relative",
        })
    rows.append({"label": "Prediction", "value": prediction, "measure": "total"})
    return pd.DataFrame(rows)


def _top_signed_feature(shap_row: pd.Series) -> tuple[str, float]:
    """Return (display_label, signed_shap_value) of the feature with the
    largest absolute SHAP contribution."""
    pairs = []
    for col in shap_row.index:
        if not col.startswith("shap_"):
            continue
        feature = col[len("shap_"):]
        if feature not in _FEATURE_COPY:
            continue
        pairs.append((feature, float(shap_row[col])))
    pairs.sort(key=lambda x: -abs(x[1]))
    feature, val = pairs[0]
    return _FEATURE_COPY[feature]["label"], val


def shap_one_liner_return(shap_row: pd.Series, ticker: str) -> str:
    """Natural-language one-liner for the Return SHAP sub-card."""
    label, val = _top_signed_feature(shap_row)
    direction = "above" if val > 0 else "below"
    return (
        f"The biggest reason **{ticker}**'s predicted return sits "
        f"**{direction}** the model's typical forecast is its **{label.lower()}**."
    )


def shap_one_liner_volatility(shap_row: pd.Series, ticker: str) -> str:
    """Natural-language one-liner for the Volatility SHAP sub-card."""
    label, val = _top_signed_feature(shap_row)
    direction = "above" if val > 0 else "below"
    return (
        f"The biggest reason **{ticker}**'s predicted risk sits "
        f"**{direction}** the model's typical forecast is its **{label.lower()}**."
    )


def decomposition_summary(decomp_row: pd.Series, ticker: str) -> str:
    """Natural-language summary for the weight decomposition sub-card.

    Identifies the dominant contribution (largest |value|) among the four
    components — return, variance, covariance, risk — and renders a
    one-sentence narrative. Handles the all-near-zero case (stock sits at
    the baseline) explicitly.
    """
    candidates = [
        ("return",     float(decomp_row["return"])),
        ("variance",   float(decomp_row["variance"])),
        ("covariance", float(decomp_row["covariance"])),
        ("risk",       float(decomp_row["risk"])),
    ]
    candidates.sort(key=lambda x: -abs(x[1]))
    top_key, top_val = candidates[0]

    if abs(top_val) < 0.001:
        return (
            f"All four contributions for **{ticker}** are small — its weight "
            f"is close to the baseline allocation."
        )

    direction = "higher" if top_val > 0 else "lower"
    return (
        f"The biggest reason **{ticker}**'s weight is **{direction}** than "
        f"a baseline allocation is its **{_DECOMP_LABELS[top_key]}**."
    )


def bucket_vol_outlook(predicted_vols: pd.Series) -> pd.Series:
    """Map predicted volatilities to retail-readable tercile band strings.

    Returns "Elevated ↑" / "Moderate" / "Low" based on cross-sectional
    terciles. Callers should pass the FULL universe predictions (93 tickers)
    so bands remain meaningful at any selection size — passing a 5-stock
    slice would give noisy terciles where each band holds ~1-2 stocks.

    Used on the Phase 5 allocation table where raw vol values lack retail
    intuition. The asymmetric treatment (return shown as raw %, vol shown
    as band label) is methodologically deliberate: returns are a unit
    retail investors interpret directly; volatility is an abstract measure
    that maps onto retail mental models faster as a categorical level.

    Returns a Series of strings indexed identically to the input.
    """
    rank_pct = predicted_vols.rank(pct=True)

    def label(p: float) -> str:
        if p > 2 / 3:
            return "Elevated ↑"
        if p < 1 / 3:
            return "Low"
        return "Moderate"

    return rank_pct.apply(label)
