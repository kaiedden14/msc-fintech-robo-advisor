"""Plain-language templating for SHAP attribution cards.

Translates raw SHAP values into retail-readable sentences. String
formatting only, no LLM calls. Templates are deterministic so the
methodology write-up can show the exact transformation rule.

The display labels and value phrasings were chosen for retail
explainability, e.g. 'Recent volatility' instead of 'volatility_21d',
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
        One row from a snapshot SHAP DataFrame, contains shap_* columns,
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
        """Format one SHAP attribution into a label + value description + pp number."""
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


# ---------- Phase 5c: bar-card data + plain-language summaries ----------


# Retail-readable labels for the 4 weight-decomposition contributions
_DECOMP_LABELS: dict[str, str] = {
    "return":     "predicted return",
    "variance":   "individual volatility",
    "covariance": "diversification with your other stocks",
    "risk":       "your risk profile",
}


def _collect_shap_pairs(shap_row: pd.Series) -> list[tuple[str, float]]:
    """Return [(feature_key, signed_shap), ...] sorted by |shap| desc."""
    pairs = []
    for col in shap_row.index:
        if not col.startswith("shap_"):
            continue
        feature = col[len("shap_"):]
        if feature not in _FEATURE_COPY:
            continue
        pairs.append((feature, float(shap_row[col])))
    pairs.sort(key=lambda x: -abs(x[1]))
    return pairs


def shap_bar_card_data(
    shap_row: pd.Series,
    top_n: int = 4,
) -> list[dict[str, Any]]:
    """Return top-N SHAP contributions, formatted for the bar-card UI.

    Filters out features whose absolute contribution rounds to 0.0 pp in
    the display layer (|value| < 0.0005 decimal = 0.05 pp). For stocks
    where the model has weak signal, this may return fewer than top_n
    entries, caller should render an empty-state message in that case.

    Each dict has:
      label             , retail-readable feature label
      contribution_pp   , signed value, in percentage points
      is_positive       , direction flag for colouring (teal vs amber)
      fill_pct          , bar width 0..100, proportional to |value| /
                           max(|value|) across the returned set
    Sort order: positives by descending magnitude, then negatives by
    descending magnitude, matches the design mockup.
    """
    _DISPLAY_ZERO = 5e-4  # 0.05 pp, below this, the value displays as 0.0
    pairs = [p for p in _collect_shap_pairs(shap_row) if abs(p[1]) >= _DISPLAY_ZERO]
    pairs = pairs[:top_n]
    if not pairs:
        return []
    max_abs = max(abs(v) for _, v in pairs)
    if max_abs < 1e-12:
        max_abs = 1.0

    positives = sorted([p for p in pairs if p[1] > 0], key=lambda x: -x[1])
    negatives = sorted([p for p in pairs if p[1] < 0], key=lambda x: x[1])

    return [
        {
            "label":           _FEATURE_COPY[feat]["label"],
            "contribution_pp": val * 100,
            "is_positive":     val > 0,
            "fill_pct":        abs(val) / max_abs * 100,
        }
        for feat, val in positives + negatives
    ]


def decomp_bar_card_data(decomp_row: pd.Series) -> list[dict[str, Any]]:
    """Return the 4 decomposition contributions formatted for the bar-card UI.

    Sort order: positives by descending magnitude, then negatives by
    descending magnitude, then zero-valued at the bottom (e.g. the `risk`
    contribution is exactly 0 for the Balanced band by methodology design).
    """
    items = [
        ("Predicted return",      float(decomp_row["return"])),
        ("Individual volatility", float(decomp_row["variance"])),
        ("Diversification",       float(decomp_row["covariance"])),
        ("Your risk profile",     float(decomp_row["risk"])),
    ]
    # Anything below ±0.05pp (the display rounding threshold) is treated
    # as zero visually, avoids the "-0.0 pp" with an invisible bar
    # display artefact when a contribution rounds to zero anyway.
    _DISPLAY_ZERO = 5e-4
    nonzero = [(lab, v) for lab, v in items if abs(v) >= _DISPLAY_ZERO]
    zeros = [(lab, v) for lab, v in items if abs(v) < _DISPLAY_ZERO]

    max_abs = max((abs(v) for _, v in nonzero), default=1.0)
    if max_abs < 1e-12:
        max_abs = 1.0

    positives = sorted([i for i in nonzero if i[1] > 0], key=lambda x: -x[1])
    negatives = sorted([i for i in nonzero if i[1] < 0], key=lambda x: x[1])

    return [
        {
            "label":           label,
            "contribution_pp": val * 100,
            "is_positive":     val > 0,
            "fill_pct":        abs(val) / max_abs * 100,
            "is_zero":         False,
        }
        for label, val in positives + negatives
    ] + [
        {
            "label":           label,
            "contribution_pp": 0.0,
            "is_positive":     False,
            "fill_pct":        0.0,
            "is_zero":         True,
        }
        for label, _ in zeros
    ]


def shap_summary_sentence(
    shap_row: pd.Series,
    ticker: str,
    kind: str = "return",
) -> str:
    """Plain-language summary naming the top 3 drivers of the outlook.

    kind: 'return' or 'risk', drives the wording. Matches the design
    mockup's narrative style.
    """
    pairs = _collect_shap_pairs(shap_row)[:3]
    if not pairs:
        return ""
    labels = [_FEATURE_COPY[f]["label"].lower() for f, _ in pairs]
    word = "return" if kind == "return" else "risk"

    if len(labels) == 1:
        return (
            f"**{ticker}**'s {word} outlook is driven mainly by its "
            f"**{labels[0]}**."
        )
    if len(labels) == 2:
        return (
            f"**{ticker}**'s {word} outlook is driven mainly by its "
            f"**{labels[0]}** and **{labels[1]}**."
        )
    return (
        f"**{ticker}**'s {word} outlook is driven mainly by its "
        f"**{labels[0]}**, **{labels[1]}**, and **{labels[2]}**."
    )


def decomposition_summary(decomp_row: pd.Series, ticker: str) -> str:
    """Natural-language summary for the weight decomposition sub-card.

    Identifies the dominant contribution (largest |value|) among the four
    components, return, variance, covariance, risk, and renders a
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
            f"All four contributions for **{ticker}** are small, its weight "
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
    so bands remain meaningful at any selection size, passing a 5-stock
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
        """Bucket a percentile rank into 'Low', 'Moderate', or 'Elevated'."""
        if p > 2 / 3:
            return "Elevated ↑"
        if p < 1 / 3:
            return "Low"
        return "Moderate"

    return rank_pct.apply(label)
