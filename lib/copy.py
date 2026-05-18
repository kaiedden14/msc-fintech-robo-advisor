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
