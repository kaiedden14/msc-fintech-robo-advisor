# Hybrid Robo-Advisor

MSc Financial Technology project — a hybrid robo-advisor for UK retail investors combining ML-driven mean-variance optimisation with two-layer explainability (SHAP + analytical weight decomposition).

**Academic prototype — not financial advice.**

## Repo layout

```
app.py                    Streamlit entry point
.streamlit/               Streamlit theme config
pages/                    One file per dashboard page (1_landing.py ... 7_rebalancing.py)
lib/                      Dashboard-only utilities (state, theme, sidebar, bands)
assets/                   CSS tokens
src/                      Upstream pipeline (data ingest, feature engineering, models)
  data/                   yfinance download + clean prices caching
  features/               7-feature engineer (post-correlation-audit)
  models/                 RF optimiser, SHAP wrapper, decomposition, projection, snapshot
data/                     Local cache (gitignored)
  raw/                    yfinance download
  processed/              Clean prices, features, snapshot files
models/                   Trained RF joblib artifacts (gitignored)
notebooks/                EDA, feature engineering, modelling notebooks
design/                   Figma mockup link and screenshots
```

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Prerequisites:
- `models/rf_return_v5.joblib` and `models/rf_volatility_v5.joblib` exist
- `data/processed/snapshot_*.parquet` exist (regenerate via `src/models/snapshot.py` if missing)
- `data/processed/prices_clean.parquet` exists (auto-built on first run by `src.data.ingest.load_or_build_clean_prices`)

## Pipeline scripts (run upstream, not from the dashboard)

- **Refresh prices to latest available trading day**: `src.data.ingest.refresh_prices(raw_path, clean_path)`
- **Regenerate monthly snapshot**: `src.models.snapshot.build_snapshot(...)`
- **Retrain models**: see `notebooks/model.ipynb`
