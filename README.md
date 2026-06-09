# Hybrid Robo-Advisor

A Streamlit-based hybrid robo-advisor for UK retail investors that combines machine-learning portfolio construction with two-layer explainability and constrained user agency.

> Academic prototype. Not financial advice.

## What it does

The product builds a recommended FTSE 100 portfolio from a user's own stock shortlist and risk preference, then explains the recommendation at two distinct levels. Per stock, it shows which input features drove each prediction. Per portfolio, it shows how much each factor (the return signal, the individual volatility signal, the diversification structure, and the user's risk preference) contributed to the weight assigned to every holding. The user can accept the recommendation, modify any weight by up to plus or minus five percentage points (the Dietvorst, Simmons and Massey constrained-agency design), or reject and start over.

The thesis behind it is simple. Mainstream robo-advisors optimise allocations well but explain them poorly. DIY platforms give control but no analytical scaffolding. The hybrid sits between the two: machine-learning-driven recommendations that the user retains decision authority over, with an explanation layer designed to make the model's view inspectable rather than opaque.

## Architecture

### Predictions

Two Random Forest regressors run on a 63-day forward horizon over a 93-stock FTSE 100 universe (quality filtered from the full constituent list). Seven features are computed daily: 12-month-minus-1-month momentum, 21-day realised volatility, 52-week drawdown, 21-day relative strength against the FTSE 100, 20-day volume ratio, 252-day beta, and the VIX. The feature set was reduced from 11 candidates after a pre-committed Spearman correlation audit dropped any feature with a median absolute rank correlation above 0.70 against another. Hyperparameters were selected by purged time-series cross-validation following López de Prado (2018) with a target-length embargo, scoring on Spearman rank correlation rather than R squared because the downstream optimiser consumes rankings, not levels.

### Optimisation

A mean-variance SLSQP optimiser turns the per-stock predictions into weights. The covariance matrix is built as `Sigma = D Cov D`, with `D` containing the RF-predicted forward volatilities on the diagonal and `Cov` a Ledoit-Wolf shrunk historical correlation matrix off-diagonal. Expected returns are cross-sectionally shrunk toward the mean of the user's selection at an intensity that varies by risk band. The per-asset cap adjusts dynamically with selection size to preserve solver headroom (30 percent at 5 stocks, 10 percent at 15). Four risk bands map to `(lambda, alpha)` pairs from Cautious `(5.0, 0.85)` through Adventurous `(0.5, 0.35)`.

### Explainability

The dashboard runs two distinct explanation layers and never merges them.

1. **SHAP TreeExplainer** with `tree_path_dependent` perturbation produces per-stock attributions for both the return and volatility predictions. Each is rendered as a bar-card with the top features by absolute contribution and a templated plain-language sentence.
2. **Analytical decomposition** runs five counterfactual versions of the optimiser, neutralising the return signal, the individual volatility signal, the correlation structure, and the user's risk preference in turn, and reads each input's contribution to every weight from the differences.

The two layers answer different questions. SHAP answers "why is this prediction this number". The decomposition answers "why is this stock at this weight". The latter is the project's principal methodological contribution.

### Web layer

A multi-page Streamlit dashboard runs the participant journey from risk profiling through to a rebalancing page with realised per-stock performance and a re-check action that re-runs the optimiser against the latest market data. Per-participant portfolio persistence stores the accepted allocation on disk, so a returning user sees their portfolio's accumulated performance, day by day, from their actual investment date. A custom sidebar replaces the default navigation. The theme uses a cream surface with navy and teal accents.

## Results

Held out on data from 2024 onwards:

| Model | Held-out Spearman | Held-out R squared | Baseline |
|---|---|---|---|
| Volatility RF | 0.459 | 0.145 | −0.711 (persistence) |
| Return RF | 0.086 | 0.002 | −0.013 (historical mean) |

The volatility model is strong and is the empirical backbone of the optimiser. The return model has a real but modest ranking signal, four times its monthly-target predecessor, and is treated as a ranking tilt rather than an alpha source. It is heavily shrunk at the optimiser layer in response.

A pipeline ablation over a nine-quarter walk-forward window compares the hybrid (RF-predicted volatilities on the diagonal of `Sigma`) against a simpler baseline (252-day historical realised volatilities on the diagonal of an otherwise identical Ledoit-Wolf-shrunk covariance):

| Configuration | Cumulative return | Annualised vol | Sharpe |
|---|---|---|---|
| **RF-vol covariance (pipeline)** | **+31.08 %** | **10.43 %** | **1.36** |
| Historical-vol covariance (baseline) | +22.42 % | 8.74 % | 1.21 |
| 1/N equal-weight | +35.80 % | 9.82 % | 1.63 |
| FTSE 100 | +25.64 % | 4.51 % | 2.58 (small-sample artefact) |

The hybrid composition delivers plus 8.66 percentage points cumulative return and plus 0.15 Sharpe ratio over the baseline at the cost of 1.70 percentage points of annualised volatility. Equal-weighted 1/N continues to beat both optimised configurations on Sharpe, consistent with DeMiguel, Garlappi and Uppal (2009). Sensitivity sweeps confirm the band defaults are robust: risk aversion `lambda` is empirically inert across `{2, 4, 6}` and shrinkage `alpha` is mathematically inert under the project's cap-binding regime when weights are near-uniform.

## Install and run

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/msc-fintech-robo-advisor.git
cd msc-fintech-robo-advisor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trained model artefacts are not stored in this repository. Regenerate them by running `notebooks/model_v6_quarterly.ipynb`, which trains both Random Forests on the 63-day forward target and writes them to `models/`. The dashboard then needs a frozen snapshot of predictions and SHAP attributions, built via:

```python
from pathlib import Path
from src.models.snapshot import build_snapshot

build_snapshot(
    return_model_path=Path("models/rf_return_v6.joblib"),
    vol_model_path=Path("models/rf_volatility_v6.joblib"),
    features_path=Path("data/processed/features.parquet"),
    output_dir=Path("data/processed"),
)
```

Once the models and snapshot exist, run the dashboard:

```bash
streamlit run app.py
```

Click "Refresh data" in the sidebar to pull a fresh set of prices from yfinance, rebuild the features panel, and regenerate the snapshot.

## Project layout

```
app.py                 Streamlit entry point
.streamlit/            Theme configuration
pages/                 Seven dashboard pages, in journey order
lib/                   Dashboard utilities (state, sidebar, theme, copy, persistence, performance)
src/data/              yfinance ingest, cleaning, refresh
src/features/          Seven features plus two forward targets
src/models/            Optimiser, SHAP explainer, weight decomposition, projection, snapshot
notebooks/             EDA, feature engineering, modelling, allocation validation
assets/                CSS tokens
data/                  Local cache (gitignored)
models/                Trained joblib artefacts (gitignored)
```

## Status

- **Built and running end to end.** A fresh participant journey reaches the rebalancing page with a saved portfolio.
- **In user testing.** Initial in-person sessions at UWE Bristol under Faculty Research Ethics approval. The instrument combines four matched pre/post Likert items, the System Usability Scale, feature-specific perceived-value items, a forced-rank attribution item, and open-text qualitative responses, with a 22-event JSONL behavioural log per session for triangulation.
- **Walk-forward backtesting** against equal-weight (1/N) and FTSE 100 benchmarks ongoing. The pipeline ablation result above is the current state of that comparison.

## Limitations

The project is honest about the boundaries an academic prototype operates within. The most important to call out:

- The investable universe is the current FTSE 100, so the model carries survivorship bias (Brown, Goetzmann and Ross, 1995). Companies demoted, delisted, or acquired during the sample period are not in the training data, so predicted returns are systematically biased upward and downside risk is understated. Correcting this would require paid historical constituent data.
- The return model has a real but modest ranking signal (held-out Spearman 0.086). The dissertation positions it as a ranking tilt and an explainability substrate rather than as an alpha source.
- The forward projection is Gaussian Monte Carlo. Real equity returns have fatter tails than the cone suggests (Cont,2001). The dashboard surfaces this caveat in-page rather than burying it.
- The within-subjects pre/post study design with no unconstrained-agency control arm can detect that participants felt more confident, understood, and satisfied after using the artefact, but it cannot isolate which component of the explanation system drove the effect. H1 through H4 are reported as exploratory, not confirmatory.
- The held-out backtest is nine quarterly observations long, so summary statistics like the Sharpe ratio carry wide confidence intervals. The pipeline ablation result above is reported with that caveat in mind.

## References

- Brown, S. J., Goetzmann, W. N., & Ross, S. A. (1995). Survival. *Journal of Finance*, 50(3), 853–873.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915–1953.
- Dietvorst, B. J., Simmons, J. P., & Massey, C. (2018). Overcoming algorithm aversion: people will use imperfect algorithms if they can (even slightly) modify them. *Management Science*, 64(3), 1155–1170.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance*, 10(5), 603–621.

---

Built as part of the MSc Financial Technology dissertation at UWE Bristol.
