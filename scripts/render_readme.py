"""Render README.md from committed result tables so reported numbers cannot drift."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path("results")
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
NAMES = {
    "fixed_schedule": "Fixed solar/evening schedule",
    "naive_d1": "Yesterday's prices",
    "rolling_7d": "7-day same-period profile",
    "ridge": "Ridge regression",
    "lgbm_median": "LightGBM median",
    "lgbm_quantile_mean": "LightGBM quantile mean",
    "lgbm_scenario_mean": "LightGBM scenario mean",
    "lgbm_cvar": "LightGBM CVaR",
    "perfect_foresight": "Perfect foresight (upper bound)",
}


def table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.to_numpy()]
    return "\n".join([head, rule, *rows])


s = pd.read_csv(R / "strategy_metrics.csv")
value = s.pivot(index="strategy", columns="region", values="annualised_aud_per_mw").loc[list(NAMES)]
capture = s.pivot(index="strategy", columns="region", values="capture_of_perfect_foresight")
value_rows = pd.DataFrame({"Strategy": [NAMES[x] for x in value.index]})
for region in REGIONS:
    value_rows[region] = [
        f"{value.loc[k, region] / 1000:,.0f}k ({capture.loc[k, region]:.0%})" for k in value.index
    ]

f = pd.read_csv(R / "forecast_metrics.csv")
mae = f.pivot(index="model", columns="region", values="mae").loc[
    ["naive_d1", "rolling_7d", "ridge", "q50", "quantile_mean"]
]
mae_rows = mae.round(1).reset_index().rename(columns={"model": "Forecast"})
mae_rows["Forecast"] = [
    "Yesterday's prices", "7-day profile", "Ridge", "LightGBM median", "LightGBM quantile mean",
]  # fmt: skip

c = pd.read_csv(R / "interval_coverage.csv")
coverage = c.groupby(["nominal", "method"])["coverage"].agg(["min", "max"]).reset_index()
qr80 = coverage.query("nominal == 0.8 and method == 'quantile_regression'").iloc[0]
cp80 = coverage.query("nominal == 0.8 and method == 'conformal'").iloc[0]
qr90 = coverage.query("nominal == 0.9 and method == 'quantile_regression'").iloc[0]
cp90 = coverage.query("nominal == 0.9 and method == 'conformal'").iloc[0]

dm = pd.read_csv(R / "forecast_significance.csv").query("model == 'q50'")
dm_naive = dm.query("baseline == 'naive_d1'")
dm_roll = dm.query("baseline == 'rolling_7d'")
naive = pd.read_csv(R / "strategy_vs_naive.csv").query("strategy == 'lgbm_quantile_mean'")
roll = pd.read_csv(R / "strategy_vs_rolling_7d_post_hoc.csv").query(
    "strategy == 'lgbm_quantile_mean'"
)
significant = roll.loc[roll["holm_p"].lt(0.05), "region"].tolist()
uplift_rows = pd.DataFrame(
    {
        "Region": roll["region"],
        "Uplift vs 7-day profile (A$/MW-yr)": roll["annualised_uplift_aud_per_mw"].map(
            "{:,.0f}".format
        ),
        "95% block-bootstrap CI": [
            f"{a:,.0f} to {b:,.0f}"
            for a, b in zip(roll["ci95_low"], roll["ci95_high"], strict=True)
        ],
        "Holm-adjusted p": roll["holm_p"].map("{:.3f}".format),
    }
)
t = pd.read_csv(R / "cvar_vs_risk_neutral.csv")
d = pd.read_csv(R / "monitoring_drift.csv")
mc = pd.read_csv(R / "monitoring_coverage.csv")
q = pd.read_csv(R / "data_quality.csv")
days = int(s["days"].iloc[0])
share_drift = (d["features_over_0_25"] > 0).mean()
cvar_mean_negative = int((t["mean_ci95_high"] < 0).sum())
cvar_tail_clear = int(((t["cvar_ci95_low"] > 0) | (t["cvar_ci95_high"] < 0)).sum())

README = f"""# NEM Battery Lab

[![CI](https://github.com/fronzi/nem-battery-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/fronzi/nem-battery-lab/actions/workflows/ci.yml)

**Does better price forecasting make a grid battery more valuable?** This project answers that
question end to end on public Australian electricity-market data: ingestion, a SQL star schema,
probabilistic forecasting, mixed-integer optimisation, risk analysis, an API and monitoring.

**Main finding.** LightGBM clearly improves forecast accuracy. That accuracy does not always become
money. Against yesterday's prices, its expected-price strategy adds value in all five regions. Against
a much simpler 7-day price profile, the added value is statistically clear only in
{", ".join(significant)}. The simple baseline is hard to beat, and this repository reports that plainly.

![Value captured by each strategy](results/figures/fig1_value_capture.png)

## Results

Test period: **{days} trading days per region, 1 Jan 2024 to 31 Oct 2026**. Models are refitted monthly
with earlier data only. Battery: 1 MW / 2 MWh, 85% round-trip efficiency, A$10/MWh discharge wear,
daily state of charge returns to 50%.

### Value by strategy (A$ per MW-year; share of perfect foresight)

{table(value_rows)}

### Forecast error (MAE, A$/MWh)

{table(mae_rows)}

### Pre-registered hypotheses ([plan](docs/RESEARCH_PLAN.md))

| ID | Outcome | Evidence |
|---|---|---|
| H1: LightGBM beats yesterday's prices on MAE | **Supported** | Daily MAE falls by A${-dm_naive["mae_difference"].max():.1f} to A${-dm_naive["mae_difference"].min():.1f}/MWh; Diebold-Mariano p ≤ {dm_naive["p_value"].max():.1e} in all regions |
| H2: conformal calibration improves coverage | **Supported** | 80% interval: {qr80["min"]:.1%}-{qr80["max"]:.1%} before, {cp80["min"]:.1%}-{cp80["max"]:.1%} after. 90% interval: {qr90["min"]:.1%}-{qr90["max"]:.1%} before, {cp90["min"]:.1%}-{cp90["max"]:.1%} after |
| H3: LightGBM dispatch beats dispatch on yesterday's prices | **Supported** | Uplift A${naive["annualised_uplift_aud_per_mw"].min():,.0f} to A${naive["annualised_uplift_aud_per_mw"].max():,.0f} per MW-year; every Holm-adjusted p ≤ {naive["holm_p"].max():.3f} |
| H4: CVaR dispatch improves tail outcomes | **Not supported** | The 5% daily CVaR difference is statistically clear in {cvar_tail_clear} of 5 regions. Mean value is lower in all 5 regions, and the interval is entirely below zero in {cvar_mean_negative} |

### The harder comparison (added after seeing the results)

I added this analysis when the 7-day profile turned out to be stronger than the pre-specified baseline.
Treat it as exploratory.

{table(uplift_rows)}

The forecast is still more accurate than the 7-day profile (daily MAE lower by
A${-dm_roll["mae_difference"].max():.1f} to A${-dm_roll["mae_difference"].min():.1f}/MWh). Dispatch value depends mostly on getting the
*shape and extremes* of the day right, so lower average error does not translate one-for-one into revenue.

<p align="center"><img src="results/figures/fig4_example_dispatch.png" width="720" alt="Forecast and optimised battery schedule for one day"></p>

## Skills demonstrated

| Target role | Essential skills shown here | Highly desirable extensions shown here |
|---|---|---|
| **Senior Data Scientist** | Python, advanced SQL (window functions, marts), feature engineering, walk-forward evaluation, baselines, statistical testing | Star schema, BI-ready Parquet, interactive [dashboard](results/dashboard.html), decision-focused metrics |
| **Machine Learning Engineer** | Packaged code, typed modules, tests, CLI, FastAPI serving, model artefacts | Docker, GitHub Actions, data-quality gates, PSI drift and coverage monitoring, run manifests |
| **Energy Modelling / Optimisation Scientist** | Market data, battery physics, MILP dispatch, efficiency and wear | Scenario optimisation, CVaR, perfect-foresight benchmark, five-region comparison |
| **Quantitative Analyst / Researcher** | No-look-ahead backtesting, P&L attribution, drawdown, VaR/CVaR | Block bootstrap, Holm correction, Diebold-Mariano tests, conformal intervals |
| **Applied / AI Scientist** | Quantile gradient boosting, pre-registered hypotheses, honest baselines | Conformalised quantile regression, translating ML into decisions |
| **R&D Scientist / Research Manager** | Decision question, milestones, risk register, reproducible reporting | [Research plan](docs/RESEARCH_PLAN.md), [decision log](docs/DECISIONS.md), [model card](docs/MODEL_CARD.md) |

## Architecture

```mermaid
flowchart LR
  A[AEMO price and demand CSVs] --> B[Validated half-hourly data]
  B --> C[(DuckDB star schema and marts)]
  B --> D[No-look-ahead features]
  D --> E[Monthly walk-forward models]
  E --> F[Quantiles and conformal intervals]
  F --> G[MILP and CVaR dispatch]
  G --> H[Settlement and risk]
  F --> I[Monitoring]
  E --> J[FastAPI service]
  C --> K[Dashboard and BI exports]
```

## Run it

```bash
make install
make test
.venv/bin/nem-lab all          # downloads about 120 MB and reproduces every table and figure
make serve                     # after `nem-lab all`: http://127.0.0.1:8000/docs
```

```bash
docker build -t nem-battery-lab .
docker run -p 8000:8000 nem-battery-lab   # /health and /dispatch; /forecast needs mounted /data
```

| Endpoint | Purpose |
|---|---|
| `POST /dispatch` | Optimal schedule for 48 half-hourly prices |
| `GET /forecast/{{region}}` | Next-day price quantiles from the trained bundle |
| `GET /health` | Service health |

## Data quality and monitoring

- {int(q["rows_5min"].sum()):,} five-minute records across five regions; {int(q["missing_half_hours"].sum())} missing half-hours and {int(q["duplicates_removed"].sum())} duplicates.
- Warehouse tests check fact grain, foreign keys and valid trading periods. All return zero failures.
- Conformal 80% coverage ranged from {mc["conformal_80_coverage"].min():.0%} to {mc["conformal_80_coverage"].max():.0%} by region-month.
- At least one feature exceeded PSI 0.25 in {share_drift:.0%} of region-months, even against the same month a
  year earlier. PSI is therefore a useful diagnostic here, but too sensitive to trigger retraining alone.

## Assumptions and limits

- **This is a research backtest, not an investment or trading recommendation.**
- The battery is a price taker: no market impact, bids, rebidding or dispatch constraints.
- Prices are half-hourly averages of 5-minute prices. A real battery can act within the half hour.
- Excludes FCAS, network charges, market fees, financing, capacity fade and outages.
- Forecasts use market history and calendar features only. Weather and AEMO pre-dispatch forecasts are absent.
- 2024-2026 contains large price spikes. Results depend on that regime.

**What would change the conclusion:** adding pre-dispatch and weather forecasts, 5-minute co-optimised
dispatch with FCAS, or a longer test period with a different volatility regime.

## Data and attribution

Source: [Australian Energy Market Operator (AEMO), Aggregated Price and Demand Data](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data).
AEMO is the author of the source material, used under its [copyright permissions](https://aemo.com.au/privacy-and-legal-notices/copyright-permissions).
Raw data are downloaded by the pipeline and are not stored in this repository.

Companion project: [trustworthy-materials-ml](https://github.com/fronzi/trustworthy-materials-ml).
Developed with AI coding assistance. Licensed under MIT.
"""
Path("README.md").write_text(README, encoding="utf-8")
