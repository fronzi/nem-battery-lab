# Research plan (written before the walk-forward results were generated)

## Decision question
Does probabilistic day-ahead price forecasting add enough value to a price-taking battery, after
efficiency losses and wear, to justify its complexity over simple operating rules?

## Pre-specified hypotheses
| ID | Hypothesis | Test | Would be falsified by |
|---|---|---|---|
| H1 | LightGBM median forecast has lower MAE than yesterday's prices | Diebold-Mariano on daily MAE, all 5 regions | No improvement, or p >= 0.05, in most regions |
| H2 | Conformal calibration moves 80% and 90% interval coverage closer to nominal | Empirical test-period coverage | Coverage error not reduced |
| H3 | Dispatch from LightGBM expected price earns more than dispatch from yesterday's prices | 7-day block bootstrap of daily P&L difference, Holm-adjusted | CI includes zero in most regions |
| H4 | CVaR dispatch improves 5% daily tail P&L at some cost in mean P&L | Daily P&L CVaR and annualised value | No tail improvement |

## Rules that protect the result
- Test period fixed in advance: 1 Jan 2024 to the last complete downloaded month.
- All five NEM regions and all strategies are reported. No region or model is dropped afterwards.
- Models are refit monthly using data before each test month only.
- Hyperparameters are fixed in `config.py`; they are not tuned on the test period.
- Perfect foresight is an upper benchmark, never an achievable strategy.

## Delivery plan and risks
| Stage | Output | Exit criterion |
|---|---|---|
| Data | Validated half-hourly data and star schema | All integrity checks equal zero |
| Forecast | Walk-forward quantiles | Leakage test passes |
| Decision | Daily dispatch and settlement | Known-answer optimiser test passes |
| Risk | P&L, tail risk, bootstrap comparisons | All regions and strategies reported |
| Productionise | CLI, API, CI, monitoring | Tests pass in CI |

| Risk | Mitigation |
|---|---|
| Look-ahead leakage | Features use data to D-1 only; mutation test in the suite |
| Extreme price spikes dominate metrics | Report MAE, RMSE, clipped MAE and pinball loss |
| Backtest overstates tradable value | State exclusions: market impact, fees, FCAS, network charges, 5-minute dispatch |
| Regime change | Monthly refits, PSI drift and rolling coverage monitoring |
