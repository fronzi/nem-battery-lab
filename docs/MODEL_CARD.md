# Model card: day-ahead NEM price quantile models

**Purpose.** Research forecasts of half-hourly regional reference prices for battery-dispatch studies.
**Not for.** Live trading, bidding, financial advice or system-security decisions.

| Item | Detail |
|---|---|
| Model | LightGBM quantile regression, seven quantiles, one model set per region |
| Target | `asinh(price / 100)` |
| Inputs | Lagged prices and demand up to the previous day, previous-day statistics, other-region prices, calendar terms |
| Training | Expanding window, monthly refit; latest 90 days reserved for conformal calibration |
| Test | 1 Jan 2024 to 31 Oct 2026, all five NEM regions |
| Metrics | MAE, RMSE, clipped MAE, pinball loss, interval coverage and decision value |
| Known weaknesses | Spikes, regime changes, no weather or pre-dispatch data |
| Monitoring | PSI by feature and monthly conformal coverage |
