"""Daily, no-look-ahead backtest of dispatch strategies, settled at realised prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import DispatchModel, fixed_schedule, settle
from .config import DT_HOURS, PERIODS_PER_DAY, ExperimentConfig
from .features import price_inverse

PRICE_FLOOR, PRICE_CEILING = -1000.0, 20000.0
POINT_STRATEGIES = {
    "naive_d1": "naive_d1",
    "rolling_7d": "rolling_7d",
    "ridge": "ridge",
    "lgbm_median": "q50",
    "lgbm_quantile_mean": "quantile_mean",
}


def scenario_prices(
    median_transformed: np.ndarray,
    residual_days: np.ndarray,
    n_scenarios: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add whole-day calibration residual paths so intraday dependence is preserved."""
    picks = rng.choice(len(residual_days), n_scenarios, replace=len(residual_days) < n_scenarios)
    prices = price_inverse(median_transformed + residual_days[picks])
    return np.clip(prices, PRICE_FLOOR, PRICE_CEILING)


def backtest_region(
    predictions: pd.DataFrame, residuals: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    model = DispatchModel(config.battery)
    rng = np.random.default_rng(config.seed)
    rule = fixed_schedule(config.battery)
    residual_by_month = {
        month: group.pivot(index="calibration_day", columns="period", values="residual").to_numpy()
        for month, group in residuals.groupby("month")
    }
    rows = []
    for day, group in predictions.groupby("day", sort=True):
        group = group.sort_values("period")
        if len(group) != PERIODS_PER_DAY:
            continue
        actual = group["price"].to_numpy()
        region = group["region"].iloc[0]
        plans = {
            "perfect_foresight": model.solve(actual),
            "fixed_schedule": rule,
        }
        for strategy, column in POINT_STRATEGIES.items():
            forecast = np.clip(group[column].to_numpy(), PRICE_FLOOR, PRICE_CEILING)
            plans[strategy] = model.solve(forecast)
        scenarios = scenario_prices(
            group["t_q50"].to_numpy(),
            residual_by_month[group["month"].iloc[0]],
            config.scenarios,
            rng,
        )
        plans["lgbm_scenario_mean"] = model.solve(scenarios.mean(axis=0))
        plans["lgbm_cvar"] = model.solve_cvar(scenarios, config.cvar_alpha, config.cvar_weight)
        for strategy, plan in plans.items():
            rows.append(
                {
                    "region": region,
                    "day": day,
                    "strategy": strategy,
                    "pnl": settle(plan, actual, config.battery),
                    "discharged_mwh": plan.discharge_mw.sum() * DT_HOURS,
                }
            )
    return pd.DataFrame(rows)
