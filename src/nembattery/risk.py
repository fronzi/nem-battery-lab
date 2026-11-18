"""Performance, tail-risk and statistical comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def max_drawdown(pnl: np.ndarray) -> float:
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    return float((peak - equity).max())


def tail_metrics(pnl: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    var = float(np.quantile(pnl, alpha))
    tail = pnl[pnl <= var]
    return {"var_5pct_daily": var, "cvar_5pct_daily": float(tail.mean())}


def block_bootstrap_mean(
    values: np.ndarray, block: int = 7, draws: int = 2000, seed: int = 0
) -> np.ndarray:
    """Circular moving-block bootstrap distribution of the sample mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets) % n
    return values[indices.reshape(draws, -1)[:, :n]].mean(axis=1)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 7) -> tuple[float, float]:
    """HAC Diebold-Mariano statistic for mean(loss_a - loss_b); negative favours A."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    centred = d - d.mean()
    variance = centred @ centred / n
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        variance += 2 * weight * (centred[k:] @ centred[:-k]) / n
    if variance <= 0:
        return 0.0, 1.0
    statistic = d.mean() / np.sqrt(variance / n)
    return float(statistic), float(2 * stats.norm.sf(abs(statistic)))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def strategy_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    wide = daily.pivot_table(index=["region", "day"], columns="strategy", values="pnl")
    rows = []
    for region, table in wide.groupby(level="region"):
        benchmark = table["perfect_foresight"].sum()
        for strategy in table.columns:
            pnl = table[strategy].to_numpy()
            std = pnl.std(ddof=1)
            rows.append(
                {
                    "region": region,
                    "strategy": strategy,
                    "days": len(pnl),
                    "annualised_aud_per_mw": pnl.mean() * 365,
                    "capture_of_perfect_foresight": pnl.sum() / benchmark,
                    "daily_mean": pnl.mean(),
                    "daily_std": std,
                    "annualised_return_to_vol": pnl.mean() / std * np.sqrt(365) if std else np.nan,
                    "max_drawdown": max_drawdown(pnl),
                    "worst_day": pnl.min(),
                    "share_positive_days": (pnl > 0).mean(),
                    **tail_metrics(pnl),
                }
            )
    return pd.DataFrame(rows)


def compare_to_baseline(
    daily: pd.DataFrame, baseline: str = "naive_d1", seed: int = 0
) -> pd.DataFrame:
    wide = daily.pivot_table(index=["region", "day"], columns="strategy", values="pnl")
    rows = []
    for region, table in wide.groupby(level="region"):
        group = []
        for strategy in table.columns:
            if strategy in (baseline, "perfect_foresight"):
                continue
            difference = (table[strategy] - table[baseline]).to_numpy()
            draws = block_bootstrap_mean(difference, seed=seed) * 365
            p_value = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
            group.append(
                {
                    "region": region,
                    "strategy": strategy,
                    "baseline": baseline,
                    "annualised_uplift_aud_per_mw": difference.mean() * 365,
                    "ci95_low": np.quantile(draws, 0.025),
                    "ci95_high": np.quantile(draws, 0.975),
                    "bootstrap_p": max(p_value, 1 / len(draws)),
                }
            )
        adjusted = holm_adjust(np.array([row["bootstrap_p"] for row in group]))
        for row, value in zip(group, adjusted, strict=True):
            row["holm_p"] = float(value)
        rows.extend(group)
    return pd.DataFrame(rows)


def compare_tail_risk(
    daily: pd.DataFrame,
    strategy: str = "lgbm_cvar",
    baseline: str = "lgbm_scenario_mean",
    draws: int = 2000,
    block: int = 7,
    seed: int = 0,
) -> pd.DataFrame:
    """Block-bootstrap the mean and 5% CVaR differences between two strategies."""
    wide = daily.pivot_table(index=["region", "day"], columns="strategy", values="pnl")
    rng = np.random.default_rng(seed)
    rows = []
    for region, table in wide.groupby(level="region"):
        a, b = table[strategy].to_numpy(), table[baseline].to_numpy()
        n = len(a)
        starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block))))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
        cvar_differences = np.array(
            [
                tail_metrics(a[i])["cvar_5pct_daily"] - tail_metrics(b[i])["cvar_5pct_daily"]
                for i in indices
            ]
        )
        mean_differences = (a[indices] - b[indices]).mean(axis=1) * 365
        rows.append(
            {
                "region": region,
                "strategy": strategy,
                "baseline": baseline,
                "annualised_mean_difference": (a - b).mean() * 365,
                "mean_ci95_low": np.quantile(mean_differences, 0.025),
                "mean_ci95_high": np.quantile(mean_differences, 0.975),
                "cvar_5pct_daily_difference": tail_metrics(a)["cvar_5pct_daily"]
                - tail_metrics(b)["cvar_5pct_daily"],
                "cvar_ci95_low": np.quantile(cvar_differences, 0.025),
                "cvar_ci95_high": np.quantile(cvar_differences, 0.975),
            }
        )
    return pd.DataFrame(rows)
