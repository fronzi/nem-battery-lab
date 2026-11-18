"""Day-ahead feature engineering. Every feature for day D uses data up to the end of day D-1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PERIODS_PER_DAY

FEATURES = [
    "price_d1", "price_d2", "price_d7", "price_same_period_7d_mean", "price_same_period_7d_std",
    "prev_day_mean", "prev_day_std", "prev_day_min", "prev_day_max", "prev_day_negative_share",
    "demand_d1", "demand_d7", "other_regions_price_d1", "period_sin", "period_cos", "dow",
    "is_weekend", "month_sin", "month_cos",
]  # fmt: skip


def price_transform(price: np.ndarray | pd.Series) -> np.ndarray:
    return np.arcsinh(np.asarray(price, dtype=float) / 100.0)


def price_inverse(value: np.ndarray | pd.Series) -> np.ndarray:
    return 100.0 * np.sinh(np.asarray(value, dtype=float))


def add_trading_keys(data: pd.DataFrame) -> pd.DataFrame:
    start = data["ts"] - pd.Timedelta(minutes=30)
    return data.assign(
        day=start.dt.normalize(),
        period=(start.dt.hour * 2 + start.dt.minute // 30).astype(int),
    )


def day_matrix(data: pd.DataFrame, value: str) -> pd.DataFrame:
    matrix = data.pivot(index="day", columns="period", values=value)
    days = pd.date_range(matrix.index.min(), matrix.index.max(), freq="D", name="day")
    return matrix.reindex(index=days, columns=range(PERIODS_PER_DAY))


def build_region_features(
    panel: pd.DataFrame, region: str, target_day: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Return one row per (day, period); ``target_day`` may have no observed prices yet."""
    panel = add_trading_keys(panel)
    own = panel.loc[panel["region"].eq(region)]
    price = day_matrix(own, "price")
    demand = day_matrix(own, "demand_mw")
    if target_day is not None:
        days = pd.date_range(price.index.min(), target_day, freq="D", name="day")
        price, demand = price.reindex(days), demand.reindex(days)
    others = panel.loc[panel["region"].ne(region)]
    other_long = pd.DataFrame(others.groupby(["day", "period"], as_index=False)["price"].mean())
    other = day_matrix(other_long, "price").reindex(price.index)

    p = price_transform(price.to_numpy())
    d = demand.to_numpy() / 1000.0
    o = price_transform(other.to_numpy())
    rows: list[pd.DataFrame] = []
    period = np.arange(PERIODS_PER_DAY)
    for i in range(7, len(price)):
        if np.isnan(p[i]).any() and price.index[i] != target_day:
            continue
        history = p[i - 7 : i]
        previous = p[i - 1]
        if np.isnan(history).any() or np.isnan(d[i - 7 : i]).any() or np.isnan(o[i - 1]).any():
            continue
        day = price.index[i]
        rows.append(
            pd.DataFrame(
                {
                    "day": day,
                    "period": period,
                    "price": price.iloc[i].to_numpy(),
                    "y": p[i],
                    "price_d1": p[i - 1],
                    "price_d2": p[i - 2],
                    "price_d7": p[i - 7],
                    "price_same_period_7d_mean": history.mean(axis=0),
                    "price_same_period_7d_std": history.std(axis=0),
                    "prev_day_mean": previous.mean(),
                    "prev_day_std": previous.std(),
                    "prev_day_min": previous.min(),
                    "prev_day_max": previous.max(),
                    "prev_day_negative_share": (previous < 0).mean(),
                    "demand_d1": d[i - 1],
                    "demand_d7": d[i - 7],
                    "other_regions_price_d1": o[i - 1],
                    "period_sin": np.sin(2 * np.pi * period / PERIODS_PER_DAY),
                    "period_cos": np.cos(2 * np.pi * period / PERIODS_PER_DAY),
                    "dow": day.day_of_week,
                    "is_weekend": int(day.day_of_week >= 5),
                    "month_sin": np.sin(2 * np.pi * day.month / 12),
                    "month_cos": np.cos(2 * np.pi * day.month / 12),
                }
            )
        )
    if not rows:
        raise ValueError(f"No complete feature rows for {region}")
    return pd.concat(rows, ignore_index=True).assign(region=region)
