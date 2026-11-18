"""Production-style monitoring: feature drift (PSI) and rolling interval coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4
) -> float:
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected, actual = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def monthly_drift(features: pd.DataFrame, test_start: str) -> pd.DataFrame:
    rows = []
    monitored = [f for f in FEATURES if not f.endswith(("_sin", "_cos")) and f != "dow"]
    for month in pd.period_range(test_start, features["day"].max().to_period("M"), freq="M"):
        start, end = month.start_time.normalize(), month.end_time.normalize()
        reference = features.loc[
            features["day"].between(start - pd.Timedelta(days=365), start - pd.Timedelta(days=1))
        ]
        current = features.loc[features["day"].between(start, end)]
        if current.empty or reference.empty:
            continue
        values = {
            name: population_stability_index(reference[name].to_numpy(), current[name].to_numpy())
            for name in monitored
        }
        top = max(values, key=lambda key: values[key])
        rows.append(
            {
                "region": features["region"].iloc[0],
                "month": str(month),
                "max_psi": values[top],
                "top_feature": top,
                "features_over_0_25": sum(value > 0.25 for value in values.values()),
            }
        )
    return pd.DataFrame(rows)


def monthly_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.assign(
        inside=predictions["price"].between(
            predictions["conformal_q10"], predictions["conformal_q90"]
        )
    )
    summary = frame.groupby(["region", "month"], as_index=False)["inside"].mean()
    return pd.DataFrame(summary).rename(columns={"inside": "conformal_80_coverage"})
