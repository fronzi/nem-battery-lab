"""Walk-forward probabilistic forecasting with monthly refits and conformal calibration."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PERIODS_PER_DAY, ExperimentConfig
from .features import FEATURES, price_inverse


def qname(q: float) -> str:
    return f"q{round(q * 100):02d}"


def fit_quantile_models(
    train: pd.DataFrame, config: ExperimentConfig, threads: int = 1
) -> dict[float, lgb.LGBMRegressor]:
    models = {}
    for quantile in config.quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=config.lgbm_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=config.seed,
            n_jobs=threads,
            verbose=-1,
        )
        models[quantile] = model.fit(train[FEATURES], train["y"])
    return models


def predict_quantiles(
    models: dict[float, lgb.LGBMRegressor], frame: pd.DataFrame
) -> dict[float, np.ndarray]:
    quantiles = sorted(models)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = np.column_stack([models[q].predict(frame[FEATURES]) for q in quantiles])
    values = np.sort(values, axis=1)  # removes quantile crossing
    return {q: values[:, i] for i, q in enumerate(quantiles)}


def conformal_adjustment(y: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float) -> float:
    """Conformalised quantile regression (CQR) score at miscoverage ``alpha``."""
    scores = np.maximum(low - y, y - high)
    n = len(scores)
    rank = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[rank - 1])


def expected_from_quantiles(prediction: dict[float, np.ndarray]) -> np.ndarray:
    """Trapezoid estimate of expected price from back-transformed predictive quantiles."""
    levels = np.array(sorted(prediction))
    prices = np.column_stack([price_inverse(prediction[q]) for q in levels])
    grid = np.concatenate(([0.0], levels, [1.0]))
    extended = np.column_stack([prices[:, 0], prices, prices[:, -1]])
    return np.trapezoid(extended, grid, axis=1)


def walk_forward_region(
    features: pd.DataFrame, config: ExperimentConfig, threads: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each test month using only days before that month."""
    features = features.sort_values(["day", "period"]).reset_index(drop=True)
    test_start = pd.Timestamp(config.test_start)
    months = pd.period_range(test_start, features["day"].max().to_period("M"), freq="M")
    outputs, residual_paths = [], []
    for month in months:
        start, end = month.start_time.normalize(), month.end_time.normalize()
        history = features.loc[features["day"].lt(start)]
        test = features.loc[features["day"].between(start, end)]
        if test.empty:
            continue
        calibration_start = start - pd.Timedelta(days=config.calibration_days)
        proper = history.loc[history["day"].lt(calibration_start)]
        calibration = history.loc[history["day"].ge(calibration_start)]
        models = fit_quantile_models(proper, config, threads)
        calib_pred = predict_quantiles(models, calibration)
        test_pred = predict_quantiles(models, test)
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
            history[FEATURES], history["y"]
        )

        out = test[["region", "day", "period", "price", "y"]].copy()
        out["month"] = str(month)
        out["naive_d1"] = price_inverse(test["price_d1"])
        out["rolling_7d"] = price_inverse(test["price_same_period_7d_mean"])
        out["ridge"] = price_inverse(ridge.predict(test[FEATURES]))
        for quantile, values in test_pred.items():
            out[f"t_{qname(quantile)}"] = values
            out[qname(quantile)] = price_inverse(values)
        out["quantile_mean"] = expected_from_quantiles(test_pred)
        for low, high in ((0.05, 0.95), (0.10, 0.90)):
            alpha = 1 - (high - low)
            shift = conformal_adjustment(
                calibration["y"].to_numpy(), calib_pred[low], calib_pred[high], alpha
            )
            out[f"conformal_{qname(low)}"] = price_inverse(test_pred[low] - shift)
            out[f"conformal_{qname(high)}"] = price_inverse(test_pred[high] + shift)
        outputs.append(out)

        residual = calibration[["day", "period", "y"]].copy()
        residual["residual"] = calibration["y"].to_numpy() - calib_pred[0.50]
        wide = residual.pivot(index="day", columns="period", values="residual").dropna()
        for day, row in wide.iterrows():
            residual_paths.append(
                pd.DataFrame(
                    {
                        "month": str(month),
                        "calibration_day": day,
                        "period": np.arange(PERIODS_PER_DAY),
                        "residual": row.to_numpy(),
                    }
                )
            )
    return pd.concat(outputs, ignore_index=True), pd.concat(residual_paths, ignore_index=True)


def pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    error = y - prediction
    return np.maximum(quantile * error, (quantile - 1) * error)


def forecast_metrics(
    predictions: pd.DataFrame, quantiles: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    point_models = ["naive_d1", "rolling_7d", "ridge", "q50", "quantile_mean"]
    for region, group in predictions.groupby("region"):
        y = group["price"].to_numpy()
        clipped = np.clip(y, -1000, 1000)
        for name in point_models:
            prediction = group[name].to_numpy()
            rows.append(
                {
                    "region": region,
                    "model": name,
                    "mae": np.abs(y - prediction).mean(),
                    "rmse": np.sqrt(((y - prediction) ** 2).mean()),
                    "mae_price_clipped_1000": np.abs(clipped - prediction).mean(),
                    "bias": (prediction - y).mean(),
                }
            )
    point = pd.DataFrame(rows)
    pinball_rows, coverage_rows = [], []
    for region, group in predictions.groupby("region"):
        y = group["price"].to_numpy()
        pinball_rows.append(
            {
                "region": region,
                "mean_pinball": np.mean(
                    [pinball(y, group[qname(q)].to_numpy(), q).mean() for q in quantiles]
                ),
            }
        )
        for low, high in ((0.05, 0.95), (0.10, 0.90)):
            for method, lo, hi in (
                ("quantile_regression", qname(low), qname(high)),
                ("conformal", f"conformal_{qname(low)}", f"conformal_{qname(high)}"),
            ):
                inside = group["price"].between(group[lo], group[hi])
                coverage_rows.append(
                    {
                        "region": region,
                        "nominal": round(high - low, 2),
                        "method": method,
                        "coverage": inside.mean(),
                        "mean_width": (group[hi] - group[lo]).mean(),
                        "median_width": (group[hi] - group[lo]).median(),
                    }
                )
    return point.merge(pd.DataFrame(pinball_rows), on="region"), pd.DataFrame(coverage_rows)
