"""Orchestration of the end-to-end experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .backtest import backtest_region
from .config import REGIONS, ExperimentConfig, Paths
from .features import FEATURES, build_region_features
from .forecast import fit_quantile_models, forecast_metrics, walk_forward_region
from .monitoring import monthly_coverage, monthly_drift
from .risk import compare_tail_risk, compare_to_baseline, diebold_mariano, strategy_metrics


def _forecast_one(region: str, paths: Paths, config: ExperimentConfig) -> str:
    panel = pd.read_parquet(paths.processed / "half_hourly.parquet")
    features = build_region_features(panel, region)
    predictions, residuals = walk_forward_region(features, config, threads=2)
    predictions.to_parquet(paths.predictions / f"{region}_predictions.parquet", index=False)
    residuals.to_parquet(paths.predictions / f"{region}_residuals.parquet", index=False)
    monthly_drift(features, config.test_start).to_parquet(
        paths.predictions / f"{region}_drift.parquet", index=False
    )
    return region


def _drift_one(region: str, paths: Paths, config: ExperimentConfig) -> None:
    panel = pd.read_parquet(paths.processed / "half_hourly.parquet")
    features = build_region_features(panel, region)
    monthly_drift(features, config.test_start).to_parquet(
        paths.predictions / f"{region}_drift.parquet", index=False
    )


def refresh_monitoring(paths: Paths, config: ExperimentConfig, jobs: int = 5) -> None:
    joblib.Parallel(n_jobs=jobs)(
        joblib.delayed(_drift_one)(region, paths, config) for region in REGIONS
    )


def run_forecasts(paths: Paths, config: ExperimentConfig, jobs: int = 5) -> None:
    paths.ensure()
    joblib.Parallel(n_jobs=jobs)(
        joblib.delayed(_forecast_one)(region, paths, config) for region in REGIONS
    )


def _backtest_one(region: str, paths: Paths, config: ExperimentConfig) -> pd.DataFrame:
    predictions = pd.read_parquet(paths.predictions / f"{region}_predictions.parquet")
    residuals = pd.read_parquet(paths.predictions / f"{region}_residuals.parquet")
    return backtest_region(predictions, residuals, config)


def run_backtests(paths: Paths, config: ExperimentConfig, jobs: int = 5) -> pd.DataFrame:
    frames = joblib.Parallel(n_jobs=jobs)(
        joblib.delayed(_backtest_one)(region, paths, config) for region in REGIONS
    )
    daily = pd.concat(frames, ignore_index=True)
    daily.to_parquet(paths.predictions / "daily_pnl.parquet", index=False)
    return daily


def _load(paths: Paths, suffix: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(paths.predictions / f"{r}_{suffix}.parquet") for r in REGIONS],
        ignore_index=True,
    )


def summarise(paths: Paths, config: ExperimentConfig) -> None:
    predictions = _load(paths, "predictions")
    point, coverage = forecast_metrics(predictions, config.quantiles)
    dm_rows = []
    for region, group in predictions.groupby("region"):
        for baseline, models in (
            ("naive_d1", ("rolling_7d", "ridge", "q50", "quantile_mean")),
            ("rolling_7d", ("ridge", "q50", "quantile_mean")),
        ):
            for model in models:
                frame = (
                    group.assign(
                        model_loss=(group["price"] - group[model]).abs(),
                        base_loss=(group["price"] - group[baseline]).abs(),
                    )
                    .groupby("day")[["model_loss", "base_loss"]]
                    .mean()
                )
                statistic, p_value = diebold_mariano(
                    frame["model_loss"].to_numpy(), frame["base_loss"].to_numpy()
                )
                dm_rows.append(
                    {
                        "region": region,
                        "model": model,
                        "baseline": baseline,
                        "mae_difference": (frame["model_loss"] - frame["base_loss"]).mean(),
                        "dm_statistic": statistic,
                        "p_value": p_value,
                    }
                )
    point.to_csv(paths.results / "forecast_metrics.csv", index=False)
    coverage.to_csv(paths.results / "interval_coverage.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(paths.results / "forecast_significance.csv", index=False)
    daily = pd.read_parquet(paths.predictions / "daily_pnl.parquet")
    strategy_metrics(daily).to_csv(paths.results / "strategy_metrics.csv", index=False)
    compare_to_baseline(daily, seed=config.seed).to_csv(
        paths.results / "strategy_vs_naive.csv", index=False
    )
    compare_to_baseline(daily, baseline="rolling_7d", seed=config.seed).to_csv(
        paths.results / "strategy_vs_rolling_7d_post_hoc.csv", index=False
    )
    compare_tail_risk(daily, seed=config.seed).to_csv(
        paths.results / "cvar_vs_risk_neutral.csv", index=False
    )
    monthly_coverage(predictions).to_csv(paths.results / "monitoring_coverage.csv", index=False)
    _load(paths, "drift").to_csv(paths.results / "monitoring_drift.csv", index=False)
    write_manifest(paths, config)


def train_final_models(paths: Paths, config: ExperimentConfig) -> None:
    panel = pd.read_parquet(paths.processed / "half_hourly.parquet")
    for region in REGIONS:
        features = build_region_features(panel, region)
        models = fit_quantile_models(features, config, threads=8)
        joblib.dump(
            {
                "models": models,
                "features": FEATURES,
                "trained_to": str(features["day"].max().date()),
            },
            paths.models / f"{region}.joblib",
        )


def write_manifest(paths: Paths, config: ExperimentConfig) -> None:
    source: Path = paths.processed / "half_hourly.parquet"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "uncommitted"
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "duckdb"]
    manifest = {
        "config": config.to_dict(),
        "processed_data_sha256": digest,
        "git_commit_at_run": commit,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in packages},
        "numpy_seed_policy": int(np.int64(config.seed)),
    }
    (paths.results / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
