"""Publication-quality figures and a self-contained BI-style dashboard."""

from __future__ import annotations

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from .battery import DispatchModel
from .config import PERIODS_PER_DAY, REGIONS, ExperimentConfig, Paths

LABELS = {
    "fixed_schedule": "Fixed schedule",
    "naive_d1": "Yesterday's prices",
    "rolling_7d": "7-day profile",
    "ridge": "Ridge",
    "lgbm_median": "LightGBM median",
    "lgbm_quantile_mean": "LightGBM quantile mean",
    "lgbm_scenario_mean": "LightGBM scenario mean",
    "lgbm_cvar": "LightGBM CVaR",
}
COLOURS = ["#6b7280", "#b45309", "#0f766e", "#2563eb", "#7c3aed", "#db2777", "#059669", "#dc2626"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, paths: Paths, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(paths.results / "figures" / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def capture_figure(paths: Paths) -> None:
    metrics = pd.read_csv(paths.results / "strategy_metrics.csv")
    metrics = metrics.loc[metrics["strategy"].ne("perfect_foresight")]
    strategies = list(LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    width = 0.8 / len(strategies)
    for i, strategy in enumerate(strategies):
        values = (
            metrics.loc[metrics["strategy"].eq(strategy)].set_index("region").loc[list(REGIONS)]
        )
        ax.bar(
            np.arange(len(REGIONS)) + i * width - 0.4 + width / 2,
            values["capture_of_perfect_foresight"] * 100,
            width,
            label=LABELS[strategy],
            color=COLOURS[i],
        )
    ax.set_xticks(np.arange(len(REGIONS)), REGIONS)
    ax.set_ylabel("Share of perfect-foresight value (%)")
    ax.set_xlabel("NEM region")
    ax.legend(ncols=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _save(fig, paths, "fig1_value_capture")


def cumulative_figure(paths: Paths) -> None:
    daily = pd.read_parquet(paths.predictions / "daily_pnl.parquet")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    axes = axes.ravel()
    shown = ["perfect_foresight", "fixed_schedule", "naive_d1", "lgbm_quantile_mean", "lgbm_cvar"]
    palette = ["#111827", "#6b7280", "#b45309", "#db2777", "#dc2626"]
    for ax, region in zip(axes, REGIONS, strict=False):
        frame = daily.loc[daily["region"].eq(region)]
        for strategy, colour in zip(shown, palette, strict=True):
            series = frame.loc[frame["strategy"].eq(strategy)].sort_values("day")
            ax.plot(
                series["day"], series["pnl"].cumsum() / 1000, color=colour, lw=1, label=strategy
            )
        ax.set_title(region, fontsize=9)
        ax.set_ylabel("Cumulative A$k per MW")
        ax.tick_params(axis="x", labelrotation=30)
    handles, names = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, names, frameon=False, fontsize=8, loc="center")
    _save(fig, paths, "fig2_cumulative_value")


def coverage_figure(paths: Paths) -> None:
    coverage = pd.read_csv(paths.results / "interval_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, nominal in zip(axes, (0.8, 0.9), strict=True):
        frame = coverage.loc[coverage["nominal"].round(2).eq(nominal)]
        for offset, method, colour in (
            (-0.18, "quantile_regression", "#6b7280"),
            (0.18, "conformal", "#0f766e"),
        ):
            values = frame.loc[frame["method"].eq(method)].set_index("region").loc[list(REGIONS)]
            ax.bar(
                np.arange(len(REGIONS)) + offset,
                values["coverage"] * 100,
                0.34,
                color=colour,
                label=method.replace("_", " "),
            )
        ax.axhline(nominal * 100, color="#111827", lw=1, ls="--", label="nominal")
        ax.set_xticks(np.arange(len(REGIONS)), REGIONS)
        ax.set_xlabel("NEM region")
        ax.set_title(f"{nominal:.0%} prediction interval", fontsize=9)
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_ylim(0, 100)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, paths, "fig3_interval_coverage")


def dispatch_figure(paths: Paths, config: ExperimentConfig) -> None:
    predictions = pd.read_parquet(paths.predictions / "NSW1_predictions.parquet")
    day = predictions["day"].max() - pd.Timedelta(days=7)
    frame = predictions.loc[predictions["day"].eq(day)].sort_values("period")
    model = DispatchModel(config.battery)
    plan = model.solve(frame["quantile_mean"].to_numpy())
    hours = (np.arange(PERIODS_PER_DAY) + 1) / 2
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4), sharex=True)
    axes[0].plot(hours, frame["price"], color="#111827", label="Realised price")
    axes[0].plot(hours, frame["quantile_mean"], color="#db2777", label="Forecast mean")
    axes[0].fill_between(
        hours,
        frame["conformal_q10"],
        frame["conformal_q90"],
        color="#db2777",
        alpha=0.15,
        label="Conformal 80% interval",
    )
    axes[0].set_ylabel("Price (A$/MWh)")
    axes[0].legend(frameon=False, ncols=3, fontsize=7)
    axes[0].set_title(f"NSW1, {day.date()}", fontsize=9)
    axes[1].bar(hours, plan.net_mw, width=0.45, color="#0f766e", label="Net discharge")
    axes[1].set_ylabel("Power (MW)")
    axes[1].set_xlabel("Hour ending (NEM time, UTC+10)")
    twin = axes[1].twinx()
    twin.plot(hours, plan.soc_mwh, color="#2563eb", label="State of charge")
    twin.set_ylabel("State of charge (MWh)")
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    twin.legend(frameon=False, fontsize=7, loc="upper right")
    _save(fig, paths, "fig4_example_dispatch")


def dashboard(paths: Paths) -> None:
    with duckdb.connect(str(paths.warehouse), read_only=True) as connection:
        monthly = connection.execute("SELECT * FROM mart_spread_trend ORDER BY month").df()
    spread = px.line(
        monthly,
        x="month",
        y="spread_12m_average",
        color="region_code",
        title="12-month average of daily top-2h minus bottom-2h price spread",
        labels={"spread_12m_average": "A$/MWh", "month": "Month", "region_code": "Region"},
    )
    negative = px.line(
        monthly,
        x="month",
        y="negative_price_share",
        color="region_code",
        title="Share of half-hours with negative prices",
        labels={"negative_price_share": "Share", "month": "Month", "region_code": "Region"},
    )
    metrics = pd.read_csv(paths.results / "strategy_metrics.csv")
    value = px.bar(
        metrics.loc[metrics["strategy"].ne("perfect_foresight")],
        x="region",
        y="annualised_aud_per_mw",
        color="strategy",
        barmode="group",
        title="Backtest value by strategy (A$ per MW-year, before market and network costs)",
        labels={"annualised_aud_per_mw": "A$ per MW-year", "region": "Region"},
    )
    parts = [
        figure.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, figure in enumerate((value, spread, negative))
    ]
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>NEM Battery Lab</title><style>body{font-family:system-ui;max-width:1100px;"
        "margin:auto;padding:16px}</style></head><body><h1>NEM Battery Lab</h1>"
        "<p>Built from public AEMO data. Backtest results exclude market impact, fees, "
        "FCAS revenue and network charges.</p>" + "".join(parts) + "</body></html>"
    )
    (paths.results / "dashboard.html").write_text(html, encoding="utf-8")


def build_report(paths: Paths, config: ExperimentConfig) -> None:
    _style()
    capture_figure(paths)
    cumulative_figure(paths)
    coverage_figure(paths)
    dispatch_figure(paths, config)
    dashboard(paths)
