"""Command line entry point: ``nem-lab all`` reproduces the published results."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import ExperimentConfig, Paths
from .ingest import build_processed, download_all
from .pipeline import run_backtests, run_forecasts, summarise, train_final_models
from .report import build_report
from .warehouse import materialise

app = typer.Typer(no_args_is_help=True, add_completion=False)
CONFIG = ExperimentConfig()


def _paths(root: Path) -> Paths:
    paths = Paths(root)
    paths.ensure()
    return paths


@app.command()
def data(root: Path = Path(".")) -> None:
    """Download raw AEMO files, validate them and build the warehouse."""
    paths = _paths(root)
    download_all(paths, CONFIG.first_month, CONFIG.last_month)
    build_processed(paths, CONFIG.first_month, CONFIG.last_month)
    typer.echo(materialise(paths))


@app.command()
def forecast(root: Path = Path("."), jobs: int = 5) -> None:
    """Run the walk-forward forecasts."""
    run_forecasts(_paths(root), CONFIG, jobs)


@app.command()
def backtest(root: Path = Path("."), jobs: int = 5) -> None:
    """Optimise and settle every strategy, then write all result tables and figures."""
    paths = _paths(root)
    run_backtests(paths, CONFIG, jobs)
    summarise(paths, CONFIG)
    build_report(paths, CONFIG)


@app.command("train-final")
def train_final(root: Path = Path(".")) -> None:
    """Train the model bundles used by the API."""
    train_final_models(_paths(root), CONFIG)


@app.command("all")
def run_all(root: Path = Path("."), jobs: int = 5) -> None:
    """Run the complete pipeline."""
    data(root)
    forecast(root, jobs)
    backtest(root, jobs)
    train_final(root)
