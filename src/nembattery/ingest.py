"""Download AEMO aggregated price-and-demand files and validate them into half-hourly tables."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .config import REGIONS, Paths

URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{month}_{region}.csv"
REQUIRED = ["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE"]


@dataclass(frozen=True)
class QualityReport:
    region: str
    rows_5min: int
    duplicates_removed: int
    expected_half_hours: int
    missing_half_hours: int
    incomplete_half_hours: int
    min_price: float
    max_price: float


def month_range(first: str, last: str) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(first, last, freq="M")]


def download_file(region: str, month: str, target: Path, timeout: int = 60) -> Path:
    """Fetch one file. Existing files are never overwritten (raw data is append-only)."""
    path = target / f"PRICE_AND_DEMAND_{month}_{region}.csv"
    if path.exists() and path.stat().st_size > 0:
        return path
    response = requests.get(
        URL.format(month=month, region=region),
        headers={"User-Agent": "nem-battery-lab research client"},
        timeout=timeout,
    )
    response.raise_for_status()
    if not response.text.startswith("REGION,"):
        raise ValueError(f"Unexpected payload for {region} {month}")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(response.text, encoding="utf-8")
    temporary.replace(path)
    return path


def download_all(paths: Paths, first: str, last: str, workers: int = 4) -> list[Path]:
    paths.ensure()
    jobs = [(region, month) for region in REGIONS for month in month_range(first, last)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: download_file(job[0], job[1], paths.raw), jobs))


def read_raw(files: list[Path]) -> pd.DataFrame:
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        missing = sorted(set(REQUIRED) - set(frame.columns))
        if missing:
            raise ValueError(f"{file.name}: missing columns {missing}")
        frames.append(frame[REQUIRED])
    data = pd.concat(frames, ignore_index=True)
    data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    return data


def to_half_hourly(raw: pd.DataFrame, region: str) -> tuple[pd.DataFrame, QualityReport]:
    """Average 5-minute records into half-hours labelled by period end (NEM time, UTC+10)."""
    data = raw.loc[raw["REGION"].eq(region) & raw["PERIODTYPE"].eq("TRADE")].copy()
    before = len(data)
    data = data.drop_duplicates(["REGION", "SETTLEMENTDATE"], keep="last")
    duplicates = before - len(data)
    data["ts"] = data["SETTLEMENTDATE"].dt.ceil("30min")
    grouped = data.groupby("ts", sort=True).agg(
        price=("RRP", "mean"), demand_mw=("TOTALDEMAND", "mean"), n_obs=("RRP", "size")
    )
    index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="30min", name="ts")
    frame = grouped.reindex(index)
    missing = int(frame["price"].isna().sum())
    incomplete = int((frame["n_obs"].fillna(0) < 6).sum()) - missing
    frame[["price", "demand_mw"]] = frame[["price", "demand_mw"]].ffill(limit=4)
    frame["n_obs"] = frame["n_obs"].fillna(0).astype(int)
    frame = frame.dropna().reset_index().assign(region=region)
    report = QualityReport(
        region=region,
        rows_5min=len(data),
        duplicates_removed=duplicates,
        expected_half_hours=len(index),
        missing_half_hours=missing,
        incomplete_half_hours=incomplete,
        min_price=float(frame["price"].min()),
        max_price=float(frame["price"].max()),
    )
    return frame[["ts", "region", "price", "demand_mw", "n_obs"]], report


def build_processed(paths: Paths, first: str, last: str) -> pd.DataFrame:
    paths.ensure()
    frames, reports = [], []
    months = month_range(first, last)
    for region in REGIONS:
        files = [paths.raw / f"PRICE_AND_DEMAND_{month}_{region}.csv" for month in months]
        frame, report = to_half_hourly(read_raw(files), region)
        frames.append(frame)
        reports.append(report.__dict__)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(paths.processed / "half_hourly.parquet", index=False)
    pd.DataFrame(reports).to_csv(paths.results / "data_quality.csv", index=False)
    return data
