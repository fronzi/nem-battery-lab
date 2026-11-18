"""FastAPI service: optimise a dispatch plan and serve next-day price forecasts."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .battery import DispatchModel, settle
from .config import PERIODS_PER_DAY, REGIONS, BatteryConfig, Paths
from .features import build_region_features, price_inverse
from .forecast import predict_quantiles, qname

app = FastAPI(title="NEM Battery Lab", version=__version__)


class DispatchRequest(BaseModel):
    prices: list[float] = Field(min_length=PERIODS_PER_DAY, max_length=PERIODS_PER_DAY)
    round_trip_efficiency: float = Field(default=0.85, gt=0, le=1)
    degradation_cost_per_mwh: float = Field(default=10.0, ge=0)


class DispatchResponse(BaseModel):
    charge_mw: list[float]
    discharge_mw: list[float]
    soc_mwh: list[float]
    expected_pnl_aud: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/dispatch", response_model=DispatchResponse)
def dispatch(request: DispatchRequest) -> DispatchResponse:
    battery = BatteryConfig(
        round_trip_efficiency=request.round_trip_efficiency,
        degradation_cost_per_mwh=request.degradation_cost_per_mwh,
    )
    prices = np.array(request.prices)
    plan = DispatchModel(battery).solve(prices)
    return DispatchResponse(
        charge_mw=plan.charge_mw.round(6).tolist(),
        discharge_mw=plan.discharge_mw.round(6).tolist(),
        soc_mwh=plan.soc_mwh.round(6).tolist(),
        expected_pnl_aud=round(settle(plan, prices, battery), 4),
    )


@app.get("/forecast/{region}")
def forecast(region: str) -> dict:
    region = region.upper()
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail="Unknown NEM region")
    paths = Paths(Path(os.getenv("NEM_LAB_ROOT", ".")))
    bundle_path = paths.models / f"{region}.joblib"
    data_path = paths.processed / "half_hourly.parquet"
    if not bundle_path.exists() or not data_path.exists():
        raise HTTPException(status_code=503, detail="Model bundle or data not available")
    bundle = joblib.load(bundle_path)
    panel = pd.read_parquet(data_path)
    last_complete = (panel["ts"].max() - pd.Timedelta(minutes=30)).normalize()
    target = last_complete + pd.Timedelta(days=1)
    features = build_region_features(panel, region, target_day=target)
    rows = features.loc[features["day"].eq(target)].sort_values("period")
    if len(rows) != PERIODS_PER_DAY:
        raise HTTPException(status_code=503, detail="Insufficient recent data")
    prediction = predict_quantiles(bundle["models"], rows)
    return {
        "region": region,
        "trading_day": str(target.date()),
        "model_trained_to": bundle["trained_to"],
        "quantiles_aud_per_mwh": {
            qname(q): np.round(price_inverse(values), 2).tolist()
            for q, values in prediction.items()
        },
    }
