from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from nembattery.api import app
from nembattery.battery import DispatchModel, fixed_schedule, settle
from nembattery.config import DT_HOURS, BatteryConfig
from nembattery.features import FEATURES, build_region_features, price_inverse, price_transform
from nembattery.forecast import conformal_adjustment, pinball
from nembattery.ingest import to_half_hourly
from nembattery.monitoring import population_stability_index
from nembattery.risk import block_bootstrap_mean, diebold_mariano, holm_adjust, max_drawdown
from nembattery.warehouse import build_warehouse, integrity_checks

BATTERY = BatteryConfig()
MODEL = DispatchModel(BATTERY)


def panel(days: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    index = pd.date_range("2024-01-01 00:30", periods=48 * days, freq="30min")
    frames = []
    for offset, region in enumerate(("NSW1", "QLD1")):
        phase = 2 * np.pi * np.arange(len(index)) / 48
        price = 80 + offset + 30 * np.sin(phase) + rng.normal(0, 3, len(index))
        frames.append(
            pd.DataFrame(
                {
                    "ts": index,
                    "region": region,
                    "price": price,
                    "demand_mw": 7000 + price,
                    "n_obs": 6,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def check_feasible(plan, battery=BATTERY) -> None:
    eta = battery.one_way_efficiency
    soc = battery.initial_soc_fraction * battery.energy_mwh
    for c, d, reported in zip(plan.charge_mw, plan.discharge_mw, plan.soc_mwh, strict=True):
        assert min(c, d) <= 1e-7
        soc += eta * c * DT_HOURS - d * DT_HOURS / eta
        assert soc == pytest.approx(reported, abs=1e-6)
        assert -1e-7 <= soc <= battery.energy_mwh + 1e-7
    assert soc == pytest.approx(battery.initial_soc_fraction * battery.energy_mwh, abs=1e-6)


def test_known_two_price_arbitrage_answer() -> None:
    """Hand-computable optimum: start half full, buy 1 MWh at $0, sell 1 MWh at $100."""
    battery = BatteryConfig(round_trip_efficiency=1.0, degradation_cost_per_mwh=0.0)
    prices = np.zeros(48)
    prices[-4:] = 100.0
    plan = DispatchModel(battery).solve(prices)
    assert settle(plan, prices, battery) == pytest.approx(100.0, abs=1e-5)
    check_feasible(plan, battery)


def test_flat_prices_do_nothing_when_cycling_costs_money() -> None:
    plan = MODEL.solve(np.full(48, 50.0))
    assert plan.net_mw == pytest.approx(np.zeros(48), abs=1e-7)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.floats(-1000, 5000, allow_nan=False), min_size=48, max_size=48))
def test_dispatch_is_feasible_and_never_loses_to_idle(prices: list[float]) -> None:
    array = np.array(prices)
    plan = MODEL.solve(array)
    check_feasible(plan)
    assert settle(plan, array, BATTERY) >= -1e-3


def test_fixed_schedule_is_feasible() -> None:
    check_feasible(fixed_schedule(BATTERY))


def test_cvar_with_identical_scenarios_matches_deterministic_value() -> None:
    prices = 50 + 40 * np.sin(np.arange(48) / 48 * 2 * np.pi)
    deterministic = settle(MODEL.solve(prices), prices, BATTERY)
    robust = settle(MODEL.solve_cvar(np.tile(prices, (5, 1)), 0.2, 0.5), prices, BATTERY)
    assert robust == pytest.approx(deterministic, rel=2e-3)


def test_features_do_not_look_ahead() -> None:
    data = panel()
    base = build_region_features(data, "NSW1")
    target = pd.Timestamp("2024-01-15")
    changed = data.copy()
    start = changed["ts"] - pd.Timedelta(minutes=30)
    changed.loc[start.dt.normalize().eq(target), ["price", "demand_mw"]] += 10_000
    after = build_region_features(changed, "NSW1")
    left = base.loc[base["day"].eq(target), FEATURES].reset_index(drop=True)
    right = after.loc[after["day"].eq(target), FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    next_day = target + pd.Timedelta(days=1)
    assert not base.loc[base["day"].eq(next_day), "price_d1"].equals(
        after.loc[after["day"].eq(next_day), "price_d1"]
    )


def test_price_transform_round_trip() -> None:
    values = np.array([-1000.0, -5.0, 0.0, 80.0, 17500.0])
    assert price_inverse(price_transform(values)) == pytest.approx(values)


def test_half_hour_aggregation_and_duplicates() -> None:
    times = pd.date_range("2024-01-01 00:05", periods=12, freq="5min")
    raw = pd.DataFrame(
        {
            "REGION": "NSW1",
            "SETTLEMENTDATE": [*list(times), times[0]],
            "TOTALDEMAND": [100.0] * 13,
            "RRP": [10.0] * 6 + [40.0] * 6 + [10.0],
            "PERIODTYPE": "TRADE",
        }
    )
    frame, report = to_half_hourly(raw, "NSW1")
    assert frame["price"].tolist() == [10.0, 40.0]
    assert report.duplicates_removed == 1
    assert report.missing_half_hours == 0


def test_warehouse_grain_and_marts() -> None:
    data = panel(3)
    with duckdb.connect(":memory:") as connection:
        build_warehouse(connection, data)
        assert not any(integrity_checks(connection).values())
        rows = connection.execute(
            "SELECT count(*), min(periods), max(periods) FROM mart_daily_market"
        ).fetchone()
    assert rows == (6, 48, 48)


def test_conformal_score_has_finite_sample_quantile() -> None:
    y = np.arange(1.0, 11.0)
    assert conformal_adjustment(y, np.zeros(10), np.zeros(10), 0.2) == 9.0


def test_pinball_is_minimised_at_empirical_quantile() -> None:
    y = np.arange(101.0)
    losses = [pinball(y, np.full(101, c), 0.9).mean() for c in (70.0, 90.0, 99.0)]
    assert np.argmin(losses) == 1


def test_risk_utilities() -> None:
    assert max_drawdown(np.array([1.0, -3.0, 2.0])) == 3.0
    assert holm_adjust(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert block_bootstrap_mean(np.ones(30), draws=20).tolist() == [1.0] * 20
    statistic, p_value = diebold_mariano(np.ones(50), np.ones(50) * 2)
    assert (statistic, p_value) == (0.0, 1.0)


def test_psi_detects_shift_only_when_present() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=5000)
    assert population_stability_index(reference, rng.normal(size=5000)) < 0.02
    assert population_stability_index(reference, rng.normal(2, 1, 5000)) > 0.25


def test_api_dispatch_contract() -> None:
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    prices = [0.0] * 44 + [200.0] * 4
    response = client.post("/dispatch", json={"prices": prices})
    assert response.status_code == 200
    assert response.json()["expected_pnl_aud"] > 0
    assert client.post("/dispatch", json={"prices": [1.0]}).status_code == 422
    assert client.get("/forecast/XXX").status_code == 404
