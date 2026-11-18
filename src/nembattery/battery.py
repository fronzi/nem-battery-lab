"""Mixed-integer battery dispatch: deterministic and CVaR-aware scenario optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from .config import DT_HOURS, PERIODS_PER_DAY, BatteryConfig


@dataclass(frozen=True)
class Dispatch:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray

    @property
    def net_mw(self) -> np.ndarray:
        return self.discharge_mw - self.charge_mw


def settle(dispatch: Dispatch, price: np.ndarray, battery: BatteryConfig) -> float:
    """Daily cash flow at realised prices, net of throughput degradation cost."""
    energy = float(np.sum(price * dispatch.net_mw) * DT_HOURS)
    wear = float(dispatch.discharge_mw.sum() * DT_HOURS * battery.degradation_cost_per_mwh)
    return energy - wear


class DispatchModel:
    """Reusable constraint set: x = [charge, discharge, soc, is_charging]."""

    def __init__(self, battery: BatteryConfig, periods: int = PERIODS_PER_DAY) -> None:
        self.battery, self.t = battery, periods
        t = periods
        self.n = 4 * t
        eta = battery.one_way_efficiency
        rows, cols, values = [], [], []
        lower, upper = [], []
        row = 0

        def add(c: int, v: float) -> None:
            rows.append(row)
            cols.append(c)
            values.append(v)

        initial = battery.initial_soc_fraction * battery.energy_mwh
        for i in range(t):
            add(2 * t + i, 1.0)
            if i:
                add(2 * t + i - 1, -1.0)
            add(i, -eta * DT_HOURS)
            add(t + i, DT_HOURS / eta)
            rhs = initial if i == 0 else 0.0
            lower.append(rhs)
            upper.append(rhs)
            row += 1
        for i in range(t):
            add(i, 1.0)
            add(3 * t + i, -battery.power_mw)
            lower.append(-np.inf)
            upper.append(0.0)
            row += 1
            add(t + i, 1.0)
            add(3 * t + i, battery.power_mw)
            lower.append(-np.inf)
            upper.append(battery.power_mw)
            row += 1
        self.a = coo_matrix((values, (rows, cols)), shape=(row, self.n)).tocsr()
        self.lower, self.upper = np.array(lower), np.array(upper)
        lb = np.zeros(self.n)
        ub = np.concatenate(
            [
                np.full(2 * t, battery.power_mw),
                np.full(t, battery.energy_mwh),
                np.ones(t),
            ]
        )
        lb[3 * t - 1] = ub[3 * t - 1] = initial  # finish where the day started
        self.lb, self.ub = lb, ub
        self.integrality = np.concatenate([np.zeros(3 * t), np.ones(t)])

    def _unpack(self, x: np.ndarray) -> Dispatch:
        t = self.t
        clean = np.where(np.abs(x[: 3 * t]) < 1e-9, 0.0, x[: 3 * t])
        return Dispatch(clean[:t], clean[t : 2 * t], clean[2 * t : 3 * t])

    def solve(self, price: np.ndarray) -> Dispatch:
        price = np.asarray(price, dtype=float)
        if price.shape != (self.t,) or not np.isfinite(price).all():
            raise ValueError("price must be a finite vector with one value per period")
        cost = np.zeros(self.n)
        cost[: self.t] = price * DT_HOURS
        cost[self.t : 2 * self.t] = (-price + self.battery.degradation_cost_per_mwh) * DT_HOURS
        result = milp(
            cost,
            constraints=LinearConstraint(self.a, self.lower, self.upper),
            bounds=Bounds(self.lb, self.ub),
            integrality=self.integrality,
            options={"mip_rel_gap": 1e-4, "time_limit": 20},
        )
        if result.x is None:
            raise RuntimeError(f"Dispatch optimisation failed: {result.message}")
        return self._unpack(result.x)

    def solve_cvar(self, scenarios: np.ndarray, alpha: float, weight: float) -> Dispatch:
        """Maximise (1-w) E[R] + w CVaR_alpha[R] with the Rockafellar-Uryasev linearisation."""
        scenarios = np.asarray(scenarios, dtype=float)
        s, t = scenarios.shape[0], self.t
        if scenarios.shape[1] != t or not np.isfinite(scenarios).all():
            raise ValueError("scenarios must be finite with shape (n_scenarios, periods)")
        wear = self.battery.degradation_cost_per_mwh
        extra = 1 + s
        cost = np.zeros(self.n + extra)
        mean = scenarios.mean(axis=0)
        cost[:t] = (1 - weight) * mean * DT_HOURS
        cost[t : 2 * t] = (1 - weight) * (-mean + wear) * DT_HOURS
        cost[self.n] = -weight
        cost[self.n + 1 :] = weight / (alpha * s)
        # z_s >= eta - R_s  <=>  eta - z_s - R_s <= 0
        revenue = csr_matrix(
            np.hstack(
                [
                    scenarios * DT_HOURS,
                    (-scenarios + wear) * DT_HOURS,
                    np.zeros((s, 2 * t)),
                ]
            )
        )
        eta_column = csr_matrix(np.ones((s, 1)))
        z = csr_matrix(-np.eye(s))
        base = hstack([self.a, csr_matrix((self.a.shape[0], extra))], format="csr")
        tail = hstack([revenue, eta_column, z], format="csr")
        matrix = vstack([base, tail], format="csr")
        lower = np.concatenate([self.lower, np.full(s, -np.inf)])
        upper = np.concatenate([self.upper, np.zeros(s)])
        lb = np.concatenate([self.lb, [-np.inf], np.zeros(s)])
        ub = np.concatenate([self.ub, [np.inf], np.full(s, np.inf)])
        integrality = np.concatenate([self.integrality, np.zeros(extra)])
        result = milp(
            cost,
            constraints=LinearConstraint(matrix, lower, upper),
            bounds=Bounds(lb, ub),
            integrality=integrality,
            options={"mip_rel_gap": 1e-3, "time_limit": 20},
        )
        if result.x is None:
            raise RuntimeError(f"CVaR optimisation failed: {result.message}")
        return self._unpack(result.x)


def fixed_schedule(battery: BatteryConfig, periods: int = PERIODS_PER_DAY) -> Dispatch:
    """Transparent rule: charge around solar noon, discharge in the evening peak."""
    charge, discharge = np.zeros(periods), np.zeros(periods)
    eta = battery.one_way_efficiency
    stored = battery.energy_mwh * (1 - battery.initial_soc_fraction)
    charge_periods = int(np.ceil(stored / (battery.power_mw * eta * DT_HOURS)))
    start_charge = 24 - charge_periods // 2
    charge[start_charge : start_charge + charge_periods] = stored / (
        charge_periods * eta * DT_HOURS
    )
    delivered = stored * eta
    discharge_periods = int(np.ceil(delivered / (battery.power_mw * DT_HOURS)))
    discharge[36 : 36 + discharge_periods] = delivered / (discharge_periods * DT_HOURS)
    delta = eta * charge * DT_HOURS - discharge * DT_HOURS / eta
    soc = battery.initial_soc_fraction * battery.energy_mwh + np.cumsum(delta)
    return Dispatch(charge, discharge, soc)
