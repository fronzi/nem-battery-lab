"""Central configuration. Every modelling assumption lives here and is written to the manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
PERIODS_PER_DAY = 48
DT_HOURS = 0.5


@dataclass(frozen=True)
class BatteryConfig:
    """Normalised 1 MW / 2 MWh price-taking battery."""

    power_mw: float = 1.0
    energy_mwh: float = 2.0
    round_trip_efficiency: float = 0.85
    initial_soc_fraction: float = 0.5
    degradation_cost_per_mwh: float = 10.0

    @property
    def one_way_efficiency(self) -> float:
        return self.round_trip_efficiency**0.5


@dataclass(frozen=True)
class ExperimentConfig:
    first_month: str = "2022-01"
    last_month: str = "2026-10"
    test_start: str = "2024-01-01"
    calibration_days: int = 90
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    lgbm_estimators: int = 150
    scenarios: int = 30
    cvar_alpha: float = 0.05
    cvar_weight: float = 0.5
    seed: int = 20261118
    battery: BatteryConfig = field(default_factory=BatteryConfig)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def warehouse(self) -> Path:
        return self.root / "data" / "warehouse.duckdb"

    @property
    def star(self) -> Path:
        return self.root / "data" / "star_schema"

    @property
    def predictions(self) -> Path:
        return self.root / "artifacts" / "predictions"

    @property
    def models(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def ensure(self) -> None:
        for path in (
            self.raw,
            self.processed,
            self.star,
            self.predictions,
            self.models,
            self.results / "figures",
        ):
            path.mkdir(parents=True, exist_ok=True)
