"""DuckDB star schema, analytical marts and BI-ready Parquet exports."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

from .config import Paths


def _sql(name: str) -> str:
    local = Path(__file__).resolve().parents[2] / "sql" / name
    if local.exists():
        return local.read_text(encoding="utf-8")
    return resources.files("nembattery").joinpath("sql", name).read_text(encoding="utf-8")


def build_warehouse(connection: duckdb.DuckDBPyConnection, data: pd.DataFrame) -> None:
    connection.register("src", data)
    connection.execute(_sql("schema.sql"))
    connection.execute(_sql("marts.sql"))
    connection.unregister("src")


def integrity_checks(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Each value must be zero; these are the warehouse's data tests."""
    queries = {
        "duplicate_fact_grain": """
            SELECT count(*) FROM (
                SELECT time_key, region_key FROM fact_market_interval
                GROUP BY 1, 2 HAVING count(*) > 1)""",
        "orphan_time_keys": """
            SELECT count(*) FROM fact_market_interval f
            ANTI JOIN dim_time t USING (time_key)""",
        "orphan_region_keys": """
            SELECT count(*) FROM fact_market_interval f
            ANTI JOIN dim_region r USING (region_key)""",
        "invalid_trading_period": """
            SELECT count(*) FROM dim_time WHERE trading_period NOT BETWEEN 1 AND 48""",
    }
    results: dict[str, int] = {}
    for name, sql in queries.items():
        row = connection.execute(sql).fetchone()
        results[name] = int(row[0]) if row else -1
    return results


def materialise(paths: Paths) -> dict[str, int]:
    data = pd.read_parquet(paths.processed / "half_hourly.parquet")
    paths.warehouse.unlink(missing_ok=True)
    with duckdb.connect(str(paths.warehouse)) as connection:
        build_warehouse(connection, data)
        checks = integrity_checks(connection)
        if any(checks.values()):
            raise RuntimeError(f"Warehouse integrity failure: {checks}")
        for table in ("dim_region", "dim_time", "fact_market_interval"):
            target = (paths.star / f"{table}.parquet").as_posix()
            connection.execute(f"COPY {table} TO '{target}' (FORMAT PARQUET)")
        monthly = connection.execute(
            "SELECT * FROM mart_spread_trend ORDER BY region_code, month"
        ).df()
    monthly.to_csv(paths.results / "monthly_market_mart.csv", index=False)
    return checks
