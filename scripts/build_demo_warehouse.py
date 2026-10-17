"""Build the self-contained demo warehouse from the committed sample Parquet files.

    uv run python scripts/build_demo_warehouse.py [--out data/demo_warehouse.duckdb]

The samples under data/sample/ are a frozen 12-day export (2026-10-03 to 2026-10-14) of the marts built by
the elt-modern-data-stack project from NOAA GHCN-Daily (public domain). Re-export with --from-warehouse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
TABLES = [
    "fct_station_day",
    "dim_station",
    "mart_daily_state_summary",
    "mart_florida_weather",
    "mart_station_coverage",
]


def export_samples(source: Path) -> None:
    """Pull the marts out of a real P1 warehouse into Parquet samples (run once, commit the files)."""
    SAMPLE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(source), read_only=True)
    try:
        for t in TABLES:
            cast = (
                "select * replace (cast(stations_with_rain as bigint) as stations_with_rain)"
                if t == "mart_daily_state_summary"
                else "select *"
            )
            if t == "mart_station_coverage":
                cast = "select * replace (cast(days_with_tmax as bigint) as days_with_tmax, cast(days_with_prcp as bigint) as days_with_prcp)"
            if t == "dim_station":
                cast = "select * from main_marts.dim_station where is_current"
            else:
                cast = f"{cast} from main_marts.{t}"
            out = SAMPLE / f"{t}.parquet"
            con.execute(f"copy ({cast}) to '{out.as_posix()}' (format parquet, compression zstd)")
            n = con.execute(f"select count(*) from read_parquet('{out.as_posix()}')").fetchone()
            print(f"{t}: {n[0] if n else 0:,} rows -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    finally:
        con.close()


def build(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    try:
        con.execute("create schema if not exists main_marts")
        for t in TABLES:
            src = SAMPLE / f"{t}.parquet"
            if not src.exists():
                raise SystemExit(f"missing sample {src}; run with --from-warehouse <p1 warehouse.duckdb>")
            con.execute(f"create table main_marts.{t} as select * from read_parquet('{src.as_posix()}')")
            n = con.execute(f"select count(*) from main_marts.{t}").fetchone()
            print(f"main_marts.{t}: {n[0] if n else 0:,} rows")
    finally:
        con.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo_warehouse.duckdb")
    ap.add_argument(
        "--from-warehouse", type=Path, default=None, help="Re-export the samples from a P1 warehouse first"
    )
    args = ap.parse_args()
    if args.from_warehouse:
        export_samples(args.from_warehouse)
    build(args.out)


if __name__ == "__main__":
    main()
