from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from us_invest_ai.data import prepare_market_data_bundle
from us_invest_ai.experiment_manifest import (
    attach_output_files,
    save_manifest,
    sha256_file,
    sidecar_manifest_path,
)


SNAPSHOT_COLUMNS = [
    "effective_date",
    "ticker",
    "selection_rank",
    "selection_score",
    "avg_dollar_volume_60",
    "close",
    "universe_age_days",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly free-approx dynamic universe snapshots from trailing liquidity."
    )
    parser.add_argument(
        "--candidate-tickers-file",
        default="us_stocks/universes/liquid_large_cap_60.txt",
        help="Ticker file used as the candidate pool.",
    )
    parser.add_argument(
        "--metadata-file",
        default="us_stocks/universes/liquid_large_cap_60_metadata.csv",
        help="Optional ticker metadata CSV kept in the snapshot manifest.",
    )
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Inclusive market-data start date.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional exclusive market-data end date.",
    )
    parser.add_argument(
        "--snapshot-size",
        type=int,
        default=30,
        help="Maximum number of names kept at each monthly snapshot.",
    )
    parser.add_argument(
        "--min-close-price",
        type=float,
        default=5.0,
        help="Minimum close price at the snapshot date.",
    )
    parser.add_argument(
        "--min-dollar-volume-60",
        type=float,
        default=50_000_000.0,
        help="Minimum trailing 60-day average dollar volume at the snapshot date.",
    )
    parser.add_argument(
        "--min-universe-age-days",
        type=int,
        default=252,
        help="Minimum age in calendar days since the ticker first appears in the downloaded history.",
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker used only to reuse the shared market-data bundle path.",
    )
    parser.add_argument(
        "--data-dir",
        default="us_stocks/data_large_cap_60_dynamic_eligibility",
        help="Data directory used for cached raw market data.",
    )
    parser.add_argument(
        "--output-csv",
        default="us_stocks/universes/generated/liquid_large_cap_60_dynamic_snapshots.csv",
        help="Snapshot CSV output path.",
    )
    return parser.parse_args()


def load_candidate_tickers(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in raw.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        tokens.extend(part.strip().upper() for part in stripped.replace(",", " ").split())

    tickers = [ticker for ticker in tokens if ticker]
    if not tickers:
        raise ValueError(f"No candidate tickers found in {path}")
    return tickers


def _monthly_snapshot_dates(prices: pd.DataFrame) -> list[pd.Timestamp]:
    normalized_dates = pd.to_datetime(prices["date"]).dt.normalize()
    monthly = pd.DataFrame({"date": normalized_dates.drop_duplicates().sort_values()})
    if monthly.empty:
        return []
    monthly["month"] = monthly["date"].dt.to_period("M")
    return [
        pd.Timestamp(date).normalize()
        for date in monthly.groupby("month", sort=True)["date"].max().tolist()
    ]


def build_dynamic_universe_snapshots(
    prices: pd.DataFrame,
    *,
    snapshot_size: int = 30,
    min_close_price: float = 5.0,
    min_dollar_volume_60: float = 50_000_000.0,
    min_universe_age_days: int = 252,
) -> pd.DataFrame:
    if snapshot_size <= 0:
        raise ValueError("snapshot_size must be positive.")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    dollar_volume = frame["close"] * frame["volume"]
    frame["avg_dollar_volume_60"] = dollar_volume.groupby(frame["ticker"]).transform(
        lambda series: series.rolling(60, min_periods=60).mean()
    )
    frame["first_seen_date"] = frame.groupby("ticker")["date"].transform("min")
    frame["universe_age_days"] = (
        frame["date"] - frame["first_seen_date"]
    ).dt.days.astype(float)

    snapshot_parts: list[pd.DataFrame] = []
    for snapshot_date in _monthly_snapshot_dates(frame):
        rows = frame.loc[
            frame["date"] == snapshot_date,
            ["date", "ticker", "close", "avg_dollar_volume_60", "universe_age_days"],
        ].copy()
        if rows.empty:
            continue

        eligible = rows.loc[rows["close"].fillna(0.0) >= float(min_close_price)].copy()
        eligible = eligible.loc[
            eligible["avg_dollar_volume_60"].fillna(0.0) >= float(min_dollar_volume_60)
        ].copy()
        eligible = eligible.loc[
            eligible["universe_age_days"].fillna(-1.0) >= float(min_universe_age_days)
        ].copy()
        if eligible.empty:
            continue

        eligible = eligible.sort_values(
            ["avg_dollar_volume_60", "ticker"],
            ascending=[False, True],
        ).head(snapshot_size)
        eligible["effective_date"] = snapshot_date
        eligible["selection_rank"] = range(1, len(eligible) + 1)
        eligible["selection_score"] = eligible["avg_dollar_volume_60"]
        snapshot_parts.append(
            eligible[
                [
                    "effective_date",
                    "ticker",
                    "selection_rank",
                    "selection_score",
                    "avg_dollar_volume_60",
                    "close",
                    "universe_age_days",
                ]
            ].copy()
        )

    if not snapshot_parts:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    snapshots = pd.concat(snapshot_parts, ignore_index=True)
    snapshots["effective_date"] = pd.to_datetime(snapshots["effective_date"]).dt.normalize()
    return snapshots.sort_values(["effective_date", "selection_rank", "ticker"]).reset_index(drop=True)


def build_snapshot_manifest(
    *,
    candidate_tickers_file: Path,
    metadata_file: Path | None,
    start: str,
    end: str | None,
    snapshot_size: int,
    min_close_price: float,
    min_dollar_volume_60: float,
    min_universe_age_days: int,
    market_data_provenance: dict[str, Any],
    snapshot_frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "free_approx_dynamic_universe_snapshots_v1",
        "query": {
            "start": start,
            "end": end,
        },
        "parameters": {
            "rebalance": "monthly",
            "snapshot_size": int(snapshot_size),
            "min_close_price": float(min_close_price),
            "min_dollar_volume_60": float(min_dollar_volume_60),
            "min_universe_age_days": int(min_universe_age_days),
        },
        "inputs": {
            "candidate_tickers_file_path": str(candidate_tickers_file),
            "candidate_tickers_file_sha256": sha256_file(candidate_tickers_file),
            "metadata_file_path": str(metadata_file) if metadata_file is not None else None,
            "metadata_file_sha256": sha256_file(metadata_file),
            "market_data_manifest_path": market_data_provenance.get("manifest_path"),
            "market_data_manifest_sha256": market_data_provenance.get("manifest_sha256"),
        },
        "market_data": {
            "source": market_data_provenance.get("source"),
            "prices_summary": market_data_provenance.get("prices_summary"),
            "benchmark_summary": market_data_provenance.get("benchmark_summary"),
        },
        "snapshot_summary": {
            "snapshot_count": int(snapshot_frame["effective_date"].nunique()) if not snapshot_frame.empty else 0,
            "rows": int(len(snapshot_frame)),
            "unique_tickers": int(snapshot_frame["ticker"].nunique()) if not snapshot_frame.empty else 0,
            "first_effective_date": (
                pd.to_datetime(snapshot_frame["effective_date"]).min().date().isoformat()
                if not snapshot_frame.empty
                else None
            ),
            "last_effective_date": (
                pd.to_datetime(snapshot_frame["effective_date"]).max().date().isoformat()
                if not snapshot_frame.empty
                else None
            ),
        },
    }


def main() -> None:
    args = _parse_args()
    candidate_tickers_file = Path(args.candidate_tickers_file).resolve()
    metadata_file = Path(args.metadata_file).resolve() if args.metadata_file else None
    data_dir = Path(args.data_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    candidate_tickers = load_candidate_tickers(candidate_tickers_file)
    market_data = prepare_market_data_bundle(
        data_dir=data_dir,
        tickers=candidate_tickers,
        benchmark=str(args.benchmark).upper(),
        start=str(args.start),
        end=args.end,
        tickers_file=candidate_tickers_file,
        metadata_file=metadata_file,
        universe_snapshots_file=None,
    )
    snapshots = build_dynamic_universe_snapshots(
        market_data.prices,
        snapshot_size=args.snapshot_size,
        min_close_price=args.min_close_price,
        min_dollar_volume_60=args.min_dollar_volume_60,
        min_universe_age_days=args.min_universe_age_days,
    )
    snapshots.to_csv(output_csv, index=False)

    manifest = build_snapshot_manifest(
        candidate_tickers_file=candidate_tickers_file,
        metadata_file=metadata_file,
        start=str(args.start),
        end=args.end,
        snapshot_size=args.snapshot_size,
        min_close_price=args.min_close_price,
        min_dollar_volume_60=args.min_dollar_volume_60,
        min_universe_age_days=args.min_universe_age_days,
        market_data_provenance=market_data.provenance,
        snapshot_frame=snapshots,
    )
    manifest = attach_output_files(
        manifest,
        {
            "snapshot_csv": output_csv,
            "market_data_manifest": market_data.provenance.get("manifest_path"),
        },
    )
    manifest_path = sidecar_manifest_path(output_csv)
    if manifest_path is None:
        raise ValueError("Could not derive a sidecar manifest path for the snapshot output.")
    save_manifest(manifest_path, manifest)

    print(f"Saved {len(snapshots)} snapshot rows across {snapshots['effective_date'].nunique() if not snapshots.empty else 0} dates.")
    print(f"Snapshot CSV: {output_csv}")
    print(f"Snapshot manifest: {manifest_path}")


if __name__ == "__main__":
    main()
