from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from invest_ai_core.manifest import attach_output_files, save_manifest
from us_invest_ai.config import RunConfig, load_config
from us_invest_ai.data import MarketDataBundle, prepare_market_data_bundle
from us_invest_ai.experiment_manifest import build_run_manifest


def refresh_market_data(
    config: RunConfig,
    *,
    data_dir: Path | None = None,
    prefer_cache: bool = False,
) -> tuple[MarketDataBundle, dict[str, Any], Path]:
    resolved_data_dir = data_dir.resolve() if data_dir is not None else config.output.data_dir
    market_data = prepare_market_data_bundle(
        data_dir=resolved_data_dir,
        tickers=config.data.tickers,
        benchmark=config.data.benchmark,
        start=config.data.start,
        end=config.data.end,
        tickers_file=config.data.tickers_file,
        metadata_file=config.data.metadata_file,
        universe_snapshots_file=config.data.universe_snapshots_file,
        prefer_cache=prefer_cache,
    )

    refresh_manifest_path = resolved_data_dir / "raw" / "refresh_run_manifest.json"
    prices_summary = market_data.provenance["prices_summary"]
    benchmark_summary = market_data.provenance["benchmark_summary"]
    manifest = build_run_manifest(
        config,
        experiment_name="refresh_market_data",
        extra={
            "resolved_data_dir": str(resolved_data_dir),
            "prefer_cache": prefer_cache,
            "market_data_source": market_data.provenance.get("source"),
            "latest_market_date": prices_summary.get("end_date"),
            "price_rows": prices_summary.get("rows"),
            "benchmark_rows": benchmark_summary.get("rows"),
        },
    )
    manifest = attach_output_files(
        manifest,
        {
            "prices": resolved_data_dir / "raw" / "prices.csv",
            "benchmark": resolved_data_dir / "raw" / "benchmark.csv",
            "market_data_manifest": resolved_data_dir / "raw" / "market_data_manifest.json",
        },
    )
    save_manifest(refresh_manifest_path, manifest)
    return market_data, manifest, refresh_manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the configured market-data bundle and record a dedicated manifest."
    )
    parser.add_argument(
        "--config",
        default="us_stocks/config/soft_price_large_cap_60_dynamic_eligibility.yaml",
        help="Base config used to resolve the data bundle location and universe.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional explicit data directory override.",
    )
    parser.add_argument(
        "--prefer-cache",
        action="store_true",
        help="Reuse a matching cache if present instead of forcing a refresh.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    market_data, _, refresh_manifest_path = refresh_market_data(
        config,
        data_dir=data_dir,
        prefer_cache=args.prefer_cache,
    )

    prices_summary = market_data.provenance["prices_summary"]
    benchmark_summary = market_data.provenance["benchmark_summary"]
    print(f"Data directory: {data_dir or config.output.data_dir}")
    print(f"Market data source: {market_data.provenance.get('source')}")
    print(f"Latest market date: {prices_summary.get('end_date')}")
    print(
        "Price rows: "
        f"{prices_summary.get('rows')} across {prices_summary.get('ticker_count')} tickers"
    )
    print(
        "Benchmark rows: "
        f"{benchmark_summary.get('rows')} across {benchmark_summary.get('ticker_count')} ticker"
    )
    print(f"Refresh manifest: {refresh_manifest_path}")


if __name__ == "__main__":
    main()
