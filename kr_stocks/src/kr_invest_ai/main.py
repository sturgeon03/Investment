from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from invest_ai_core.manifest import attach_output_files, save_manifest, sha256_file
from kr_invest_ai.dart_client import DARTOpenAPIClient
from kr_invest_ai.data_bundle import KRResearchDataRequest
from kr_invest_ai.pipeline import load_corp_codes_csv, run_kr_data_pipeline


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first Korea-market data bundle pipeline.")
    parser.add_argument("--tickers", nargs="+", required=True, help="One or more KR tickers or listing codes.")
    parser.add_argument("--price-start-date", required=True, help="Price window start date in YYYY-MM-DD.")
    parser.add_argument("--price-end-date", required=True, help="Price window end date in YYYY-MM-DD.")
    parser.add_argument("--benchmark-ticker", default=None, help="Optional KR benchmark ticker or listing code.")
    parser.add_argument("--filings-start-date", default=None, help="Optional DART window start date in YYYY-MM-DD.")
    parser.add_argument("--filings-end-date", default=None, help="Optional DART window end date in YYYY-MM-DD.")
    parser.add_argument("--corp-code-map-csv", default=None, help="Optional CSV with columns ticker,corp_code.")
    parser.add_argument("--data-dir", default="kr_stocks/data", help="Directory for cached raw data.")
    parser.add_argument("--artifacts-dir", default="kr_stocks/artifacts", help="Directory for run manifests.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache reuse for this run.")
    parser.add_argument("--use-dart", action="store_true", help="Enable DART fetch when corp codes and dates are provided.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    corp_codes = load_corp_codes_csv(args.corp_code_map_csv)
    dart_client = None
    if args.use_dart:
        dart_client = DARTOpenAPIClient()

    request = KRResearchDataRequest(
        tickers=tuple(args.tickers),
        price_start_date=date.fromisoformat(args.price_start_date),
        price_end_date=date.fromisoformat(args.price_end_date),
        benchmark_ticker=args.benchmark_ticker,
        filings_start_date=_parse_date(args.filings_start_date),
        filings_end_date=_parse_date(args.filings_end_date),
    )
    run = run_kr_data_pipeline(
        request,
        data_dir=args.data_dir,
        corp_codes_by_ticker=corp_codes,
        dart_client=dart_client,
        use_cache=not args.no_cache,
    )

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "pipeline": "kr_research_data_pipeline",
        "from_cache": run.from_cache,
        "tickers": list(request.tickers),
        "benchmark_ticker": request.benchmark_ticker,
        "price_rows": len(run.bundle.prices),
        "benchmark_rows": len(run.bundle.benchmark_prices),
        "filing_rows": len(run.bundle.filings),
        "raw_manifest_path": str(run.manifest_path),
        "raw_manifest_sha256": sha256_file(run.manifest_path),
        "corp_code_map_csv": str(Path(args.corp_code_map_csv).resolve()) if args.corp_code_map_csv else None,
        "corp_code_map_sha256": sha256_file(args.corp_code_map_csv),
    }
    save_manifest(
        artifacts_dir / "run_manifest.json",
        attach_output_files(
            run_manifest,
            {
                "prices": run.prices_path,
                "benchmark": run.benchmark_path,
                "filings": run.filings_path,
                "raw_manifest": run.manifest_path,
            },
        ),
    )

    print(f"From cache: {run.from_cache}")
    print(f"Price rows: {len(run.bundle.prices)}")
    print(f"Benchmark rows: {len(run.bundle.benchmark_prices)}")
    print(f"Filing rows: {len(run.bundle.filings)}")
    print(f"Saved raw outputs to: {run.raw_dir}")
    print(f"Saved run manifest to: {artifacts_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
