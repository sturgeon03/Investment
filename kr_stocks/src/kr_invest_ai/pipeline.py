from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from invest_ai_core.manifest import attach_output_files, save_manifest, sha256_file, sha256_payload
from kr_invest_ai.data_bundle import KRResearchDataBundle, KRResearchDataRequest, build_kr_research_data_bundle
from kr_invest_ai.dart_client import DARTOpenAPIClient
from kr_invest_ai.market_config import KRMarketAdapterConfig, load_kr_market_adapter_config
from kr_invest_ai.market_data_client import KRHistoricalMarketDataClient


@dataclass(slots=True)
class KRPipelineRun:
    bundle: KRResearchDataBundle
    raw_dir: Path
    manifest_path: Path
    prices_path: Path
    benchmark_path: Path
    filings_path: Path
    from_cache: bool
    manifest: dict[str, Any]


def load_corp_codes_csv(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"ticker", "corp_code"}
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise ValueError(f"Corp code map is missing required columns: {missing}")
    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        ticker = str(row["ticker"]).strip()
        corp_code = str(row["corp_code"]).strip()
        if ticker and corp_code:
            mapping[ticker] = corp_code
    return mapping


def build_request_signature(
    request: KRResearchDataRequest,
    *,
    corp_codes_by_ticker: dict[str, str] | None = None,
    dart_enabled: bool,
    market_data_provider: str | None,
) -> str:
    return sha256_payload(
        {
            "request": request,
            "corp_codes_by_ticker": dict(sorted((corp_codes_by_ticker or {}).items())),
            "dart_enabled": dart_enabled,
            "market_data_provider": market_data_provider,
        }
    )


def load_cached_bundle(
    data_dir: str | Path,
    *,
    request_signature: str,
) -> KRPipelineRun | None:
    raw_dir = Path(data_dir) / "raw"
    prices_path = raw_dir / "prices.csv"
    benchmark_path = raw_dir / "benchmark.csv"
    filings_path = raw_dir / "filings.csv"
    manifest_path = raw_dir / "kr_market_data_manifest.json"
    if not (manifest_path.exists() and prices_path.exists() and filings_path.exists() and benchmark_path.exists()):
        return None

    cached_manifest = _load_manifest(manifest_path)
    if cached_manifest.get("request_signature") != request_signature:
        return None

    bundle = KRResearchDataBundle(
        prices=_load_prices(prices_path),
        benchmark_prices=_load_prices(benchmark_path),
        filings=_load_filings(filings_path),
        price_bar_count=int(cached_manifest.get("price_bar_count", 0)),
        benchmark_bar_count=int(cached_manifest.get("benchmark_bar_count", 0)),
        filing_count=int(cached_manifest.get("filing_count", 0)),
        provenance=dict(cached_manifest.get("bundle_provenance", {})),
    )
    return KRPipelineRun(
        bundle=bundle,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        prices_path=prices_path,
        benchmark_path=benchmark_path,
        filings_path=filings_path,
        from_cache=True,
        manifest=cached_manifest,
    )


def save_bundle_outputs(
    bundle: KRResearchDataBundle,
    *,
    data_dir: str | Path,
    request: KRResearchDataRequest,
    request_signature: str,
    corp_codes_by_ticker: dict[str, str] | None = None,
    dart_enabled: bool,
    market_data_provider: str | None,
) -> KRPipelineRun:
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prices_path = raw_dir / "prices.csv"
    benchmark_path = raw_dir / "benchmark.csv"
    filings_path = raw_dir / "filings.csv"
    manifest_path = raw_dir / "kr_market_data_manifest.json"
    ticker_conventions_path = Path(__file__).resolve().parents[2] / "config" / "ticker_conventions.yaml"

    bundle.prices.to_csv(prices_path, index=False)
    bundle.benchmark_prices.to_csv(benchmark_path, index=False)
    bundle.filings.to_csv(filings_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_signature": request_signature,
        "request": {
            "tickers": list(request.tickers),
            "price_start_date": request.price_start_date.isoformat(),
            "price_end_date": request.price_end_date.isoformat(),
            "benchmark_ticker": request.benchmark_ticker,
            "filings_start_date": request.filings_start_date.isoformat() if request.filings_start_date else None,
            "filings_end_date": request.filings_end_date.isoformat() if request.filings_end_date else None,
            "default_vendor_suffix": request.default_vendor_suffix,
        },
        "corp_codes_by_ticker": dict(sorted((corp_codes_by_ticker or {}).items())),
        "price_bar_count": bundle.price_bar_count,
        "benchmark_bar_count": bundle.benchmark_bar_count,
        "filing_count": bundle.filing_count,
        "bundle_provenance": bundle.provenance,
        "inputs": {
            "dart_enabled": dart_enabled,
            "market_data_provider": market_data_provider,
            "ticker_conventions_path": str(ticker_conventions_path),
            "ticker_conventions_sha256": sha256_file(ticker_conventions_path),
        },
    }
    save_manifest(
        manifest_path,
        attach_output_files(
            manifest,
            {
                "prices": prices_path,
                "benchmark": benchmark_path,
                "filings": filings_path,
            },
        ),
    )
    return KRPipelineRun(
        bundle=bundle,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        prices_path=prices_path,
        benchmark_path=benchmark_path,
        filings_path=filings_path,
        from_cache=False,
        manifest=_load_manifest(manifest_path),
    )


def run_kr_data_pipeline(
    request: KRResearchDataRequest,
    *,
    data_dir: str | Path = "kr_stocks/data",
    corp_codes_by_ticker: dict[str, str] | None = None,
    market_data_client: KRHistoricalMarketDataClient | None = None,
    dart_client: DARTOpenAPIClient | None = None,
    adapter_config: KRMarketAdapterConfig | None = None,
    holiday_dates: set[date] | None = None,
    use_cache: bool = True,
) -> KRPipelineRun:
    config = adapter_config or load_kr_market_adapter_config()
    market_data_provider = getattr(market_data_client, "provider_name", None)
    request_signature = build_request_signature(
        request,
        corp_codes_by_ticker=corp_codes_by_ticker,
        dart_enabled=dart_client is not None,
        market_data_provider=market_data_provider,
    )

    if use_cache:
        cached_run = load_cached_bundle(data_dir, request_signature=request_signature)
        if cached_run is not None:
            return cached_run

    bundle = build_kr_research_data_bundle(
        request,
        corp_codes_by_ticker=corp_codes_by_ticker,
        market_data_client=market_data_client,
        dart_client=dart_client,
        adapter_config=config,
        holiday_dates=holiday_dates,
    )
    return save_bundle_outputs(
        bundle,
        data_dir=data_dir,
        request=request,
        request_signature=request_signature,
        corp_codes_by_ticker=corp_codes_by_ticker,
        dart_enabled=dart_client is not None,
        market_data_provider=market_data_provider,
    )


def load_corp_code_mapping(path: str | Path | None) -> dict[str, str]:
    return load_corp_codes_csv(path)


def run_kr_research_pipeline(
    request: KRResearchDataRequest,
    *,
    data_dir: str | Path = "kr_stocks/data",
    corp_codes_by_ticker: dict[str, str] | None = None,
    market_data_client: KRHistoricalMarketDataClient | None = None,
    dart_client: DARTOpenAPIClient | None = None,
    adapter_config: KRMarketAdapterConfig | None = None,
    holiday_dates: set[date] | None = None,
    use_cache: bool = True,
) -> KRPipelineRun:
    return run_kr_data_pipeline(
        request,
        data_dir=data_dir,
        corp_codes_by_ticker=corp_codes_by_ticker,
        market_data_client=market_data_client,
        dart_client=dart_client,
        adapter_config=adapter_config,
        holiday_dates=holiday_dates,
        use_cache=use_cache,
    )


def _load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def _load_filings(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "filed_at" in frame.columns:
        frame["filed_at"] = pd.to_datetime(frame["filed_at"])
    if "session_date" in frame.columns:
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    return frame


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
