from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from kr_invest_ai.dart_adapter import NormalizedDARTFiling, filings_to_frame
from kr_invest_ai.dart_client import DARTListFilingsRequest, DARTOpenAPIClient
from kr_invest_ai.market_config import KRMarketAdapterConfig, load_kr_market_adapter_config
from kr_invest_ai.market_data_client import KRDailyOHLCVRequest, KRHistoricalMarketDataClient, bars_to_frame


@dataclass(frozen=True, slots=True)
class KRResearchDataRequest:
    tickers: tuple[str, ...]
    price_start_date: date
    price_end_date: date
    benchmark_ticker: str | None = None
    filings_start_date: date | None = None
    filings_end_date: date | None = None
    default_vendor_suffix: str | None = "KS"


@dataclass(slots=True)
class KRResearchDataBundle:
    prices: pd.DataFrame
    benchmark_prices: pd.DataFrame
    filings: pd.DataFrame
    price_bar_count: int
    benchmark_bar_count: int
    filing_count: int
    provenance: dict[str, object]


def build_kr_research_data_bundle(
    request: KRResearchDataRequest,
    *,
    corp_codes_by_ticker: dict[str, str] | None = None,
    market_data_client: KRHistoricalMarketDataClient | None = None,
    dart_client: DARTOpenAPIClient | None = None,
    adapter_config: KRMarketAdapterConfig | None = None,
    holiday_dates: set[date] | None = None,
) -> KRResearchDataBundle:
    config = adapter_config or load_kr_market_adapter_config()
    market_client = market_data_client or KRHistoricalMarketDataClient(adapter_config=config)

    price_frames: list[pd.DataFrame] = []
    price_bar_count = 0
    for ticker in request.tickers:
        bars = market_client.fetch_daily_bars(
            KRDailyOHLCVRequest(
                ticker=ticker,
                start_date=request.price_start_date,
                end_date=request.price_end_date,
                default_vendor_suffix=request.default_vendor_suffix,
            )
        )
        price_bar_count += len(bars)
        price_frames.append(bars_to_frame(bars))

    prices = (
        pd.concat(price_frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        if price_frames
        else bars_to_frame([])
    )

    benchmark_prices = bars_to_frame([])
    benchmark_bar_count = 0
    if request.benchmark_ticker:
        benchmark_bars = market_client.fetch_daily_bars(
            KRDailyOHLCVRequest(
                ticker=request.benchmark_ticker,
                start_date=request.price_start_date,
                end_date=request.price_end_date,
                default_vendor_suffix=request.default_vendor_suffix,
            )
        )
        benchmark_bar_count = len(benchmark_bars)
        benchmark_prices = bars_to_frame(benchmark_bars)

    normalized_filings: list[NormalizedDARTFiling] = []
    if dart_client is not None and request.filings_start_date is not None and request.filings_end_date is not None:
        for ticker, corp_code in sorted((corp_codes_by_ticker or {}).items()):
            if ticker not in request.tickers:
                continue
            normalized_filings.extend(
                dart_client.fetch_normalized_filings(
                    DARTListFilingsRequest(
                        begin_date=request.filings_start_date,
                        end_date=request.filings_end_date,
                        corp_code=corp_code,
                    ),
                    holiday_dates=holiday_dates,
                )
            )

    filings = filings_to_frame(normalized_filings)
    return KRResearchDataBundle(
        prices=prices,
        benchmark_prices=benchmark_prices,
        filings=filings,
        price_bar_count=price_bar_count,
        benchmark_bar_count=benchmark_bar_count,
        filing_count=len(normalized_filings),
        provenance={
            "market_data_provider": getattr(market_client, "provider_name", None),
            "dart_enabled": dart_client is not None,
            "ticker_count": len(request.tickers),
            "benchmark_ticker": request.benchmark_ticker,
            "price_window": {
                "start_date": request.price_start_date.isoformat(),
                "end_date": request.price_end_date.isoformat(),
            },
            "filings_window": (
                {
                    "start_date": request.filings_start_date.isoformat(),
                    "end_date": request.filings_end_date.isoformat(),
                }
                if request.filings_start_date is not None and request.filings_end_date is not None
                else None
            ),
        },
    )
