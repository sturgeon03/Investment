from __future__ import annotations

import unittest
from datetime import date
from datetime import datetime
from zoneinfo import ZoneInfo

from kr_invest_ai.data_bundle import KRResearchDataRequest, build_kr_research_data_bundle
from kr_invest_ai.market_config import load_kr_market_adapter_config
from kr_invest_ai.market_data_client import (
    KRDailyOHLCVRequest,
    KRHistoricalMarketDataClient,
    NormalizedDailyOHLCVBar,
)
from kr_invest_ai.dart_adapter import NormalizedDARTFiling
from kr_invest_ai.dart_client import DARTListFilingsRequest, DARTOpenAPIClient
from kr_invest_ai.tickers import canonicalize_kr_ticker


class _StubMarketDataClient(KRHistoricalMarketDataClient):
    def __init__(self) -> None:
        super().__init__(adapter_config=load_kr_market_adapter_config())
        self.requests: list[KRDailyOHLCVRequest] = []

    def fetch_daily_bars(self, request: KRDailyOHLCVRequest) -> list[NormalizedDailyOHLCVBar]:
        self.requests.append(request)
        canonical = request.canonical_ticker(adapter_config=self.adapter_config)
        return [
            NormalizedDailyOHLCVBar(
                date=request.start_date,
                canonical_ticker=canonical,
                provider_symbol=request.vendor_symbol(adapter_config=self.adapter_config),
                open=100.0,
                high=110.0,
                low=95.0,
                close=105.0,
                adj_close=105.0,
                volume=1_000.0,
                provider=self.provider_name,
                currency="KRW",
                exchange_timezone="Asia/Seoul",
            )
        ]


class _StubDARTClient(DARTOpenAPIClient):
    def __init__(self) -> None:
        super().__init__(api_key="test-key", adapter_config=load_kr_market_adapter_config())
        self.requests: list[DARTListFilingsRequest] = []

    def fetch_normalized_filings(
        self,
        request: DARTListFilingsRequest,
        *,
        holiday_dates: set[date] | None = None,
    ) -> list[NormalizedDARTFiling]:
        self.requests.append(request)
        canonical = canonicalize_kr_ticker("005930.KS", conventions=self.adapter_config.ticker_conventions)
        return [
            NormalizedDARTFiling(
                receipt_no="20260319000123",
                issuer_code=request.corp_code,
                issuer_name="Samsung Electronics",
                canonical_ticker=canonical,
                filing_type="EARNINGS",
                filing_subtype=None,
                title="Earnings release",
                filed_at=datetime(2026, 3, 19, 15, 30, tzinfo=ZoneInfo("Asia/Seoul")),
                session_date=date(2026, 3, 19),
                category="earnings",
                body_text=None,
                source_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260319000123",
                raw_fields={"corp_code": request.corp_code or ""},
            )
        ]


class KRDataBundleTests(unittest.TestCase):
    def test_build_kr_research_data_bundle_combines_prices_and_filings(self) -> None:
        bundle = build_kr_research_data_bundle(
            KRResearchDataRequest(
                tickers=("005930", "000660"),
                price_start_date=date(2026, 3, 19),
                price_end_date=date(2026, 3, 19),
                benchmark_ticker="069500",
                filings_start_date=date(2026, 3, 1),
                filings_end_date=date(2026, 3, 19),
            ),
            corp_codes_by_ticker={"005930": "00126380"},
            market_data_client=_StubMarketDataClient(),
            dart_client=_StubDARTClient(),
        )

        self.assertEqual(bundle.price_bar_count, 2)
        self.assertEqual(bundle.benchmark_bar_count, 1)
        self.assertEqual(bundle.filing_count, 1)
        self.assertEqual(sorted(bundle.prices["ticker"].unique().tolist()), ["000660.KS", "005930.KS"])
        self.assertEqual(bundle.benchmark_prices.loc[0, "ticker"], "069500.KS")
        self.assertEqual(bundle.filings.loc[0, "listing_code"], "005930")
        self.assertTrue(bundle.provenance["dart_enabled"])

    def test_build_kr_research_data_bundle_skips_filings_when_client_missing(self) -> None:
        bundle = build_kr_research_data_bundle(
            KRResearchDataRequest(
                tickers=("005930",),
                price_start_date=date(2026, 3, 19),
                price_end_date=date(2026, 3, 19),
            ),
            market_data_client=_StubMarketDataClient(),
            dart_client=None,
        )

        self.assertEqual(bundle.filing_count, 0)
        self.assertTrue(bundle.filings.empty)


if __name__ == "__main__":
    unittest.main()
