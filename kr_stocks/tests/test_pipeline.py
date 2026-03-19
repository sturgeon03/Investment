from __future__ import annotations

import tempfile
import unittest
from datetime import date
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from kr_invest_ai.data_bundle import KRResearchDataRequest
from kr_invest_ai.dart_adapter import NormalizedDARTFiling
from kr_invest_ai.dart_client import DARTListFilingsRequest, DARTOpenAPIClient
from kr_invest_ai.market_config import load_kr_market_adapter_config
from kr_invest_ai.market_data_client import (
    KRDailyOHLCVRequest,
    KRHistoricalMarketDataClient,
    NormalizedDailyOHLCVBar,
)
from kr_invest_ai.pipeline import (
    build_request_signature,
    load_cached_bundle,
    load_corp_code_mapping,
    load_corp_codes_csv,
    run_kr_data_pipeline,
    run_kr_research_pipeline,
)
from kr_invest_ai.tickers import canonicalize_kr_ticker


class _StubMarketDataClient(KRHistoricalMarketDataClient):
    def __init__(self) -> None:
        super().__init__(adapter_config=load_kr_market_adapter_config())
        self.calls = 0

    def fetch_daily_bars(self, request: KRDailyOHLCVRequest) -> list[NormalizedDailyOHLCVBar]:
        self.calls += 1
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
                volume=1000.0,
                provider=self.provider_name,
                currency="KRW",
                exchange_timezone="Asia/Seoul",
            )
        ]


class _StubDARTClient(DARTOpenAPIClient):
    def __init__(self) -> None:
        super().__init__(api_key="test-key", adapter_config=load_kr_market_adapter_config())
        self.calls = 0

    def fetch_normalized_filings(
        self,
        request: DARTListFilingsRequest,
        *,
        holiday_dates: set[date] | None = None,
    ) -> list[NormalizedDARTFiling]:
        self.calls += 1
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


class KRPipelineTests(unittest.TestCase):
    def test_load_corp_code_mapping_reads_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "corp_codes.csv"
            csv_path.write_text("ticker,corp_code\n005930,00126380\n", encoding="utf-8")
            mapping = load_corp_codes_csv(csv_path)

        self.assertEqual(mapping, {"005930": "00126380"})

    def test_run_kr_research_pipeline_writes_cache_and_reuses_it(self) -> None:
        request = KRResearchDataRequest(
            tickers=("005930",),
            price_start_date=date(2026, 3, 19),
            price_end_date=date(2026, 3, 19),
            benchmark_ticker="069500",
            filings_start_date=date(2026, 3, 1),
            filings_end_date=date(2026, 3, 19),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            market_client = _StubMarketDataClient()
            dart_client = _StubDARTClient()
            first_run = run_kr_data_pipeline(
                request,
                data_dir=temp_dir,
                corp_codes_by_ticker={"005930": "00126380"},
                market_data_client=market_client,
                dart_client=dart_client,
            )

            self.assertFalse(first_run.from_cache)
            self.assertTrue(first_run.prices_path.exists())
            self.assertTrue(first_run.benchmark_path.exists())
            self.assertTrue(first_run.filings_path.exists())
            self.assertTrue(first_run.manifest_path.exists())
            self.assertEqual(market_client.calls, 2)
            self.assertEqual(dart_client.calls, 1)
            self.assertIn("output_files", first_run.manifest)
            self.assertEqual(first_run.manifest["output_files"]["prices"]["path"], str(first_run.prices_path))
            self.assertEqual(first_run.manifest["output_files"]["benchmark"]["path"], str(first_run.benchmark_path))

            second_run = run_kr_research_pipeline(
                request,
                data_dir=temp_dir,
                corp_codes_by_ticker={"005930": "00126380"},
                market_data_client=_StubMarketDataClient(),
                dart_client=_StubDARTClient(),
            )

        self.assertTrue(second_run.from_cache)
        self.assertEqual(len(second_run.bundle.prices), 1)
        self.assertEqual(len(second_run.bundle.benchmark_prices), 1)
        self.assertEqual(len(second_run.bundle.filings), 1)

    def test_build_request_signature_changes_when_dart_state_changes(self) -> None:
        request = KRResearchDataRequest(
            tickers=("005930",),
            price_start_date=date(2026, 3, 1),
            price_end_date=date(2026, 3, 19),
        )

        with_dart = build_request_signature(
            request,
            corp_codes_by_ticker={"005930": "00126380"},
            dart_enabled=True,
            market_data_provider="stub_market",
        )
        without_dart = build_request_signature(
            request,
            corp_codes_by_ticker={"005930": "00126380"},
            dart_enabled=False,
            market_data_provider="stub_market",
        )

        self.assertNotEqual(with_dart, without_dart)

    def test_load_cached_bundle_returns_none_for_mismatched_signature(self) -> None:
        request = KRResearchDataRequest(
            tickers=("005930",),
            price_start_date=date(2026, 3, 19),
            price_end_date=date(2026, 3, 19),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = run_kr_data_pipeline(
                request,
                data_dir=temp_dir,
                market_data_client=_StubMarketDataClient(),
                use_cache=False,
            )

            self.assertIsNotNone(load_cached_bundle(temp_dir, request_signature=run.manifest["request_signature"]))
            self.assertIsNone(load_cached_bundle(temp_dir, request_signature="not-the-same"))


if __name__ == "__main__":
    unittest.main()
