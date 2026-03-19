from __future__ import annotations

import unittest
from datetime import date

from kr_invest_ai.market_config import load_kr_market_adapter_config
from kr_invest_ai.market_data_client import (
    KRDailyOHLCVRequest,
    KRHistoricalMarketDataClient,
    KRMarketDataError,
    bars_to_frame,
)


class _StubMarketDataClient(KRHistoricalMarketDataClient):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(adapter_config=load_kr_market_adapter_config())
        self.payload = payload
        self.last_symbol: str | None = None
        self.last_params: dict[str, str] | None = None

    def _request_json(self, provider_symbol: str, params: dict[str, str]) -> dict[str, object]:
        self.last_symbol = provider_symbol
        self.last_params = params
        return self.payload


class KRMarketDataClientTests(unittest.TestCase):
    def test_request_builds_vendor_symbol_and_params(self) -> None:
        request = KRDailyOHLCVRequest(
            ticker="005930",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 19),
            default_vendor_suffix="KS",
        )
        adapter = load_kr_market_adapter_config()

        self.assertEqual(request.vendor_symbol(adapter), "005930.KS")
        params = request.to_query_params()
        self.assertEqual(params["interval"], "1d")
        self.assertEqual(params["includeAdjustedClose"], "true")

    def test_fetch_daily_bars_normalizes_provider_response(self) -> None:
        client = _StubMarketDataClient(
            {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "currency": "KRW",
                                "exchangeTimezoneName": "Asia/Seoul",
                            },
                            "timestamp": [1773878400, 1773964800],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [70000.0, 71000.0],
                                        "high": [70500.0, 71500.0],
                                        "low": [69500.0, 70500.0],
                                        "close": [70200.0, 71200.0],
                                        "volume": [1000000, 1200000],
                                    }
                                ],
                                "adjclose": [{"adjclose": [70200.0, 71200.0]}],
                            },
                        }
                    ],
                    "error": None,
                }
            }
        )

        bars = client.fetch_daily_bars(
            KRDailyOHLCVRequest(
                ticker="005930.KS",
                start_date=date(2026, 3, 18),
                end_date=date(2026, 3, 19),
            )
        )

        self.assertEqual(client.last_symbol, "005930.KS")
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].canonical_ticker.listing_code, "005930")
        self.assertEqual(bars[0].provider_symbol, "005930.KS")
        self.assertEqual(bars[0].currency, "KRW")

    def test_fetch_daily_bars_raises_on_provider_error(self) -> None:
        client = _StubMarketDataClient({"chart": {"result": None, "error": {"code": "bad"}}})

        with self.assertRaises(KRMarketDataError):
            client.fetch_daily_bars(
                KRDailyOHLCVRequest(
                    ticker="005930.KS",
                    start_date=date(2026, 3, 18),
                    end_date=date(2026, 3, 19),
                )
            )

    def test_bars_to_frame_returns_research_ready_columns(self) -> None:
        client = _StubMarketDataClient(
            {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "currency": "KRW",
                                "exchangeTimezoneName": "Asia/Seoul",
                            },
                            "timestamp": [1773878400],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [70000.0],
                                        "high": [70500.0],
                                        "low": [69500.0],
                                        "close": [70200.0],
                                        "volume": [1000000],
                                    }
                                ],
                                "adjclose": [{"adjclose": [70200.0]}],
                            },
                        }
                    ],
                    "error": None,
                }
            }
        )

        bars = client.fetch_daily_bars(
            KRDailyOHLCVRequest(
                ticker="005930.KS",
                start_date=date(2026, 3, 18),
                end_date=date(2026, 3, 18),
            )
        )
        frame = bars_to_frame(bars)

        self.assertEqual(
            list(frame.columns),
            [
                "date",
                "provider_symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "provider",
                "currency",
                "exchange_timezone",
                "ticker",
                "listing_code",
                "vendor_suffix",
            ],
        )
        self.assertEqual(frame.loc[0, "ticker"], "005930.KS")
        self.assertEqual(frame.loc[0, "listing_code"], "005930")


if __name__ == "__main__":
    unittest.main()
