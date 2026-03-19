from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kr_invest_ai.research import run_kr_research_backtest, save_kr_research_outputs


class KRResearchTests(unittest.TestCase):
    def test_run_kr_research_backtest_returns_artifacts(self) -> None:
        dates = pd.date_range("2026-01-01", periods=40, freq="B")
        rows = []
        for current_date in dates:
            rows.extend(
                [
                    {
                        "date": current_date,
                        "ticker": "005930.KS",
                        "listing_code": "005930",
                        "vendor_suffix": "KS",
                        "provider_symbol": "005930.KS",
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0 + len(rows) * 0.1,
                        "adj_close": 100.0 + len(rows) * 0.1,
                        "volume": 1000.0,
                        "provider": "stub",
                        "currency": "KRW",
                        "exchange_timezone": "Asia/Seoul",
                    },
                    {
                        "date": current_date,
                        "ticker": "000660.KS",
                        "listing_code": "000660",
                        "vendor_suffix": "KS",
                        "provider_symbol": "000660.KS",
                        "open": 90.0,
                        "high": 91.0,
                        "low": 89.0,
                        "close": 90.0 + len(rows) * 0.05,
                        "adj_close": 90.0 + len(rows) * 0.05,
                        "volume": 1200.0,
                        "provider": "stub",
                        "currency": "KRW",
                        "exchange_timezone": "Asia/Seoul",
                    },
                ]
            )
        prices = pd.DataFrame(rows)
        filings = pd.DataFrame(
            [
                {"session_date": dates[-2], "ticker": "005930.KS", "category": "earnings"},
                {"session_date": dates[-1], "ticker": "005930.KS", "category": "earnings"},
            ]
        )
        benchmark = pd.DataFrame(
            [{"date": current_date, "ticker": "069500.KS", "close": 100.0 + idx} for idx, current_date in enumerate(dates)]
        )

        run = run_kr_research_backtest(prices, benchmark, filings, transaction_cost_bps=5.0)

        self.assertFalse(run.features.empty)
        self.assertIn("rel_ret_20", run.features.columns)
        self.assertIn("benchmark_ret_20", run.features.columns)
        self.assertFalse(run.target_weights.empty)
        self.assertIn("cagr", run.backtest_result.summary.columns)
        self.assertFalse(run.benchmark_prices.empty)

    def test_save_kr_research_outputs_writes_manifest(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": "2026-03-18", "ticker": "005930.KS", "listing_code": "005930", "vendor_suffix": "KS", "provider_symbol": "005930.KS", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0, "volume": 1000.0, "provider": "stub", "currency": "KRW", "exchange_timezone": "Asia/Seoul"},
                {"date": "2026-03-19", "ticker": "005930.KS", "listing_code": "005930", "vendor_suffix": "KS", "provider_symbol": "005930.KS", "open": 101.0, "high": 102.0, "low": 100.0, "close": 102.0, "adj_close": 102.0, "volume": 1200.0, "provider": "stub", "currency": "KRW", "exchange_timezone": "Asia/Seoul"},
            ]
        )
        benchmark = pd.DataFrame([{"date": "2026-03-18", "ticker": "069500.KS", "close": 100.0}])
        run = run_kr_research_backtest(prices, benchmark, pd.DataFrame(), transaction_cost_bps=5.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = save_kr_research_outputs(run, output_dir=temp_dir)
            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["pipeline"], "kr_research_backtest")
        self.assertIn("summary", manifest["output_files"])
        self.assertIn("benchmark", manifest["output_files"])
