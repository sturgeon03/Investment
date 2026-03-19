from __future__ import annotations

import unittest

import pandas as pd

from kr_invest_ai.features import build_kr_feature_frame


class KRFeatureAssemblyTests(unittest.TestCase):
    def test_build_kr_feature_frame_computes_price_and_filing_columns(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": "2026-03-17", "ticker": "005930.KS", "close": 70000.0, "high": 71000.0, "low": 69000.0, "volume": 1000.0},
                {"date": "2026-03-18", "ticker": "005930.KS", "close": 71400.0, "high": 72000.0, "low": 70000.0, "volume": 1100.0},
                {"date": "2026-03-19", "ticker": "005930.KS", "close": 72114.0, "high": 73000.0, "low": 71000.0, "volume": 1200.0},
            ]
        )
        benchmark = pd.DataFrame(
            [
                {"date": "2026-03-17", "ticker": "069500.KS", "close": 30000.0, "high": 30100.0, "low": 29900.0, "volume": 900.0},
                {"date": "2026-03-18", "ticker": "069500.KS", "close": 30300.0, "high": 30400.0, "low": 30200.0, "volume": 950.0},
                {"date": "2026-03-19", "ticker": "069500.KS", "close": 30603.0, "high": 30700.0, "low": 30500.0, "volume": 980.0},
            ]
        )
        filings = pd.DataFrame(
            [
                {"session_date": "2026-03-18", "ticker": "005930.KS", "category": "earnings"},
                {"session_date": "2026-03-18", "ticker": "005930.KS", "category": "governance"},
                {"session_date": "2026-03-19", "ticker": "005930.KS", "category": "capital_event"},
            ]
        )

        features = build_kr_feature_frame(prices, filings, benchmark)

        self.assertIn("ret_1", features.columns)
        self.assertIn("avg_dollar_volume_20", features.columns)
        self.assertIn("filing_count_20", features.columns)
        self.assertIn("days_since_last_filing", features.columns)
        self.assertIn("rel_ret_1", features.columns)
        self.assertIn("market_trend_ok", features.columns)
        self.assertEqual(float(features.loc[features["date"] == pd.Timestamp("2026-03-18"), "filing_count"].iloc[0]), 2.0)
        self.assertEqual(float(features.loc[features["date"] == pd.Timestamp("2026-03-18"), "earnings_filing_count"].iloc[0]), 1.0)
        self.assertEqual(float(features.loc[features["date"] == pd.Timestamp("2026-03-19"), "capital_event_count"].iloc[0]), 1.0)
        self.assertEqual(float(features.loc[features["date"] == pd.Timestamp("2026-03-19"), "days_since_last_filing"].iloc[0]), 0.0)
        latest = features.loc[features["date"] == pd.Timestamp("2026-03-19")].iloc[0]
        self.assertAlmostEqual(float(latest["benchmark_ret_1"]), 0.01, places=6)
        self.assertAlmostEqual(float(latest["rel_ret_1"]), 0.0, places=6)
        self.assertEqual(float(latest["market_trend_ok"]), 1.0)

    def test_build_kr_feature_frame_handles_missing_filings(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": "2026-03-17", "ticker": "000660.KS", "close": 100000.0, "high": 101000.0, "low": 99000.0, "volume": 500.0},
                {"date": "2026-03-18", "ticker": "000660.KS", "close": 101000.0, "high": 102000.0, "low": 99500.0, "volume": 600.0},
            ]
        )

        features = build_kr_feature_frame(prices, None)

        self.assertTrue((features["filing_count"] == 0.0).all())
        self.assertTrue((features["days_since_last_filing"] == 9999.0).all())
        self.assertTrue((features["benchmark_ret_20"] == 0.0).all())
        self.assertTrue((features["market_trend_ok"] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
