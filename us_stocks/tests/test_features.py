from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.features import build_features


class FeatureTests(unittest.TestCase):
    def test_build_features_creates_richer_price_and_volume_columns(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=260)
        prices = pd.DataFrame(
            {
                "date": list(dates) * 1,
                "ticker": ["AAA"] * len(dates),
                "open": [100.0 + index for index in range(len(dates))],
                "high": [101.0 + index for index in range(len(dates))],
                "low": [99.0 + index for index in range(len(dates))],
                "close": [100.0 * (1.002 ** index) for index in range(len(dates))],
                "volume": [1_000 + (index % 10) * 100 for index in range(len(dates))],
            }
        )
        benchmark = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["SPY"] * len(dates),
                "open": [200.0 + index * 0.3 for index in range(len(dates))],
                "high": [201.0 + index * 0.3 for index in range(len(dates))],
                "low": [199.0 + index * 0.3 for index in range(len(dates))],
                "close": [200.0 * (1.001 ** index) for index in range(len(dates))],
                "volume": [10_000_000 + (index % 5) * 50_000 for index in range(len(dates))],
            }
        )

        features = build_features(prices, benchmark)
        latest = features.iloc[-1]

        for column in [
            "ret_5",
            "ret_120",
            "vol_60",
            "price_vs_sma20",
            "price_vs_sma200",
            "sma20_vs_sma50",
            "drawdown_60",
            "dollar_volume_20",
            "volume_ratio_20",
            "log_dollar_volume_20",
            "range_pct_20",
            "vol_ratio_20_60",
            "momentum_20_60_gap",
            "benchmark_ret_20",
            "benchmark_vol_20",
            "rel_ret_20",
            "benchmark_price_vs_sma200",
            "relative_momentum_gap",
        ]:
            self.assertIn(column, features.columns)
            self.assertTrue(pd.notna(latest[column]), msg=f"{column} should be populated on the latest row")

        self.assertIn("market_trend_ok", features.columns)
        self.assertIn("market_high_vol_regime", features.columns)

    def test_build_features_creates_sector_and_cross_sectional_context(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=260)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.extend(
                [
                    {
                        "date": date,
                        "ticker": "AAA",
                        "open": 100.0 + index,
                        "high": 101.0 + index,
                        "low": 99.0 + index,
                        "close": 100.0 * (1.003 ** index),
                        "volume": 1_000_000 + (index % 5) * 5_000,
                    },
                    {
                        "date": date,
                        "ticker": "BBB",
                        "open": 80.0 + index * 0.3,
                        "high": 81.0 + index * 0.3,
                        "low": 79.0 + index * 0.3,
                        "close": 80.0 * (1.001 ** index),
                        "volume": 900_000 + (index % 7) * 4_000,
                    },
                    {
                        "date": date,
                        "ticker": "CCC",
                        "open": 120.0 + index * 0.2,
                        "high": 121.0 + index * 0.2,
                        "low": 119.0 + index * 0.2,
                        "close": 120.0 * (1.002 ** index),
                        "volume": 1_100_000 + (index % 6) * 3_000,
                    },
                ]
            )
        prices = pd.DataFrame(records)
        benchmark = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["SPY"] * len(dates),
                "open": [200.0 + index * 0.2 for index in range(len(dates))],
                "high": [201.0 + index * 0.2 for index in range(len(dates))],
                "low": [199.0 + index * 0.2 for index in range(len(dates))],
                "close": [200.0 * (1.0015 ** index) for index in range(len(dates))],
                "volume": [10_000_000 + (index % 5) * 25_000 for index in range(len(dates))],
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "sector": ["Technology", "Technology", "Healthcare"],
            }
        )

        features = build_features(prices, benchmark, metadata)
        latest = features.loc[features["date"] == features["date"].max()].copy()

        for column in [
            "cs_ret_20_z",
            "cs_rel_ret_20_z",
            "universe_momentum_rank_pct",
            "sector_ret_20_gap",
            "sector_rel_ret_20_gap",
            "sector_momentum_rank_pct",
            "sector_trend_share",
            "market_breadth_trend_share",
        ]:
            self.assertIn(column, latest.columns)
            self.assertTrue(latest[column].notna().all(), msg=f"{column} should be populated")

        aaa = latest.loc[latest["ticker"] == "AAA"].iloc[0]
        bbb = latest.loc[latest["ticker"] == "BBB"].iloc[0]
        self.assertGreater(float(aaa["sector_ret_20_gap"]), float(bbb["sector_ret_20_gap"]))
        self.assertGreaterEqual(float(latest["market_breadth_trend_share"].iloc[0]), 0.0)
        self.assertLessEqual(float(latest["market_breadth_trend_share"].iloc[0]), 1.0)

    def test_build_features_applies_universe_eligibility_rules(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=80)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.extend(
                [
                    {
                        "date": date,
                        "ticker": "AAA",
                        "open": 100.0 + index * 0.1,
                        "high": 101.0 + index * 0.1,
                        "low": 99.0 + index * 0.1,
                        "close": 100.0 * (1.002 ** index),
                        "volume": 2_000_000,
                    },
                    {
                        "date": date,
                        "ticker": "BBB",
                        "open": 4.0 + index * 0.01,
                        "high": 4.2 + index * 0.01,
                        "low": 3.8 + index * 0.01,
                        "close": 4.0 * (1.001 ** index),
                        "volume": 10_000,
                    },
                ]
            )
        prices = pd.DataFrame(records)

        features = build_features(
            prices,
            eligibility_rules={
                "min_close_price": 5.0,
                "min_dollar_volume_20": 50_000_000.0,
                "min_universe_age_days": 30,
            },
        )

        latest = features.loc[features["date"] == features["date"].max()].copy()
        aaa = latest.loc[latest["ticker"] == "AAA"].iloc[0]
        bbb = latest.loc[latest["ticker"] == "BBB"].iloc[0]

        self.assertIn("eligible_universe", features.columns)
        self.assertIn("universe_age_days", features.columns)
        self.assertTrue(bool(aaa["eligible_universe"]))
        self.assertFalse(bool(bbb["eligible_universe"]))
        self.assertGreaterEqual(float(aaa["universe_age_days"]), 30.0)

    def test_build_features_respects_universe_snapshots(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=260)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.extend(
                [
                    {
                        "date": date,
                        "ticker": "AAA",
                        "open": 100.0 + index * 0.2,
                        "high": 101.0 + index * 0.2,
                        "low": 99.0 + index * 0.2,
                        "close": 100.0 * (1.0025 ** index),
                        "volume": 1_000_000 + (index % 4) * 2_500,
                    },
                    {
                        "date": date,
                        "ticker": "BBB",
                        "open": 90.0 + index * 0.15,
                        "high": 91.0 + index * 0.15,
                        "low": 89.0 + index * 0.15,
                        "close": 90.0 * (1.0015 ** index),
                        "volume": 900_000 + (index % 6) * 3_000,
                    },
                    {
                        "date": date,
                        "ticker": "CCC",
                        "open": 70.0 + index * 0.1,
                        "high": 71.0 + index * 0.1,
                        "low": 69.0 + index * 0.1,
                        "close": 70.0 * (1.0035 ** index),
                        "volume": 850_000 + (index % 5) * 4_000,
                    },
                ]
            )

        prices = pd.DataFrame(records)
        benchmark = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["SPY"] * len(dates),
                "open": [200.0 + index * 0.2 for index in range(len(dates))],
                "high": [201.0 + index * 0.2 for index in range(len(dates))],
                "low": [199.0 + index * 0.2 for index in range(len(dates))],
                "close": [200.0 * (1.0012 ** index) for index in range(len(dates))],
                "volume": [10_000_000 + (index % 5) * 25_000 for index in range(len(dates))],
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "sector": ["Technology", "Technology", "Healthcare"],
            }
        )
        snapshots = pd.DataFrame(
            {
                "effective_date": ["2024-01-01", "2024-01-01", "2024-09-02", "2024-09-02", "2024-09-02"],
                "ticker": ["AAA", "BBB", "AAA", "BBB", "CCC"],
            }
        )

        features = build_features(prices, benchmark, metadata, snapshots)
        before_switch = features.loc[features["date"] < pd.Timestamp("2024-09-02")]
        after_switch = features.loc[features["date"] >= pd.Timestamp("2024-09-02")]

        self.assertEqual(sorted(before_switch["ticker"].unique().tolist()), ["AAA", "BBB"])
        self.assertIn("CCC", after_switch["ticker"].unique().tolist())
        self.assertGreaterEqual(
            pd.to_datetime(features.loc[features["ticker"] == "CCC", "date"]).min(),
            pd.Timestamp("2024-09-02"),
        )

        latest_before = before_switch.loc[before_switch["date"] == before_switch["date"].max()]
        self.assertEqual(len(latest_before), 2)
        self.assertTrue(latest_before["cs_ret_20_z"].notna().all())
        self.assertTrue(latest_before["market_breadth_trend_share"].notna().all())


if __name__ == "__main__":
    unittest.main()
