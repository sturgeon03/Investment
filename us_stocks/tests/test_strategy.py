from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.strategy import generate_target_weights


def _build_features() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2025-01-30",
            "2025-01-30",
            "2025-01-31",
            "2025-01-31",
            "2025-02-27",
            "2025-02-27",
            "2025-02-28",
            "2025-02-28",
        ]
    )
    tickers = ["AAA", "BBB"] * 4
    ret_20 = [0.10, 0.05, 0.20, 0.01, 0.08, 0.02, 0.18, 0.03]
    ret_60 = [0.12, 0.04, 0.25, 0.02, 0.10, 0.03, 0.22, 0.05]
    vol_20 = [0.20, 0.18, 0.15, 0.30, 0.19, 0.20, 0.14, 0.29]

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "close": [100, 90, 101, 91, 103, 92, 105, 93],
            "ret_20": ret_20,
            "ret_60": ret_60,
            "vol_20": vol_20,
            "trend_ok": [True] * 8,
        }
    )


class StrategyTests(unittest.TestCase):
    def test_generate_target_weights_rebalances_monthly(self) -> None:
        config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=1,
            trend_filter_mode="hard",
            trend_penalty=0.35,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=0.25,
        )

        weights, ranking_history = generate_target_weights(_build_features(), config)

        self.assertEqual(weights.loc[pd.Timestamp("2025-01-31"), "AAA"], 1.0)
        self.assertEqual(weights.loc[pd.Timestamp("2025-01-31"), "BBB"], 0.0)
        self.assertEqual(weights.loc[pd.Timestamp("2025-02-28"), "AAA"], 1.0)
        self.assertTrue(
            ranking_history.loc[
                ranking_history["date"] == pd.Timestamp("2025-01-31"), "selected"
            ].any()
        )

    def test_llm_scores_can_override_price_signal(self) -> None:
        config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=1,
            trend_filter_mode="hard",
            trend_penalty=0.35,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=2.0,
        )
        llm_scores = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
                "ticker": ["AAA", "BBB"],
                "llm_score": [0.0, 5.0],
            }
        )

        weights, _ = generate_target_weights(_build_features(), config, llm_scores)

        self.assertEqual(weights.loc[pd.Timestamp("2025-01-31"), "AAA"], 0.0)
        self.assertEqual(weights.loc[pd.Timestamp("2025-01-31"), "BBB"], 1.0)

    def test_soft_trend_filter_allows_non_trending_stock_with_penalty(self) -> None:
        features = _build_features().copy()
        features.loc[(features["date"] == pd.Timestamp("2025-01-31")) & (features["ticker"] == "AAA"), "trend_ok"] = False
        config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=1,
            trend_filter_mode="soft",
            trend_penalty=0.10,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=0.25,
        )

        weights, _ = generate_target_weights(features, config)

        self.assertEqual(weights.loc[pd.Timestamp("2025-01-31"), "AAA"], 1.0)

    def test_generate_target_weights_excludes_ineligible_universe_rows(self) -> None:
        features = _build_features().copy()
        features["eligible_universe"] = True
        features.loc[
            (features["date"] == pd.Timestamp("2025-01-31")) & (features["ticker"] == "AAA"),
            "eligible_universe",
        ] = False
        config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=1,
            trend_filter_mode="hard",
            trend_penalty=0.35,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=0.25,
        )

        weights, ranking_history = generate_target_weights(features, config)

        self.assertEqual(weights.loc[pd.Timestamp("2025-01-31"), "AAA"], 0.0)
        self.assertEqual(weights.loc[pd.Timestamp("2025-01-31"), "BBB"], 1.0)
        selected = ranking_history.loc[
            (ranking_history["date"] == pd.Timestamp("2025-01-31")) & ranking_history["selected"]
        ]
        self.assertEqual(selected["ticker"].iloc[0], "BBB")


if __name__ == "__main__":
    unittest.main()
