from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.features import build_features
from us_invest_ai.walkforward import (
    WalkForwardConfig,
    prepare_learning_frame,
    select_live_candidates,
    select_walkforward_splits,
)


class WalkForwardTests(unittest.TestCase):
    def test_select_walkforward_splits_respects_embargo(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=280)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.append(
                {
                    "date": date,
                    "ticker": "AAA",
                    "open": 100.0 + index,
                    "high": 100.5 + index,
                    "low": 99.5 + index,
                    "close": 100.0 * (1.002 ** index),
                    "volume": 1_000,
                }
            )
        features = build_features(pd.DataFrame(records))
        strategy_config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=20,
            trend_filter_mode="soft",
            trend_penalty=0.10,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=0.0,
        )
        walkforward_config = WalkForwardConfig(
            label_horizon_days=5,
            validation_window_days=10,
            embargo_days=5,
            min_training_samples=20,
            min_validation_samples=5,
        )
        frame = prepare_learning_frame(features, strategy_config, walkforward_config)

        train_frame, validation_frame = select_walkforward_splits(frame, pd.Timestamp("2024-12-13"), walkforward_config)

        cutoff = pd.Timestamp("2024-12-13") - pd.offsets.BDay(5)
        self.assertFalse(train_frame.empty)
        self.assertFalse(validation_frame.empty)
        self.assertTrue((pd.to_datetime(train_frame["label_available_date"]) < cutoff).all())
        self.assertTrue((pd.to_datetime(validation_frame["label_available_date"]) < cutoff).all())

    def test_prepare_learning_frame_preserves_attached_llm_scores(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=280)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.append(
                {
                    "date": date,
                    "ticker": "AAA",
                    "open": 100.0 + index,
                    "high": 100.5 + index,
                    "low": 99.5 + index,
                    "close": 100.0 * (1.002 ** index),
                    "volume": 1_000,
                }
            )
        features = build_features(pd.DataFrame(records))
        strategy_config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=20,
            trend_filter_mode="soft",
            trend_penalty=0.10,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=0.25,
        )
        walkforward_config = WalkForwardConfig(
            label_horizon_days=20,
            validation_window_days=20,
            embargo_days=20,
            min_training_samples=40,
            min_validation_samples=20,
            use_llm_feature=True,
        )
        llm_scores = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-10-01"]),
                "ticker": ["AAA"],
                "llm_score": [0.8],
            }
        )

        frame = prepare_learning_frame(features, strategy_config, walkforward_config, llm_scores)
        attached = frame.loc[frame["date"] >= pd.Timestamp("2024-10-01"), "llm_score"]

        self.assertFalse(attached.empty)
        self.assertTrue((attached == 0.8).all())
        self.assertIn("llm_score", walkforward_config.resolved_feature_columns())

    def test_select_live_candidates_excludes_ineligible_universe_rows(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=280)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.append(
                {
                    "date": date,
                    "ticker": "AAA",
                    "open": 100.0 + index,
                    "high": 100.5 + index,
                    "low": 99.5 + index,
                    "close": 100.0 * (1.002 ** index),
                    "volume": 1_000,
                }
            )
            records.append(
                {
                    "date": date,
                    "ticker": "BBB",
                    "open": 90.0 + index,
                    "high": 90.5 + index,
                    "low": 89.5 + index,
                    "close": 90.0 * (1.002 ** index),
                    "volume": 1_000,
                }
            )
        features = build_features(pd.DataFrame(records))
        features["eligible_universe"] = True
        features.loc[
            (features["ticker"] == "AAA") & (features["date"] == features["date"].max()),
            "eligible_universe",
        ] = False
        strategy_config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=20,
            trend_filter_mode="soft",
            trend_penalty=0.10,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=0.0,
        )
        walkforward_config = WalkForwardConfig(
            label_horizon_days=20,
            validation_window_days=20,
            embargo_days=20,
            min_training_samples=40,
            min_validation_samples=20,
        )

        frame = prepare_learning_frame(features, strategy_config, walkforward_config)
        candidates = select_live_candidates(frame, features["date"].max(), strategy_config)

        self.assertEqual(sorted(candidates["ticker"].tolist()), ["BBB"])


if __name__ == "__main__":
    unittest.main()
