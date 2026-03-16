from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.features import build_features
from us_invest_ai.walkforward import WalkForwardConfig, prepare_learning_frame, select_walkforward_splits


class WalkForwardTests(unittest.TestCase):
    def test_select_walkforward_splits_respects_embargo(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=120)
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

        train_frame, validation_frame = select_walkforward_splits(frame, pd.Timestamp("2024-06-14"), walkforward_config)

        cutoff = pd.Timestamp("2024-06-14") - pd.offsets.BDay(5)
        self.assertFalse(train_frame.empty)
        self.assertFalse(validation_frame.empty)
        self.assertTrue((pd.to_datetime(train_frame["label_available_date"]) < cutoff).all())
        self.assertTrue((pd.to_datetime(validation_frame["label_available_date"]) < cutoff).all())


if __name__ == "__main__":
    unittest.main()
