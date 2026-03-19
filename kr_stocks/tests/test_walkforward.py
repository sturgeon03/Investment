from __future__ import annotations

import unittest

import pandas as pd

from kr_invest_ai.strategy import KRStrategyConfig
from kr_invest_ai.walkforward import KRWalkForwardConfig, prepare_learning_frame, select_walkforward_splits


class KRWalkForwardTests(unittest.TestCase):
    def test_select_walkforward_splits_respects_label_cutoff(self) -> None:
        dates = pd.date_range("2025-01-01", periods=140, freq="B")
        rows = []
        for idx, current_date in enumerate(dates):
            for ticker, base_close in (("005930.KS", 100.0), ("000660.KS", 90.0)):
                rows.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "close": base_close + idx,
                        "ret_1": 0.01,
                        "ret_5": 0.02,
                        "ret_20": 0.04,
                        "rel_ret_5": 0.01,
                        "rel_ret_20": 0.015,
                        "benchmark_ret_20": 0.02,
                        "avg_dollar_volume_20": 1_000_000.0,
                        "vol_20": 0.10,
                        "benchmark_vol_20": 0.08,
                        "range_20": 0.02,
                        "market_trend_ok": 1.0,
                        "filing_count_20": 0.0,
                        "earnings_filing_count_60": 0.0,
                        "capital_event_count_60": 0.0,
                        "days_since_last_filing": 9999.0,
                    }
                )
        features = pd.DataFrame(rows)
        config = KRWalkForwardConfig(
            label_horizon_days=5,
            validation_window_days=20,
            min_training_samples=40,
            min_validation_samples=20,
        )
        frame = prepare_learning_frame(features, KRStrategyConfig(min_history_days=20), config)
        rebalance_date = pd.Timestamp("2025-06-30")

        train_frame, validation_frame = select_walkforward_splits(frame, rebalance_date, config)

        self.assertFalse(train_frame.empty)
        self.assertFalse(validation_frame.empty)
        cutoff = rebalance_date - pd.offsets.BDay(config.resolved_embargo_days())
        self.assertTrue((pd.to_datetime(train_frame["label_available_date"]).dt.normalize() < cutoff).all())
        self.assertTrue((pd.to_datetime(validation_frame["label_available_date"]).dt.normalize() < cutoff).all())


if __name__ == "__main__":
    unittest.main()
