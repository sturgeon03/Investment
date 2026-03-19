from __future__ import annotations

import unittest

import pandas as pd

from kr_invest_ai.ml_strategy import (
    KRMLModelConfig,
    generate_ridge_target_weights,
    generate_ridge_walkforward_target_weights,
)
from kr_invest_ai.strategy import KRStrategyConfig


class KRMLStrategyTests(unittest.TestCase):
    def test_generate_ridge_target_weights_returns_monthly_weights(self) -> None:
        dates = pd.date_range("2026-01-01", periods=80, freq="B")
        rows = []
        for idx, current_date in enumerate(dates):
            rows.append(
                {
                    "date": current_date,
                    "ticker": "005930.KS",
                    "close": 100.0 + idx,
                    "ret_1": 0.01,
                    "ret_5": 0.02,
                    "ret_20": 0.05,
                    "rel_ret_5": 0.01,
                    "rel_ret_20": 0.02,
                    "benchmark_ret_20": 0.03,
                    "avg_dollar_volume_20": 1_000_000.0,
                    "vol_20": 0.10,
                    "benchmark_vol_20": 0.08,
                    "range_20": 0.02,
                    "market_trend_ok": 1.0,
                    "filing_count_20": 0.0,
                    "earnings_filing_count_60": 1.0 if idx > 60 else 0.0,
                    "capital_event_count_60": 0.0,
                    "days_since_last_filing": 5.0,
                }
            )
            rows.append(
                {
                    "date": current_date,
                    "ticker": "000660.KS",
                    "close": 90.0 + idx * 0.5,
                    "ret_1": 0.005,
                    "ret_5": 0.01,
                    "ret_20": 0.03,
                    "rel_ret_5": 0.002,
                    "rel_ret_20": 0.005,
                    "benchmark_ret_20": 0.025,
                    "avg_dollar_volume_20": 900_000.0,
                    "vol_20": 0.12,
                    "benchmark_vol_20": 0.08,
                    "range_20": 0.03,
                    "market_trend_ok": 1.0,
                    "filing_count_20": 0.0,
                    "earnings_filing_count_60": 0.0,
                    "capital_event_count_60": 0.0,
                    "days_since_last_filing": 9_999.0,
                }
            )

        features = pd.DataFrame(rows)
        weights, ranking_history = generate_ridge_target_weights(
            features,
            strategy_config=KRStrategyConfig(top_n=1, min_history_days=20),
            model_config=KRMLModelConfig(label_horizon_days=5, min_training_samples=20, ridge_alpha=1.0),
        )

        self.assertFalse(weights.empty)
        self.assertIn("predicted_return", ranking_history.columns)
        self.assertGreater(float(weights.sum(axis=1).max()), 0.0)

    def test_generate_ridge_walkforward_target_weights_emits_validation_metrics(self) -> None:
        dates = pd.date_range("2025-01-01", periods=220, freq="B")
        rows = []
        for idx, current_date in enumerate(dates):
            rows.append(
                {
                    "date": current_date,
                    "ticker": "005930.KS",
                    "close": 100.0 + idx,
                    "ret_1": 0.01,
                    "ret_5": 0.02,
                    "ret_20": 0.05,
                    "rel_ret_5": 0.01,
                    "rel_ret_20": 0.02,
                    "benchmark_ret_20": 0.025,
                    "avg_dollar_volume_20": 1_000_000.0,
                    "vol_20": 0.10,
                    "benchmark_vol_20": 0.08,
                    "range_20": 0.02,
                    "market_trend_ok": 1.0,
                    "filing_count_20": 0.0,
                    "earnings_filing_count_60": 1.0 if idx > 80 else 0.0,
                    "capital_event_count_60": 0.0,
                    "days_since_last_filing": 5.0,
                }
            )
            rows.append(
                {
                    "date": current_date,
                    "ticker": "000660.KS",
                    "close": 90.0 + idx * 0.5,
                    "ret_1": 0.005,
                    "ret_5": 0.01,
                    "ret_20": 0.03,
                    "rel_ret_5": 0.002,
                    "rel_ret_20": 0.005,
                    "benchmark_ret_20": 0.025,
                    "avg_dollar_volume_20": 900_000.0,
                    "vol_20": 0.12,
                    "benchmark_vol_20": 0.08,
                    "range_20": 0.03,
                    "market_trend_ok": 1.0,
                    "filing_count_20": 0.0,
                    "earnings_filing_count_60": 0.0,
                    "capital_event_count_60": 0.0,
                    "days_since_last_filing": 9999.0,
                }
            )
        features = pd.DataFrame(rows)

        weights, history = generate_ridge_walkforward_target_weights(
            features,
            strategy_config=KRStrategyConfig(top_n=1, min_history_days=20),
            model_config=KRMLModelConfig(
                label_horizon_days=5,
                min_training_samples=40,
                validation_window_days=20,
                min_validation_samples=20,
                ridge_alpha=1.0,
            ),
        )

        self.assertFalse(weights.empty)
        self.assertIn("validation_sample_count", history.columns)
        self.assertIn("validation_mse", history.columns)
        selected = history.loc[history["selected"].fillna(False)]
        self.assertFalse(selected.empty)
        self.assertGreater(float(selected["validation_sample_count"].max()), 0.0)
