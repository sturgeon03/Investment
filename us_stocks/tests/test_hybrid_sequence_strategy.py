from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.features import build_features
from us_invest_ai.hybrid_sequence_strategy import (
    HybridSequenceModelConfig,
    fit_hybrid_sequence_model,
    generate_hybrid_sequence_target_weights,
    predict_hybrid_sequence_model,
)


class HybridSequenceStrategyTests(unittest.TestCase):
    def test_fit_hybrid_sequence_model_uses_static_branch(self) -> None:
        train_sequences = np.array(
            [
                [[1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0]],
            ],
            dtype=np.float32,
        )
        train_static = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        train_targets = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
        validation_sequences = np.array(
            [
                [[1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0]],
            ],
            dtype=np.float32,
        )
        validation_static = np.array([[1.5], [2.5]], dtype=np.float32)
        validation_targets = np.array([4.0, 6.0], dtype=np.float32)

        model = fit_hybrid_sequence_model(
            train_sequences=train_sequences,
            train_static_features=train_static,
            train_targets=train_targets,
            validation_sequences=validation_sequences,
            validation_static_features=validation_static,
            validation_targets=validation_targets,
            sequence_feature_columns=("ret_1",),
            static_feature_columns=("ret_20",),
            config=HybridSequenceModelConfig(
                lookback_window=3,
                kernel_size=2,
                sequence_hidden_channels=4,
                static_hidden_dim=4,
                learning_rate=0.01,
                max_epochs=250,
                batch_size=2,
                patience=40,
                random_seed=17,
            ),
        )
        low_predicted = predict_hybrid_sequence_model(
            model,
            np.array([[[1.0], [1.0], [1.0]]], dtype=np.float32),
            np.array([[0.5]], dtype=np.float32),
        )
        high_predicted = predict_hybrid_sequence_model(
            model,
            np.array([[[1.0], [1.0], [1.0]]], dtype=np.float32),
            np.array([[2.5]], dtype=np.float32),
        )

        self.assertLess(float(low_predicted[0]), float(high_predicted[0]))
        self.assertAlmostEqual(float(high_predicted[0]), 6.0, delta=1.0)
        self.assertGreaterEqual(model.validation_mse, 0.0)

    def test_generate_hybrid_sequence_target_weights_prefers_stronger_trend_asset(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=320)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.append(
                {
                    "date": date,
                    "ticker": "AAA",
                    "close": 100.0 * (1.004 ** index),
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "volume": 1_000,
                }
            )
            records.append(
                {
                    "date": date,
                    "ticker": "BBB",
                    "close": 100.0 * (0.998 ** index),
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "volume": 1_000,
                }
            )
        prices = pd.DataFrame(records)
        features = build_features(prices)

        strategy_config = StrategyConfig(
            rebalance="monthly",
            top_n=1,
            min_history_days=200,
            trend_filter_mode="soft",
            trend_penalty=0.10,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.20,
            llm_weight=0.0,
        )
        model_config = HybridSequenceModelConfig(
            label_horizon_days=20,
            min_training_samples=40,
            validation_window_days=10,
            embargo_days=20,
            min_validation_samples=10,
            sequence_feature_columns=(
                "ret_1",
                "ret_20",
                "vol_20",
                "price_vs_sma50",
                "market_trend_flag",
                "market_high_vol_flag",
            ),
            static_feature_columns=(
                "ret_20",
                "ret_60",
                "vol_20",
                "price_vs_sma50",
                "price_vs_sma200",
                "trend_flag",
                "market_trend_flag",
                "market_high_vol_flag",
            ),
            lookback_window=20,
            kernel_size=5,
            sequence_hidden_channels=6,
            static_hidden_dim=8,
            max_epochs=100,
            batch_size=64,
            patience=15,
        )

        weights, history = generate_hybrid_sequence_target_weights(
            features=features,
            strategy_config=strategy_config,
            model_config=model_config,
            eval_start="2024-11-01",
        )

        last_date = weights.index.max()
        self.assertEqual(weights.loc[last_date, "AAA"], 1.0)
        self.assertEqual(weights.loc[last_date, "BBB"], 0.0)
        selected = history.loc[(history["date"] == last_date) & (history["selected"])]
        self.assertEqual(selected["ticker"].iloc[0], "AAA")
        self.assertGreaterEqual(float(selected["validation_mse"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
