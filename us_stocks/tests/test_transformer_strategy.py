from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.features import build_features
from us_invest_ai.transformer_strategy import (
    TransformerModelConfig,
    fit_transformer_model,
    generate_transformer_target_weights,
    predict_transformer_model,
)


class TransformerStrategyTests(unittest.TestCase):
    def test_fit_transformer_model_learns_simple_sequence_relationship(self) -> None:
        train_sequences = np.array(
            [
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [4.0]],
                [[3.0], [4.0], [5.0]],
                [[4.0], [5.0], [6.0]],
            ],
            dtype=np.float32,
        )
        train_targets = np.array([3.0, 5.0, 7.0, 9.0, 11.0], dtype=np.float32)
        validation_sequences = np.array(
            [
                [[1.5], [2.5], [3.5]],
                [[2.5], [3.5], [4.5]],
            ],
            dtype=np.float32,
        )
        validation_targets = np.array([6.0, 8.0], dtype=np.float32)

        model = fit_transformer_model(
            train_sequences=train_sequences,
            train_targets=train_targets,
            validation_sequences=validation_sequences,
            validation_targets=validation_targets,
            feature_columns=("ret_1",),
            config=TransformerModelConfig(
                lookback_window=3,
                model_dim=6,
                learning_rate=0.01,
                max_epochs=180,
                batch_size=2,
                patience=30,
                random_seed=23,
            ),
        )
        predicted = predict_transformer_model(
            model,
            np.array([[[3.5], [4.5], [5.5]]], dtype=np.float32),
        )

        self.assertAlmostEqual(float(predicted[0]), 10.0, delta=1.5)
        self.assertGreaterEqual(model.validation_mse, 0.0)

    def test_fit_transformer_model_clips_objective_targets_and_predictions(self) -> None:
        train_sequences = np.array(
            [
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [4.0]],
                [[3.0], [4.0], [5.0]],
                [[4.0], [5.0], [6.0]],
            ],
            dtype=np.float32,
        )
        train_targets = np.array([0.02, 0.03, 0.04, 0.05, 0.60], dtype=np.float32)
        validation_sequences = np.array(
            [
                [[1.5], [2.5], [3.5]],
                [[2.5], [3.5], [4.5]],
            ],
            dtype=np.float32,
        )
        validation_targets = np.array([0.05, 0.55], dtype=np.float32)

        model = fit_transformer_model(
            train_sequences=train_sequences,
            train_targets=train_targets,
            validation_sequences=validation_sequences,
            validation_targets=validation_targets,
            feature_columns=("ret_1",),
            config=TransformerModelConfig(
                lookback_window=3,
                model_dim=4,
                learning_rate=0.01,
                max_epochs=80,
                batch_size=2,
                patience=10,
                random_seed=11,
                target_clip_quantile=0.8,
            ),
        )
        predicted = predict_transformer_model(
            model,
            np.array([[[3.5], [4.5], [5.5]]], dtype=np.float32),
        )

        self.assertEqual(model.target_clip_quantile, 0.8)
        self.assertIsNotNone(model.target_clip_lower)
        self.assertIsNotNone(model.target_clip_upper)
        self.assertLess(float(model.target_clip_upper), 0.60)
        self.assertGreaterEqual(float(predicted[0]), float(model.target_clip_lower) - 1e-9)
        self.assertLessEqual(float(predicted[0]), float(model.target_clip_upper) + 1e-9)

    def test_generate_transformer_target_weights_prefers_stronger_trend_asset(self) -> None:
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
        model_config = TransformerModelConfig(
            label_horizon_days=20,
            min_training_samples=40,
            validation_window_days=10,
            embargo_days=20,
            min_validation_samples=10,
            lookback_window=20,
            model_dim=8,
            max_epochs=80,
            batch_size=64,
            patience=14,
        )

        weights, history = generate_transformer_target_weights(
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

    def test_generate_transformer_target_weights_supports_clipped_objective(self) -> None:
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
        model_config = TransformerModelConfig(
            label_horizon_days=20,
            min_training_samples=40,
            validation_window_days=10,
            embargo_days=20,
            min_validation_samples=10,
            lookback_window=20,
            target_clip_quantile=0.9,
            model_dim=4,
            max_epochs=40,
            batch_size=64,
            patience=8,
        )

        weights, history = generate_transformer_target_weights(
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


if __name__ == "__main__":
    unittest.main()
