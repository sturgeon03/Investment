from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.features import build_features
from us_invest_ai.ml_strategy import (
    MLModelConfig,
    fit_ridge_model,
    generate_ml_target_weights,
    generate_ridge_walkforward_target_weights,
    predict_ridge_model,
)


class MLStrategyTests(unittest.TestCase):
    def test_fit_ridge_model_learns_simple_relationship(self) -> None:
        train = pd.DataFrame(
            {
                "ret_1": [0.0, 1.0, 2.0, 3.0],
                "future_return": [0.0, 2.0, 4.0, 6.0],
            }
        )

        model = fit_ridge_model(train, ["ret_1"], alpha=0.001)
        predicted = predict_ridge_model(model, pd.DataFrame({"ret_1": [4.0]}))

        self.assertAlmostEqual(float(predicted[0]), 8.0, delta=0.2)

    def test_generate_ml_target_weights_prefers_stronger_trend_asset(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=320)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.append(
                {"date": date, "ticker": "AAA", "close": 100.0 * (1.004 ** index), "open": 0.0, "high": 0.0, "low": 0.0, "volume": 1_000}
            )
            records.append(
                {"date": date, "ticker": "BBB", "close": 100.0 * (0.998 ** index), "open": 0.0, "high": 0.0, "low": 0.0, "volume": 1_000}
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
        model_config = MLModelConfig(
            label_horizon_days=20,
            ridge_alpha=1.0,
            min_training_samples=120,
        )

        weights, history = generate_ml_target_weights(
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

    def test_generate_ridge_walkforward_target_weights_prefers_stronger_trend_asset(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=320)
        records: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            records.append(
                {"date": date, "ticker": "AAA", "close": 100.0 * (1.004 ** index), "open": 0.0, "high": 0.0, "low": 0.0, "volume": 1_000}
            )
            records.append(
                {"date": date, "ticker": "BBB", "close": 100.0 * (0.998 ** index), "open": 0.0, "high": 0.0, "low": 0.0, "volume": 1_000}
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
        model_config = MLModelConfig(
            label_horizon_days=20,
            ridge_alpha=1.0,
            min_training_samples=120,
            validation_window_days=20,
            embargo_days=20,
            min_validation_samples=20,
        )

        weights, history = generate_ridge_walkforward_target_weights(
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
