from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.router_dataset import build_router_training_frame


class RouterDatasetTests(unittest.TestCase):
    def test_build_router_training_frame_outputs_expected_columns(self) -> None:
        features = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=70, freq="D"),
                "ticker": ["AAA"] * 70,
                "close": [100 + index for index in range(70)],
                "ret_1": [0.01] * 70,
                "ret_20": [0.05] * 70,
                "ret_60": [0.10] * 70,
                "vol_20": [0.2] * 70,
                "trend_ok": [True] * 70,
            }
        )
        llm_scores = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-10", "2025-01-10", "2025-01-10"]),
                "ticker": ["AAA", "AAA", "AAA"],
                "horizon_bucket": ["short_term", "swing", "long_term"],
                "llm_score": [0.1, 0.2, 0.3],
            }
        )

        dataset = build_router_training_frame(features, llm_scores)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            list(dataset.columns),
            [
                "date",
                "ticker",
                "horizon_bucket",
                "llm_score",
                "price_features",
                "next_5d_return",
                "next_20d_return",
                "next_60d_return",
                "best_realized_horizon_label",
            ],
        )
        self.assertTrue((dataset["best_realized_horizon_label"] == "long_term").all())


if __name__ == "__main__":
    unittest.main()
