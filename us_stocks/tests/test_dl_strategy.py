from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.dl_strategy import MLPModelConfig, fit_mlp_model, predict_mlp_model


class DLStrategyTests(unittest.TestCase):
    def test_fit_mlp_model_learns_simple_relationship(self) -> None:
        train = pd.DataFrame(
            {
                "ret_1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "future_return": [0.0, 2.0, 4.0, 6.0, 8.0],
            }
        )
        validation = pd.DataFrame(
            {
                "ret_1": [1.5, 2.5],
                "future_return": [3.0, 5.0],
            }
        )

        model = fit_mlp_model(
            train_frame=train,
            validation_frame=validation,
            feature_columns=["ret_1"],
            config=MLPModelConfig(
                hidden_dim=8,
                learning_rate=0.02,
                max_epochs=300,
                batch_size=2,
                patience=40,
                random_seed=11,
            ),
        )
        predicted = predict_mlp_model(model, pd.DataFrame({"ret_1": [3.5]}))

        self.assertAlmostEqual(float(predicted[0]), 7.0, delta=0.8)


if __name__ == "__main__":
    unittest.main()
