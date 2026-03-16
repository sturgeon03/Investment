from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.backtest import run_backtest


class BacktestTests(unittest.TestCase):
    def test_backtest_uses_previous_day_weights(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
                ),
                "ticker": ["AAA", "AAA", "AAA", "AAA"],
                "close": [100.0, 110.0, 121.0, 133.1],
            }
        )
        weights = pd.DataFrame(
            {"AAA": [1.0, 1.0, 1.0, 1.0]},
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        )

        result = run_backtest(prices, weights, transaction_cost_bps=0.0)

        self.assertAlmostEqual(result.daily_returns.iloc[0], 0.0)
        self.assertAlmostEqual(result.daily_returns.iloc[1], 0.10)
        self.assertAlmostEqual(result.daily_returns.iloc[2], 0.10)
        self.assertAlmostEqual(result.daily_returns.iloc[3], 0.10)


if __name__ == "__main__":
    unittest.main()
