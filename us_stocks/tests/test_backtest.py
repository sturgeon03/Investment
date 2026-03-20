from __future__ import annotations

import unittest

import pandas as pd

import invest_ai_core.performance as shared_performance
from us_invest_ai.backtest import build_summary, run_backtest
from us_invest_ai.config import BacktestConfig, RiskConfig


class BacktestTests(unittest.TestCase):
    def test_shared_performance_summary_is_compatible_with_us_wrapper(self) -> None:
        self.assertIs(shared_performance.build_summary, build_summary)

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

    def test_backtest_applies_risk_limits_to_live_weights(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "ticker": ["AAA", "AAA"],
                "close": [100.0, 110.0],
            }
        )
        weights = pd.DataFrame(
            {"AAA": [1.0, 1.0]},
            index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        )
        risk = RiskConfig(
            capital_base=100_000.0,
            cash_buffer=0.10,
            max_position_weight=1.0,
            min_trade_notional=100.0,
            allow_fractional_shares=True,
        )

        result = run_backtest(prices, weights, transaction_cost_bps=0.0, risk_config=risk)

        self.assertAlmostEqual(result.turnover.iloc[0], 0.90, places=6)
        self.assertAlmostEqual(result.daily_returns.iloc[1], 0.09, places=6)

    def test_build_summary_reports_downside_and_active_metrics(self) -> None:
        strategy_returns = pd.Series(
            [0.01, -0.02, 0.015, -0.005],
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
        )
        benchmark_returns = pd.Series(
            [0.005, -0.01, 0.01, -0.004],
            index=strategy_returns.index,
        )
        turnover = pd.Series([0.9, 0.0, 0.1, 0.0], index=strategy_returns.index)

        summary = build_summary(strategy_returns, turnover, benchmark_returns)

        self.assertIn("sortino", summary.columns)
        self.assertIn("calmar", summary.columns)
        self.assertIn("information_ratio", summary.columns)
        self.assertIn("tracking_error", summary.columns)
        self.assertGreater(float(summary.loc[0, "downside_volatility"]), 0.0)

    def test_backtest_tracks_spread_and_market_impact_costs(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                "ticker": ["AAA", "AAA", "AAA"],
                "close": [100.0, 110.0, 110.0],
                "volume": [1_000.0, 1_000.0, 1_000.0],
            }
        )
        weights = pd.DataFrame(
            {"AAA": [1.0, 1.0, 0.0]},
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )

        result = run_backtest(
            prices,
            weights,
            backtest_config=BacktestConfig(
                transaction_cost_bps=10.0,
                spread_cost_bps=5.0,
                market_impact_bps=20.0,
                market_impact_exponent=1.0,
                liquidity_lookback_days=1,
            ),
        )

        self.assertAlmostEqual(result.gross_daily_returns.iloc[0], 0.0)
        self.assertAlmostEqual(result.gross_daily_returns.iloc[1], 0.10)
        self.assertAlmostEqual(result.linear_costs.iloc[0], 0.001)
        self.assertAlmostEqual(result.spread_costs.iloc[0], 0.0005)
        self.assertAlmostEqual(result.market_impact_costs.iloc[0], 0.002)
        self.assertAlmostEqual(result.max_participation_rate.iloc[0], 1.0)
        self.assertIn("gross_total_return", result.summary.columns)
        self.assertIn("avg_daily_total_cost", result.summary.columns)
        self.assertGreater(float(result.summary.loc[0, "cost_drag_total_return"]), 0.0)


if __name__ == "__main__":
    unittest.main()
