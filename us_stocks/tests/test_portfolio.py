from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.config import RiskConfig
from us_invest_ai.portfolio import (
    apply_risk_limits,
    build_rebalance_orders,
    build_target_portfolio,
    latest_prices_by_ticker,
)


class PortfolioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.risk = RiskConfig(
            capital_base=10_000.0,
            cash_buffer=0.10,
            max_position_weight=0.40,
            min_trade_notional=100.0,
            allow_fractional_shares=False,
        )

    def test_apply_risk_limits_caps_and_scales_total_weight(self) -> None:
        weights = pd.Series({"AAA": 0.5, "BBB": 0.5, "CCC": 0.5})

        adjusted = apply_risk_limits(weights, self.risk)

        self.assertAlmostEqual(float(adjusted.sum()), 0.90, places=6)
        self.assertTrue((adjusted <= 0.30 + 1e-9).all())

    def test_build_target_portfolio_uses_latest_available_date(self) -> None:
        target_weights = pd.DataFrame(
            {
                "AAA": [0.6, 0.0],
                "BBB": [0.4, 1.0],
            },
            index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
        )
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-02-28", "2025-02-28"]),
                "ticker": ["AAA", "BBB"],
                "close": [100.0, 50.0],
            }
        )

        portfolio = build_target_portfolio(target_weights, prices, self.risk, as_of_date="2025-03-01")

        self.assertEqual(portfolio.loc[0, "ticker"], "BBB")
        self.assertEqual(portfolio.loc[0, "date"], pd.Timestamp("2025-02-28"))
        self.assertEqual(portfolio.loc[0, "target_shares"], 80.0)
        self.assertAlmostEqual(portfolio.loc[0, "target_weight"], 0.40, places=6)

    def test_build_rebalance_orders_uses_target_date_for_full_exit(self) -> None:
        target_portfolio = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-02-28"]),
                "ticker": ["BBB"],
                "raw_weight": [0.40],
                "target_weight": [0.40],
                "close": [50.0],
                "target_notional": [4_000.0],
                "target_shares": [80.0],
                "target_notional_after_rounding": [4_000.0],
            }
        )
        current_positions = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "shares": [10.0, 20.0],
            }
        )
        price_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-02-28", "2025-02-28"]),
                "ticker": ["AAA", "BBB"],
                "close": [25.0, 50.0],
            }
        )

        orders = build_rebalance_orders(
            target_portfolio,
            current_positions,
            self.risk,
            latest_prices=latest_prices_by_ticker(price_history),
        )

        sell_aaa = orders.loc[orders["ticker"] == "AAA"].iloc[0]
        buy_bbb = orders.loc[orders["ticker"] == "BBB"].iloc[0]
        self.assertEqual(sell_aaa["date"], pd.Timestamp("2025-02-28"))
        self.assertEqual(sell_aaa["side"], "SELL")
        self.assertEqual(buy_bbb["side"], "BUY")


if __name__ == "__main__":
    unittest.main()
