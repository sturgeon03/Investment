from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.ticker_signal_audit import build_daily_signal_strategy, build_signal_event_frame


class TickerSignalAuditTests(unittest.TestCase):
    def test_build_signal_event_frame_uses_next_trading_day_after_signal(self) -> None:
        scores = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-03"]),
                "ticker": ["AAA"],
                "llm_score": [0.6],
                "document_count": [1],
                "section_count": [2],
                "avg_confidence": [0.8],
                "avg_risk_flag": [0.1],
            }
        )
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]),
                "close": [100.0, 101.0, 104.0, 106.0],
            }
        )

        events = build_signal_event_frame(scores, prices, "2025-01-01", "2025-01-31", forward_horizons=(1, 2))

        self.assertEqual(len(events), 1)
        self.assertEqual(pd.Timestamp(events.loc[0, "trade_date"]), pd.Timestamp("2025-01-06"))
        self.assertAlmostEqual(float(events.loc[0, "trade_close"]), 101.0)
        self.assertAlmostEqual(float(events.loc[0, "forward_1d_return"]), 104.0 / 101.0 - 1.0)
        self.assertAlmostEqual(float(events.loc[0, "forward_2d_return"]), 106.0 / 101.0 - 1.0)

    def test_build_daily_signal_strategy_follows_positive_scores_until_next_signal(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"]
                ),
                "close": [100.0, 110.0, 99.0, 108.9, 119.79],
            }
        )
        events = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2025-01-03", "2025-01-07"]),
                "trade_date": pd.to_datetime(["2025-01-06", "2025-01-08"]),
                "llm_score": [0.5, -0.2],
            }
        )

        daily = build_daily_signal_strategy(
            prices,
            events,
            "2025-01-06",
            "2025-01-10",
            initial_capital=100_000.0,
            long_threshold=0.0,
        )

        self.assertAlmostEqual(float(daily.loc[0, "target_position"]), 0.5)
        self.assertAlmostEqual(float(daily.loc[1, "live_position"]), 0.5)
        self.assertAlmostEqual(float(daily.loc[2, "target_position"]), 0.0)
        self.assertAlmostEqual(float(daily.loc[2, "live_position"]), 0.5)
        self.assertAlmostEqual(float(daily.loc[3, "live_position"]), 0.0)
        self.assertAlmostEqual(float(daily.loc[2, "strategy_return"]), -0.05)
        self.assertAlmostEqual(float(daily.loc[4, "signal_strategy_value"]), 99_750.0)


if __name__ == "__main__":
    unittest.main()
