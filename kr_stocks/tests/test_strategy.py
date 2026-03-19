from __future__ import annotations

import unittest

import pandas as pd

from kr_invest_ai.strategy import KRStrategyConfig, generate_target_weights


class KRStrategyTests(unittest.TestCase):
    def test_generate_target_weights_prefers_earnings_support_when_momentum_is_close(self) -> None:
        dates = pd.date_range("2026-01-01", periods=25, freq="B")
        rows = []
        for offset, current_date in enumerate(dates, start=1):
            rows.append(
                {
                    "date": current_date,
                    "ticker": "005930.KS",
                    "ret_5": 0.02 + offset * 0.0001,
                    "ret_20": 0.10 + offset * 0.0001,
                    "vol_20": 0.10,
                    "avg_dollar_volume_20": 1_000_000.0,
                    "earnings_filing_count_60": 3.0 if offset == len(dates) else 0.0,
                    "capital_event_count_60": 0.0,
                }
            )
            rows.append(
                {
                    "date": current_date,
                    "ticker": "000660.KS",
                    "ret_5": 0.02 + offset * 0.0001,
                    "ret_20": 0.10 + offset * 0.0001,
                    "vol_20": 0.10,
                    "avg_dollar_volume_20": 1_000_000.0,
                    "earnings_filing_count_60": 0.0,
                    "capital_event_count_60": 0.0,
                }
            )

        features = pd.DataFrame(rows)
        weights, ranking_history = generate_target_weights(
            features,
            KRStrategyConfig(top_n=1, earnings_weight=0.6, momentum_20_weight=0.5),
        )

        last_rebalance = pd.Timestamp(dates.max())
        self.assertEqual(float(weights.loc[last_rebalance, "005930.KS"]), 1.0)
        selected = ranking_history.loc[(ranking_history["date"] == last_rebalance) & (ranking_history["selected"])]
        self.assertEqual(selected["ticker"].iloc[0], "005930.KS")
