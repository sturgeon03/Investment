from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.compare_configs import _changed_rebalance_count, _signal_metrics


class CompareConfigTests(unittest.TestCase):
    def test_signal_metrics_report_coverage_and_average_abs_signal(self) -> None:
        ranking_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28"]),
                "ticker": ["AAA", "BBB", "AAA"],
                "llm_score": [0.0, 0.4, -0.2],
            }
        )

        coverage, avg_abs = _signal_metrics(ranking_history)

        self.assertAlmostEqual(coverage, 2 / 3, places=4)
        self.assertAlmostEqual(avg_abs, 0.2, places=4)

    def test_changed_rebalance_count_detects_weight_changes(self) -> None:
        base_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"]),
                "ticker": ["AAA", "BBB", "AAA", "BBB"],
                "weight": [1.0, 0.0, 1.0, 0.0],
            }
        )
        candidate_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"]),
                "ticker": ["AAA", "BBB", "AAA", "BBB"],
                "weight": [1.0, 0.0, 0.0, 1.0],
            }
        )

        self.assertEqual(_changed_rebalance_count(base_history, candidate_history), 1)


if __name__ == "__main__":
    unittest.main()
