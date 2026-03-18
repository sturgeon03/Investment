from __future__ import annotations

import unittest

import pandas as pd

from us_invest_ai.stability_report import _build_evaluation_windows


class StabilityReportTests(unittest.TestCase):
    def test_build_evaluation_windows_returns_chronological_windows(self) -> None:
        trading_dates = list(pd.bdate_range("2024-01-01", periods=600))

        windows = _build_evaluation_windows(
            trading_dates=trading_dates,
            window_trading_days=252,
            step_trading_days=126,
            window_count=3,
        )

        self.assertEqual(len(windows), 3)
        self.assertLess(pd.Timestamp(windows[0]["eval_end"]), pd.Timestamp(windows[-1]["eval_end"]))
        self.assertEqual(
            windows[-1]["window_label"],
            f"{pd.Timestamp(windows[-1]['eval_start']).date().isoformat()}_to_{pd.Timestamp(windows[-1]['eval_end']).date().isoformat()}",
        )

    def test_build_evaluation_windows_requires_sufficient_history(self) -> None:
        trading_dates = list(pd.bdate_range("2024-01-01", periods=120))

        with self.assertRaises(ValueError):
            _build_evaluation_windows(
                trading_dates=trading_dates,
                window_trading_days=252,
                step_trading_days=126,
                window_count=2,
            )


if __name__ == "__main__":
    unittest.main()
