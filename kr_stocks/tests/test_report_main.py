from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from invest_ai_core.backtest import BacktestResult
from kr_invest_ai.research import KRResearchRun
from kr_invest_ai.report_main import main


class KRReportMainCLITests(unittest.TestCase):
    def test_report_main_writes_comparison_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            features = pd.DataFrame(
                [
                    {
                        "date": "2026-03-18",
                        "ticker": "005930.KS",
                        "close": 100.0,
                        "ret_1": 0.01,
                        "ret_5": 0.02,
                        "ret_20": 0.03,
                        "rel_ret_5": 0.01,
                        "rel_ret_20": 0.015,
                        "benchmark_ret_20": 0.015,
                        "avg_dollar_volume_20": 1_000_000.0,
                        "vol_20": 0.10,
                        "benchmark_vol_20": 0.08,
                        "range_20": 0.02,
                        "market_trend_ok": 1.0,
                        "filing_count_20": 0.0,
                        "earnings_filing_count_60": 1.0,
                        "capital_event_count_60": 0.0,
                        "days_since_last_filing": 5.0,
                    },
                    {
                        "date": "2026-03-19",
                        "ticker": "005930.KS",
                        "close": 101.0,
                        "ret_1": 0.01,
                        "ret_5": 0.02,
                        "ret_20": 0.03,
                        "rel_ret_5": 0.01,
                        "rel_ret_20": 0.015,
                        "benchmark_ret_20": 0.015,
                        "avg_dollar_volume_20": 1_000_000.0,
                        "vol_20": 0.10,
                        "benchmark_vol_20": 0.08,
                        "range_20": 0.02,
                        "market_trend_ok": 1.0,
                        "filing_count_20": 0.0,
                        "earnings_filing_count_60": 1.0,
                        "capital_event_count_60": 0.0,
                        "days_since_last_filing": 4.0,
                    },
                ]
            )
            prices = pd.DataFrame(
                [
                    {"date": "2026-03-18", "ticker": "005930.KS", "close": 100.0},
                    {"date": "2026-03-19", "ticker": "005930.KS", "close": 101.0},
                ]
            )
            benchmark = pd.DataFrame(
                [
                    {"date": "2026-03-18", "ticker": "069500.KS", "close": 100.0},
                    {"date": "2026-03-19", "ticker": "069500.KS", "close": 100.5},
                ]
            )
            rules_history = pd.DataFrame(
                [
                    {"date": "2026-03-19", "ticker": "005930.KS", "selected": True, "weight": 1.0, "score": 0.5}
                ]
            )
            ridge_history = pd.DataFrame(
                [
                    {
                        "date": "2026-03-19",
                        "ticker": "005930.KS",
                        "selected": True,
                        "weight": 1.0,
                        "predicted_return": 0.04,
                        "train_sample_count": 80,
                        "validation_sample_count": 20,
                        "validation_mse": 0.001,
                    }
                ]
            )
            rules_result = BacktestResult(
                summary=pd.DataFrame([{"total_return": 0.01, "cagr": 0.10, "annual_volatility": 0.05, "sharpe": 1.5, "max_drawdown": -0.02, "avg_daily_turnover": 0.1}]),
                equity_curve=pd.DataFrame({"strategy": [1.0, 1.01], "benchmark": [1.0, 1.005]}, index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
                daily_returns=pd.Series([0.0, 0.01], index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
                turnover=pd.Series([1.0, 0.0], index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
                benchmark_returns=pd.Series([0.0, 0.005], index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
            )
            ridge_result = BacktestResult(
                summary=pd.DataFrame([{"total_return": 0.015, "cagr": 0.12, "annual_volatility": 0.06, "sharpe": 1.6, "max_drawdown": -0.02, "avg_daily_turnover": 0.1}]),
                equity_curve=pd.DataFrame({"strategy": [1.0, 1.015], "benchmark": [1.0, 1.005]}, index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
                daily_returns=pd.Series([0.0, 0.015], index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
                turnover=pd.Series([1.0, 0.0], index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
                benchmark_returns=pd.Series([0.0, 0.005], index=pd.to_datetime(["2026-03-18", "2026-03-19"])),
            )
            research_run = KRResearchRun(
                prices=prices,
                features=features,
                ranking_history=rules_history,
                target_weights=pd.DataFrame({"005930.KS": [1.0]}, index=[pd.Timestamp("2026-03-19")]),
                backtest_result=rules_result,
                benchmark_prices=benchmark,
                raw_run=None,
                manifest={},
            )

            output_dir = Path(temp_dir) / "report"
            argv = [
                "kr_invest_ai.report_main",
                "--tickers",
                "005930",
                "--price-start-date",
                "2026-03-18",
                "--price-end-date",
                "2026-03-19",
                "--benchmark-ticker",
                "069500",
                "--artifacts-dir",
                str(output_dir),
            ]

            with patch("kr_invest_ai.report_main.run_kr_research_pipeline", return_value=research_run):
                with patch("kr_invest_ai.report_main.generate_ridge_walkforward_target_weights", return_value=(research_run.target_weights, ridge_history)):
                    with patch("kr_invest_ai.report_main.run_backtest", return_value=ridge_result):
                        with patch("sys.argv", argv):
                            main()

            summary_path = output_dir / "comparison_summary.csv"
            manifest_path = output_dir / "report_manifest.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue((output_dir / "comparison_values.svg").exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline"], "kr_report_compare")
            self.assertEqual(manifest["benchmark_ticker"], "069500")
            self.assertEqual(manifest["spread_cost_bps"], 0.0)
            self.assertEqual(manifest["market_impact_bps"], 0.0)
            summary = pd.read_csv(summary_path)
            self.assertEqual(set(summary["model_name"]), {"kr_rules_baseline", "kr_ridge_walkforward"})
            self.assertIn("avg_train_sample_count", summary.columns)


if __name__ == "__main__":
    unittest.main()
