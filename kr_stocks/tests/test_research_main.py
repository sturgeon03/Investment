from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from invest_ai_core.backtest import BacktestResult
from kr_invest_ai.research import KRResearchRun
from kr_invest_ai.research_main import main


class KRResearchMainCLITests(unittest.TestCase):
    def test_research_main_passes_benchmark_ticker_and_saves_outputs(self) -> None:
        research_run = KRResearchRun(
            prices=pd.DataFrame([{"date": "2026-03-19", "ticker": "005930.KS", "close": 70000.0}]),
            features=pd.DataFrame([{"date": "2026-03-19", "ticker": "005930.KS"}]),
            ranking_history=pd.DataFrame([{"date": "2026-03-19", "ticker": "005930.KS", "selected": True, "weight": 1.0}]),
            target_weights=pd.DataFrame({"005930.KS": [1.0]}, index=[pd.Timestamp("2026-03-19")]),
            backtest_result=BacktestResult(
                summary=pd.DataFrame([{"total_return": 0.01, "cagr": 0.10, "annual_volatility": 0.05, "sharpe": 1.0, "max_drawdown": -0.02, "avg_daily_turnover": 0.1}]),
                equity_curve=pd.DataFrame({"strategy": [1.0]}, index=pd.to_datetime(["2026-03-19"])),
                daily_returns=pd.Series([0.0], index=pd.to_datetime(["2026-03-19"])),
                turnover=pd.Series([1.0], index=pd.to_datetime(["2026-03-19"])),
                benchmark_returns=None,
            ),
            benchmark_prices=pd.DataFrame([{"date": "2026-03-19", "ticker": "069500.KS", "close": 30000.0}]),
            raw_run=None,
            manifest={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "research"
            argv = [
                "kr_invest_ai.research_main",
                "--tickers",
                "005930",
                "--price-start-date",
                "2026-03-01",
                "--price-end-date",
                "2026-03-19",
                "--benchmark-ticker",
                "069500",
                "--artifacts-dir",
                str(output_dir),
            ]

            with patch("kr_invest_ai.research_main.run_kr_research_pipeline", return_value=research_run) as mocked_run:
                with patch("kr_invest_ai.research_main.save_kr_research_outputs", return_value=output_dir) as mocked_save:
                    with patch("sys.argv", argv):
                        main()

            request = mocked_run.call_args.args[0]
            self.assertEqual(request.benchmark_ticker, "069500")
            self.assertEqual(mocked_save.call_args.kwargs["output_dir"], str(output_dir))


if __name__ == "__main__":
    unittest.main()
