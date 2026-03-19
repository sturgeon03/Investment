from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import invest_ai_core.artifacts as shared_artifacts
import invest_ai_core.evaluation as shared_evaluation
import invest_ai_core.reporting as shared_reporting
import invest_ai_core.runtime as shared_runtime
from us_invest_ai.deep_learning_report import _build_summary_row
from us_invest_ai.performance_report import _build_svg, _build_value_curve


class ReportingCoreTests(unittest.TestCase):
    class _StubResult:
        def __init__(self) -> None:
            index = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
            self.daily_returns = pd.Series([0.0, 0.10, -0.05], index=index)
            self.turnover = pd.Series([1.0, 0.2, 0.1], index=index)
            self.benchmark_returns = pd.Series([0.0, 0.02, 0.01], index=index)

    def test_shared_value_curve_is_compatible_with_us_wrapper(self) -> None:
        self.assertIs(shared_reporting.build_value_curve, _build_value_curve)

    def test_build_value_curve_keeps_expected_columns(self) -> None:
        index = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
        strategy_returns = pd.Series([0.0, 0.10, -0.05], index=index)
        benchmark_returns = pd.Series([0.0, 0.02, 0.01], index=index)

        curve = shared_reporting.build_value_curve(
            strategy_returns,
            benchmark_returns,
            initial_capital=100_000.0,
        )

        self.assertEqual(list(curve.columns), ["date", "strategy_value", "benchmark_value"])
        self.assertEqual(float(curve["strategy_value"].iloc[0]), 100_000.0)
        self.assertEqual(float(curve["benchmark_value"].iloc[0]), 100_000.0)

    def test_build_value_curve_omits_benchmark_column_when_missing(self) -> None:
        index = pd.to_datetime(["2025-01-01", "2025-01-02"])
        strategy_returns = pd.Series([0.0, 0.05], index=index)

        curve = shared_reporting.build_value_curve(
            strategy_returns,
            benchmark_returns=None,
            initial_capital=50_000.0,
        )

        self.assertEqual(list(curve.columns), ["date", "strategy_value"])
        self.assertEqual(float(curve["strategy_value"].iloc[-1]), 52_500.0)

    def test_compute_signal_metrics_handles_missing_column(self) -> None:
        metrics = shared_reporting.compute_signal_metrics(pd.DataFrame({"date": ["2025-01-01"]}))
        self.assertEqual(metrics, (0.0, 0.0))

    def test_compute_signal_metrics_measures_coverage_and_abs_score(self) -> None:
        history = pd.DataFrame({"llm_score": [0.0, 1.5, -0.5, 0.0]})
        signal_coverage, avg_abs = shared_reporting.compute_signal_metrics(history)
        self.assertEqual(signal_coverage, 0.5)
        self.assertEqual(avg_abs, 0.5)

    def test_shared_evaluation_row_is_compatible_with_us_wrapper(self) -> None:
        self.assertIs(shared_reporting.build_evaluation_row, _build_summary_row)

    def test_build_evaluation_row_adds_selected_metrics_and_window_metadata(self) -> None:
        summary = pd.DataFrame({"cagr": [0.12], "sharpe": [1.4]})
        curve = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "strategy_value": [100_000.0, 108_000.0],
            }
        )
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "selected": [True, False],
                "train_sample_count": [240, 260],
                "validation_sample_count": [120, 140],
                "validation_mse": [0.04, 0.08],
            }
        )

        row = shared_reporting.build_evaluation_row(
            model_name="transformer_walkforward",
            summary=summary,
            history=history,
            curve=curve,
            eval_start=pd.Timestamp("2025-01-02"),
            eval_end=pd.Timestamp("2025-01-03"),
            initial_capital=100_000.0,
            window_label="window_1",
            include_rebalance_count=True,
            extra={"objective_name": "clip_q95"},
        )

        self.assertEqual(row.loc[0, "model_name"], "transformer_walkforward")
        self.assertEqual(row.loc[0, "window_label"], "window_1")
        self.assertEqual(row.loc[0, "eval_start"], "2025-01-02")
        self.assertEqual(row.loc[0, "eval_end"], "2025-01-03")
        self.assertEqual(float(row.loc[0, "ending_capital"]), 108_000.0)
        self.assertEqual(float(row.loc[0, "profit_dollars"]), 8_000.0)
        self.assertEqual(int(row.loc[0, "rebalance_count"]), 2)
        self.assertEqual(float(row.loc[0, "avg_train_sample_count"]), 240.0)
        self.assertEqual(float(row.loc[0, "avg_validation_sample_count"]), 120.0)
        self.assertEqual(float(row.loc[0, "avg_validation_mse"]), 0.04)
        self.assertEqual(row.loc[0, "objective_name"], "clip_q95")

    def test_build_evaluation_row_returns_nan_metrics_without_selected_rows(self) -> None:
        summary = pd.DataFrame({"cagr": [0.05]})
        curve = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "strategy_value": [101_000.0]})
        history = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "selected": [False]})

        row = shared_reporting.build_evaluation_row(
            model_name="configured_baseline",
            summary=summary,
            history=history,
            curve=curve,
            eval_start=pd.Timestamp("2025-01-02"),
            eval_end=pd.Timestamp("2025-01-02"),
            initial_capital=100_000.0,
        )

        self.assertTrue(pd.isna(row.loc[0, "avg_train_sample_count"]))
        self.assertTrue(pd.isna(row.loc[0, "avg_validation_sample_count"]))
        self.assertTrue(pd.isna(row.loc[0, "avg_validation_mse"]))

    def test_svg_wrapper_writes_expected_title(self) -> None:
        curves = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "configured_baseline_value": [100_000.0, 101_000.0],
                "benchmark_value": [100_000.0, 100_500.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chart.svg"
            _build_svg(curves, benchmark_name="benchmark_value", output_path=output_path)
            contents = output_path.read_text(encoding="utf-8")

        self.assertIn("US Stocks Strategy Value - Last 2 Trading Days", contents)
        self.assertIn("configured_baseline", contents)
        self.assertIn("benchmark", contents)

    def test_evaluate_backtest_window_builds_curve_and_summary(self) -> None:
        evaluation = shared_evaluation.evaluate_backtest_window(
            self._StubResult(),
            pd.Timestamp("2025-01-02"),
            initial_capital=100_000.0,
        )

        self.assertEqual(len(evaluation.returns), 2)
        self.assertIn("sharpe", evaluation.summary.columns)
        self.assertEqual(list(evaluation.curve.columns), ["date", "strategy_value", "benchmark_value"])

    def test_build_backtest_evaluation_row_uses_requested_window(self) -> None:
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
                "selected": [False, True, True],
                "train_sample_count": [10, 20, 30],
            }
        )

        row = shared_evaluation.build_backtest_evaluation_row(
            model_name="stub",
            result=self._StubResult(),
            history=history,
            eval_start=pd.Timestamp("2025-01-02"),
            eval_end=pd.Timestamp("2025-01-03"),
            initial_capital=100_000.0,
            window_label="window_a",
            include_rebalance_count=True,
        )

        self.assertEqual(row.loc[0, "window_label"], "window_a")
        self.assertEqual(row.loc[0, "rebalance_count"], 2)
        self.assertEqual(row.loc[0, "avg_train_sample_count"], 25.0)

    def test_write_dataframe_artifacts_saves_named_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = shared_artifacts.write_dataframe_artifacts(
                temp_dir,
                [
                    shared_artifacts.DataFrameArtifact(
                        "summary",
                        pd.DataFrame({"value": [1, 2]}),
                        "summary.csv",
                    ),
                    shared_artifacts.DataFrameArtifact(
                        "weights",
                        pd.DataFrame({"weight": [0.25]}),
                        "weights.csv",
                    ),
                ],
            )

            self.assertEqual(sorted(output_files), ["summary", "weights"])
            self.assertTrue(output_files["summary"].exists())
            self.assertTrue(output_files["weights"].exists())

    def test_write_manifest_with_outputs_records_output_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = shared_artifacts.write_dataframe_artifacts(
                temp_dir,
                [
                    shared_artifacts.DataFrameArtifact(
                        "summary",
                        pd.DataFrame({"value": [1]}),
                        "summary.csv",
                    ),
                ],
            )
            manifest_path = shared_artifacts.write_manifest_with_outputs(
                temp_dir,
                "report_manifest.json",
                {"experiment_name": "reporting_core_test"},
                output_files,
            )
            manifest = pd.read_json(manifest_path, typ="series").to_dict()

            self.assertEqual(manifest["experiment_name"], "reporting_core_test")
            self.assertIn("summary", manifest["output_files"])
            self.assertTrue(manifest["output_files"]["summary"]["exists"])

    def test_write_research_artifact_bundle_saves_standard_research_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = shared_runtime.write_research_artifact_bundle(
                temp_dir,
                shared_runtime.ResearchArtifactBundle(
                    features=pd.DataFrame({"date": ["2025-01-01"], "ticker": ["AAA"]}),
                    ranking_history=pd.DataFrame({"date": ["2025-01-01"], "ticker": ["AAA"]}),
                    target_weights=pd.DataFrame({"AAA": [1.0]}, index=pd.to_datetime(["2025-01-01"])),
                    equity_curve=pd.DataFrame({"strategy": [1.0]}, index=pd.to_datetime(["2025-01-01"])),
                    summary=pd.DataFrame({"cagr": [0.1]}),
                ),
            )

            self.assertIn("features", output_files)
            self.assertIn("equity_curve", output_files)
            self.assertTrue(output_files["target_weights"].exists())


if __name__ == "__main__":
    unittest.main()
