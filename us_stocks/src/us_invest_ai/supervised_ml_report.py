from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from invest_ai_core.artifacts import (
    DataFrameArtifact,
    ensure_output_dir,
    write_dataframe_artifacts,
    write_manifest_with_outputs,
)
from invest_ai_core.evaluation import evaluate_backtest_window
from invest_ai_core.reporting import compute_signal_metrics
from us_invest_ai.backtest import run_backtest
from us_invest_ai.config import load_config
from us_invest_ai.data import prepare_market_data_bundle
from us_invest_ai.experiment_manifest import build_run_manifest
from us_invest_ai.features import build_features
from us_invest_ai.ml_strategy import MLModelConfig, generate_ml_target_weights
from us_invest_ai.performance_report import _build_svg
from us_invest_ai.signals import load_llm_scores
from us_invest_ai.strategy import generate_target_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised return model on historical labels and evaluate it on the last year."
    )
    parser.add_argument(
        "--config",
        default="us_stocks/config/soft_price.yaml",
        help="Base config for market data, transaction costs, and portfolio construction.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Calendar-day evaluation window ending on the latest trading day.",
    )
    parser.add_argument(
        "--label-horizon-days",
        type=int,
        default=20,
        help="Forward return horizon used as the training label.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=8.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--min-training-samples",
        type=int,
        default=252,
        help="Minimum labeled observations required before making live predictions.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital for the value curve.",
    )
    parser.add_argument(
        "--use-llm-feature",
        action="store_true",
        help="Include the attached llm_score column as one ML feature when the config enables LLM.",
    )
    parser.add_argument(
        "--output-dir",
        default="us_stocks/artifacts/supervised_ml_last_year",
        help="Directory for the summary, value curves, and SVG chart.",
    )
    return parser.parse_args()


def _signal_metrics(ranking_history: pd.DataFrame) -> tuple[float, float]:
    return compute_signal_metrics(ranking_history)


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    market_data = prepare_market_data_bundle(
        data_dir=config.output.data_dir,
        tickers=config.data.tickers,
        benchmark=config.data.benchmark,
        start=config.data.start,
        end=config.data.end,
        tickers_file=config.data.tickers_file,
        metadata_file=config.data.metadata_file,
        universe_snapshots_file=config.data.universe_snapshots_file,
    )
    prices = market_data.prices
    benchmark_prices = market_data.benchmark_prices
    features = build_features(
        prices,
        benchmark_prices,
        market_data.ticker_metadata,
        market_data.universe_snapshots,
        {
            "min_close_price": config.eligibility.min_close_price,
            "min_dollar_volume_20": config.eligibility.min_dollar_volume_20,
            "min_universe_age_days": config.eligibility.min_universe_age_days,
        },
    )

    latest_date = pd.to_datetime(prices["date"]).max().normalize()
    eval_start = latest_date - pd.Timedelta(days=args.lookback_days)

    llm_scores = None
    if config.llm.enabled:
        llm_scores = load_llm_scores(config.llm.signal_path, config.llm.horizon_bucket)

    ml_weights, ml_history = generate_ml_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=MLModelConfig(
            label_horizon_days=args.label_horizon_days,
            ridge_alpha=args.ridge_alpha,
            min_training_samples=args.min_training_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
        ),
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    ml_result = run_backtest(
        prices=prices,
        target_weights=ml_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )

    baseline_weights, baseline_history = generate_target_weights(features, config.strategy, llm_scores)
    baseline_result = run_backtest(
        prices=prices,
        target_weights=baseline_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )

    ml_returns = ml_result.daily_returns.loc[ml_result.daily_returns.index >= eval_start]
    ml_turnover = ml_result.turnover.reindex(ml_returns.index)
    ml_benchmark = (
        ml_result.benchmark_returns.reindex(ml_returns.index) if ml_result.benchmark_returns is not None else None
    )
    baseline_returns = baseline_result.daily_returns.loc[baseline_result.daily_returns.index >= eval_start]
    baseline_turnover = baseline_result.turnover.reindex(baseline_returns.index)
    baseline_benchmark = (
        baseline_result.benchmark_returns.reindex(baseline_returns.index)
        if baseline_result.benchmark_returns is not None
        else None
    )

    ml_evaluation = evaluate_backtest_window(ml_result, eval_start, initial_capital=args.initial_capital)
    baseline_evaluation = evaluate_backtest_window(
        baseline_result,
        eval_start,
        initial_capital=args.initial_capital,
    )
    ml_summary = ml_evaluation.summary
    baseline_summary = baseline_evaluation.summary

    ml_curve = ml_evaluation.curve
    baseline_curve = baseline_evaluation.curve
    eval_end = pd.to_datetime(ml_returns.index.max()).normalize()
    eval_start_actual = pd.to_datetime(ml_returns.index.min()).normalize()

    summary_rows: list[pd.DataFrame] = []
    for model_name, summary, history, curve in (
        ("supervised_ml", ml_summary, ml_history.loc[ml_history["date"] >= eval_start_actual], ml_curve),
        ("configured_baseline", baseline_summary, baseline_history.loc[baseline_history["date"] >= eval_start_actual], baseline_curve),
    ):
        signal_coverage, avg_llm_abs_score = _signal_metrics(history)
        ending_capital = float(curve["strategy_value"].iloc[-1])
        row = summary.copy()
        row.insert(0, "model_name", model_name)
        row["eval_start"] = eval_start_actual.date().isoformat()
        row["eval_end"] = eval_end.date().isoformat()
        row["starting_capital"] = args.initial_capital
        row["ending_capital"] = ending_capital
        row["profit_dollars"] = ending_capital - args.initial_capital
        row["signal_coverage"] = signal_coverage
        row["avg_llm_abs_score"] = avg_llm_abs_score
        if model_name == "supervised_ml":
            selected = history.loc[history["selected"]]
            row["avg_train_sample_count"] = float(selected["train_sample_count"].mean()) if not selected.empty else 0.0
        else:
            row["avg_train_sample_count"] = np.nan
        summary_rows.append(row)

    summary_frame = pd.concat(summary_rows, ignore_index=True)
    values_frame = pd.DataFrame({"date": ml_curve["date"]})
    values_frame["supervised_ml_value"] = ml_curve["strategy_value"].to_numpy()
    values_frame["configured_baseline_value"] = baseline_curve["strategy_value"].to_numpy()
    if "benchmark_value" in ml_curve.columns:
        values_frame["benchmark_value"] = ml_curve["benchmark_value"].to_numpy()
    output_dir = ensure_output_dir(args.output_dir)
    output_files = write_dataframe_artifacts(
        output_dir,
        [
            DataFrameArtifact("summary", summary_frame, "supervised_ml_summary_last_year.csv"),
            DataFrameArtifact("values", values_frame, "supervised_ml_values_last_year.csv"),
            DataFrameArtifact(
                "ranking_history",
                ml_history.loc[ml_history["date"] >= eval_start_actual],
                "supervised_ml_ranking_history_last_year.csv",
            ),
        ],
    )
    summary_path = output_files["summary"]
    values_path = output_files["values"]
    ml_history_path = output_files["ranking_history"]

    chart_path = output_dir / "supervised_ml_values_last_year.svg"
    benchmark_name = "benchmark_value" if "benchmark_value" in values_frame.columns else "configured_baseline_value"
    _build_svg(values_frame, benchmark_name=benchmark_name, output_path=chart_path)
    manifest = build_run_manifest(
        config,
        experiment_name="supervised_ml_report",
        extra={
            "lookback_days": args.lookback_days,
            "label_horizon_days": args.label_horizon_days,
            "ridge_alpha": args.ridge_alpha,
            "min_training_samples": args.min_training_samples,
            "use_llm_feature": args.use_llm_feature and config.llm.enabled,
            "latest_market_date": latest_date.date().isoformat(),
            "market_data_source": market_data.provenance.get("source"),
            "market_data_manifest_path": market_data.provenance.get("manifest_path"),
            "market_data_manifest_sha256": market_data.provenance.get("manifest_sha256"),
        },
    )
    write_manifest_with_outputs(
        output_dir,
        "report_manifest.json",
        manifest,
        {
            "summary": summary_path,
            "values": values_path,
            "chart": chart_path,
            "ranking_history": ml_history_path,
        },
    )

    print(summary_frame.to_string(index=False))
    print(f"Latest market date: {latest_date.date().isoformat()}")
    print(f"Evaluation window: {eval_start_actual.date().isoformat()} to {eval_end.date().isoformat()}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved values to: {values_path}")
    print(f"Saved chart to: {chart_path}")
    print(f"Saved ranking history to: {ml_history_path}")


if __name__ == "__main__":
    main()
