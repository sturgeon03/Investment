from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from invest_ai_core.artifacts import DataFrameArtifact, ensure_output_dir, write_dataframe_artifacts, write_manifest_with_outputs
from invest_ai_core.evaluation import build_backtest_evaluation_row, evaluate_backtest_window
from invest_ai_core.reporting import build_evaluation_row, build_svg_chart, compute_signal_metrics
from invest_ai_core.config import BacktestConfig
from kr_invest_ai.data_bundle import KRResearchDataRequest
from kr_invest_ai.ml_strategy import KRMLModelConfig, generate_ridge_walkforward_target_weights
from kr_invest_ai.research import run_kr_research_pipeline
from kr_invest_ai.strategy import KRStrategyConfig
from invest_ai_core.backtest import run_backtest


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the first KR rules baseline against a first KR walk-forward ridge baseline.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--price-start-date", required=True)
    parser.add_argument("--price-end-date", required=True)
    parser.add_argument("--benchmark-ticker", default=None)
    parser.add_argument("--filings-start-date", default=None)
    parser.add_argument("--filings-end-date", default=None)
    parser.add_argument("--corp-code-map-csv", default=None)
    parser.add_argument("--data-dir", default="kr_stocks/data")
    parser.add_argument("--artifacts-dir", default="kr_stocks/artifacts/report")
    parser.add_argument("--use-dart", action="store_true")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--spread-cost-bps", type=float, default=0.0)
    parser.add_argument("--market-impact-bps", type=float, default=0.0)
    parser.add_argument("--market-impact-exponent", type=float, default=0.5)
    parser.add_argument("--liquidity-lookback-days", type=int, default=20)
    parser.add_argument("--label-horizon-days", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=8.0)
    parser.add_argument("--min-training-samples", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backtest_config = BacktestConfig(
        transaction_cost_bps=args.transaction_cost_bps,
        spread_cost_bps=args.spread_cost_bps,
        market_impact_bps=args.market_impact_bps,
        market_impact_exponent=args.market_impact_exponent,
        liquidity_lookback_days=args.liquidity_lookback_days,
    )
    request = KRResearchDataRequest(
        tickers=tuple(args.tickers),
        price_start_date=date.fromisoformat(args.price_start_date),
        price_end_date=date.fromisoformat(args.price_end_date),
        benchmark_ticker=args.benchmark_ticker,
        filings_start_date=_parse_date(args.filings_start_date),
        filings_end_date=_parse_date(args.filings_end_date),
    )
    research_run = run_kr_research_pipeline(
        request,
        data_dir=args.data_dir,
        corp_code_map_csv=args.corp_code_map_csv,
        use_dart=args.use_dart,
        strategy_config=KRStrategyConfig(top_n=args.top_n),
        backtest_config=backtest_config,
    )

    ridge_weights, ridge_history = generate_ridge_walkforward_target_weights(
        research_run.features,
        strategy_config=KRStrategyConfig(top_n=args.top_n),
        model_config=KRMLModelConfig(
            label_horizon_days=args.label_horizon_days,
            ridge_alpha=args.ridge_alpha,
            min_training_samples=args.min_training_samples,
        ),
    )
    ridge_result = run_backtest(
        prices=research_run.prices,
        target_weights=ridge_weights,
        transaction_cost_bps=backtest_config.transaction_cost_bps,
        benchmark_prices=research_run.benchmark_prices if not research_run.benchmark_prices.empty else None,
        spread_cost_bps=backtest_config.spread_cost_bps,
        market_impact_bps=backtest_config.market_impact_bps,
        market_impact_exponent=backtest_config.market_impact_exponent,
        liquidity_lookback_days=backtest_config.liquidity_lookback_days,
    )

    rules_eval = evaluate_backtest_window(
        research_run.backtest_result,
        pd.to_datetime(research_run.features["date"].min()).normalize(),
        initial_capital=100_000.0,
    )
    ridge_eval = evaluate_backtest_window(
        ridge_result,
        pd.to_datetime(research_run.features["date"].min()).normalize(),
        initial_capital=100_000.0,
    )

    summary_frame = pd.concat(
        [
            build_evaluation_row(
                "kr_rules_baseline",
                rules_eval.summary,
                research_run.ranking_history,
                rules_eval.curve,
                pd.to_datetime(research_run.features["date"].min()).normalize(),
                pd.to_datetime(research_run.features["date"].max()).normalize(),
                100_000.0,
            ),
            build_backtest_evaluation_row(
                "kr_ridge_walkforward",
                ridge_result,
                ridge_history,
                pd.to_datetime(research_run.features["date"].min()).normalize(),
                pd.to_datetime(research_run.features["date"].max()).normalize(),
                initial_capital=100_000.0,
            ),
        ],
        ignore_index=True,
    )

    for model_name, history in (("kr_rules_baseline", research_run.ranking_history), ("kr_ridge_walkforward", ridge_history)):
        signal_coverage, avg_abs = compute_signal_metrics(history)
        summary_frame.loc[summary_frame["model_name"] == model_name, "signal_coverage"] = signal_coverage
        summary_frame.loc[summary_frame["model_name"] == model_name, "avg_llm_abs_score"] = avg_abs

    values_frame = pd.DataFrame({"date": rules_eval.curve["date"]})
    values_frame["kr_rules_baseline_value"] = rules_eval.curve["strategy_value"].to_numpy()
    values_frame["kr_ridge_walkforward_value"] = ridge_eval.curve["strategy_value"].to_numpy()
    if "benchmark_value" in rules_eval.curve.columns:
        values_frame["benchmark_value"] = rules_eval.curve["benchmark_value"].to_numpy()

    output_dir = ensure_output_dir(args.artifacts_dir)
    output_files = write_dataframe_artifacts(
        output_dir,
        [
            DataFrameArtifact("summary", summary_frame, "comparison_summary.csv"),
            DataFrameArtifact("values", values_frame, "comparison_values.csv"),
            DataFrameArtifact("rules_history", research_run.ranking_history, "rules_ranking_history.csv"),
            DataFrameArtifact("ridge_history", ridge_history, "ridge_ranking_history.csv"),
        ],
    )
    chart_path = output_dir / "comparison_values.svg"
    build_svg_chart(
        values_frame,
        benchmark_name="benchmark_value" if "benchmark_value" in values_frame.columns else "kr_rules_baseline_value",
        output_path=chart_path,
        chart_title="KR Research Lane Value Comparison",
    )
    write_manifest_with_outputs(
        output_dir,
        "report_manifest.json",
        {
            "pipeline": "kr_report_compare",
            "benchmark_ticker": request.benchmark_ticker,
            "transaction_cost_bps": backtest_config.transaction_cost_bps,
            "spread_cost_bps": backtest_config.spread_cost_bps,
            "market_impact_bps": backtest_config.market_impact_bps,
            "market_impact_exponent": backtest_config.market_impact_exponent,
            "liquidity_lookback_days": backtest_config.liquidity_lookback_days,
            "label_horizon_days": args.label_horizon_days,
            "ridge_alpha": args.ridge_alpha,
            "min_training_samples": args.min_training_samples,
        },
        {
            **output_files,
            "chart": chart_path,
        },
    )

    print(summary_frame.to_string(index=False))
    print(f"Saved report artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
