from __future__ import annotations

import argparse
from dataclasses import asdict
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
from invest_ai_core.reporting import (
    build_svg_chart,
    build_value_curve as _build_value_curve,
    compute_signal_metrics,
)
from us_invest_ai.backtest import run_backtest
from us_invest_ai.config import DataConfig, EligibilityConfig, load_config
from us_invest_ai.data import prepare_market_data_bundle
from us_invest_ai.experiment_manifest import build_run_manifest
from us_invest_ai.features import build_features
from us_invest_ai.signals import load_llm_scores
from us_invest_ai.strategy import generate_target_weights


DEFAULT_CONFIGS = [
    "us_stocks/config/base.yaml",
    "us_stocks/config/soft_price.yaml",
    "us_stocks/config/with_llm_short.yaml",
    "us_stocks/config/with_llm_swing.yaml",
    "us_stocks/config/with_llm_long.yaml",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a last-year performance report for one or more US stocks strategy configs."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="One or more config files to compare.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Calendar-day lookback window ending on the latest downloaded trading day.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital used to convert returns into portfolio value.",
    )
    parser.add_argument(
        "--output-dir",
        default="us_stocks/artifacts/last_year_report",
        help="Directory where the comparison table, value curves, and SVG chart are saved.",
    )
    return parser.parse_args()


def _assert_matching_market_data(
    reference: DataConfig,
    reference_eligibility: EligibilityConfig,
    candidate: DataConfig,
    candidate_eligibility: EligibilityConfig,
    config_path: Path,
) -> None:
    if asdict(reference) != asdict(candidate) or asdict(reference_eligibility) != asdict(candidate_eligibility):
        raise ValueError(
            f"Config {config_path} does not match the market data settings of the first config. "
            "Compare configs only when data, universe, and eligibility settings are identical."
        )


def _slice_ranking_history(ranking_history: pd.DataFrame, eval_start: pd.Timestamp) -> pd.DataFrame:
    history = ranking_history.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    return history.loc[history["date"] >= eval_start].copy()


def _signal_metrics(ranking_history: pd.DataFrame) -> tuple[float, float]:
    return compute_signal_metrics(ranking_history)


def _changed_rebalance_count(base_history: pd.DataFrame, candidate_history: pd.DataFrame) -> int:
    if base_history.empty or candidate_history.empty:
        return 0

    base_weights = (
        base_history.pivot_table(index="date", columns="ticker", values="weight", fill_value=0.0)
        .sort_index()
        .sort_index(axis=1)
    )
    candidate_weights = (
        candidate_history.pivot_table(index="date", columns="ticker", values="weight", fill_value=0.0)
        .sort_index()
        .sort_index(axis=1)
    )
    common_index = sorted(set(base_weights.index).intersection(candidate_weights.index))
    common_columns = sorted(set(base_weights.columns).union(candidate_weights.columns))
    if not common_index:
        return 0

    base_aligned = base_weights.reindex(index=common_index, columns=common_columns, fill_value=0.0)
    candidate_aligned = candidate_weights.reindex(index=common_index, columns=common_columns, fill_value=0.0)
    return int(
        sum(
            not np.allclose(base_aligned.loc[date].to_numpy(), candidate_aligned.loc[date].to_numpy())
            for date in common_index
        )
    )


def _build_svg(curves: pd.DataFrame, benchmark_name: str, output_path: Path) -> None:
    build_svg_chart(
        curves,
        benchmark_name=benchmark_name,
        output_path=output_path,
        chart_title=f"US Stocks Strategy Value - Last {len(curves)} Trading Days",
    )


def main() -> None:
    args = _parse_args()
    configs = [load_config(config_path) for config_path in args.configs]
    if not configs:
        raise ValueError("At least one config is required.")

    reference_data = configs[0].data
    reference_eligibility = configs[0].eligibility
    for config_path, config in zip(args.configs[1:], configs[1:]):
        _assert_matching_market_data(
            reference_data,
            reference_eligibility,
            config.data,
            config.eligibility,
            Path(config_path),
        )

    market_data = prepare_market_data_bundle(
        data_dir=configs[0].output.data_dir,
        tickers=reference_data.tickers,
        benchmark=reference_data.benchmark,
        start=reference_data.start,
        end=reference_data.end,
        tickers_file=configs[0].data.tickers_file,
        metadata_file=configs[0].data.metadata_file,
        universe_snapshots_file=configs[0].data.universe_snapshots_file,
    )
    prices = market_data.prices
    benchmark_prices = market_data.benchmark_prices
    features = build_features(
        prices,
        benchmark_prices,
        market_data.ticker_metadata,
        market_data.universe_snapshots,
        {
            "min_close_price": configs[0].eligibility.min_close_price,
            "min_dollar_volume_20": configs[0].eligibility.min_dollar_volume_20,
            "min_universe_age_days": configs[0].eligibility.min_universe_age_days,
        },
    )

    latest_date = pd.to_datetime(prices["date"]).max().normalize()
    requested_eval_start = latest_date - pd.Timedelta(days=args.lookback_days)

    runs: list[dict[str, object]] = []
    for config_path, config in zip(args.configs, configs):
        llm_scores = (
            load_llm_scores(config.llm.signal_path, config.llm.horizon_bucket)
            if config.llm.enabled
            else None
        )
        weights, ranking_history = generate_target_weights(features, config.strategy, llm_scores)
        result = run_backtest(
            prices=prices,
            target_weights=weights,
            backtest_config=config.backtest,
            benchmark_prices=benchmark_prices,
            risk_config=config.risk,
        )

        ranking_slice = _slice_ranking_history(ranking_history, requested_eval_start)
        evaluation = evaluate_backtest_window(
            result,
            requested_eval_start,
            initial_capital=args.initial_capital,
        )
        actual_eval_start = pd.to_datetime(evaluation.returns.index.min()).normalize()
        actual_eval_end = pd.to_datetime(evaluation.returns.index.max()).normalize()
        runs.append(
            {
                "config_name": Path(str(config_path)).stem,
                "config_path": str(Path(str(config_path))),
                "summary": evaluation.summary,
                "ranking_history": ranking_slice,
                "value_curve": evaluation.curve,
                "eval_start": actual_eval_start,
                "eval_end": actual_eval_end,
            }
        )

    base_history = runs[0]["ranking_history"]
    rows: list[pd.DataFrame] = []
    for run in runs:
        summary = run["summary"].copy()
        signal_coverage, avg_llm_abs_score = _signal_metrics(run["ranking_history"])
        ending_value = float(run["value_curve"]["strategy_value"].iloc[-1])
        starting_value = float(args.initial_capital)
        summary.insert(0, "config_name", str(run["config_name"]))
        summary.insert(1, "config_path", str(run["config_path"]))
        summary["eval_start"] = pd.Timestamp(run["eval_start"]).date().isoformat()
        summary["eval_end"] = pd.Timestamp(run["eval_end"]).date().isoformat()
        summary["starting_capital"] = starting_value
        summary["ending_capital"] = ending_value
        summary["profit_dollars"] = ending_value - starting_value
        summary["signal_coverage"] = signal_coverage
        summary["avg_llm_abs_score"] = avg_llm_abs_score
        summary["changed_rebalance_count"] = _changed_rebalance_count(base_history, run["ranking_history"])
        rows.append(summary)

    comparison = pd.concat(rows, ignore_index=True).sort_values(
        ["ending_capital", "sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)
    best_config_name = str(comparison.loc[0, "config_name"])
    best_run = next(run for run in runs if run["config_name"] == best_config_name)

    value_curves = pd.DataFrame({"date": best_run["value_curve"]["date"]})
    for run in runs:
        curve = pd.DataFrame(run["value_curve"]).rename(
            columns={"strategy_value": f"{run['config_name']}_value"}
        )
        value_curves = value_curves.merge(curve[["date", f"{run['config_name']}_value"]], on="date", how="left")
    if "benchmark_value" in best_run["value_curve"].columns:
        value_curves["benchmark_value"] = best_run["value_curve"]["benchmark_value"].to_numpy()
    output_dir = ensure_output_dir(args.output_dir)
    output_files = write_dataframe_artifacts(
        output_dir,
        [
            DataFrameArtifact("comparison", comparison, "comparison_last_year.csv"),
            DataFrameArtifact("portfolio_values", value_curves, "portfolio_values_last_year.csv"),
        ],
    )
    comparison_path = output_files["comparison"]
    values_path = output_files["portfolio_values"]
    chart_path = output_dir / "portfolio_values_last_year.svg"

    benchmark_name = "benchmark_value" if "benchmark_value" in value_curves.columns else value_curves.columns[-1]
    _build_svg(value_curves, benchmark_name=benchmark_name, output_path=chart_path)
    manifest = build_run_manifest(
        configs[0],
        experiment_name="performance_report",
        extra={
            "configs": [str(Path(config_path).resolve()) for config_path in args.configs],
            "lookback_days": args.lookback_days,
            "initial_capital": args.initial_capital,
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
            "comparison": comparison_path,
            "portfolio_values": values_path,
            "chart": chart_path,
        },
    )

    print(comparison.to_string(index=False))
    print(f"Latest market date: {latest_date.date().isoformat()}")
    print(f"Evaluation window: {comparison.loc[0, 'eval_start']} to {comparison.loc[0, 'eval_end']}")
    print(f"Best config over this window: {best_config_name}")
    print(f"Saved comparison to: {comparison_path}")
    print(f"Saved portfolio values to: {values_path}")
    print(f"Saved chart to: {chart_path}")


if __name__ == "__main__":
    main()
