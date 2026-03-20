from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from invest_ai_core.evaluation import evaluate_backtest_window
from invest_ai_core.reporting import compute_signal_metrics
from us_invest_ai.backtest import run_backtest
from us_invest_ai.config import DataConfig, EligibilityConfig, load_config
from us_invest_ai.data import prepare_market_data_bundle
from us_invest_ai.experiment_manifest import attach_output_files, build_run_manifest, save_manifest
from us_invest_ai.features import build_features
from us_invest_ai.signals import load_llm_scores
from us_invest_ai.strategy import generate_target_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple US stocks strategy configs.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "us_stocks/config/base.yaml",
            "us_stocks/config/soft_price.yaml",
            "us_stocks/config/with_llm_short.yaml",
            "us_stocks/config/with_llm_swing.yaml",
            "us_stocks/config/with_llm_long.yaml",
        ],
        help="One or more config files to compare.",
    )
    parser.add_argument(
        "--eval-start",
        default=None,
        help="Optional evaluation start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-csv",
        default="us_stocks/artifacts/comparison.csv",
        help="Path to save the comparison table.",
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


def _slice_result(result, eval_start: str | None) -> pd.DataFrame:
    if not eval_start:
        return result.summary.copy()

    return evaluate_backtest_window(
        result,
        pd.Timestamp(eval_start),
        initial_capital=100_000.0,
    ).summary


def _slice_ranking_history(ranking_history: pd.DataFrame, eval_start: str | None) -> pd.DataFrame:
    history = ranking_history.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    if eval_start:
        history = history.loc[history["date"] >= pd.Timestamp(eval_start)]
    return history


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

    changed = 0
    for date in common_index:
        if not np.allclose(base_aligned.loc[date].to_numpy(), candidate_aligned.loc[date].to_numpy()):
            changed += 1
    return changed


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
        runs.append(
            {
                "config_path": config_path,
                "result": result,
                "ranking_history": _slice_ranking_history(ranking_history, args.eval_start),
            }
        )

    base_history = runs[0]["ranking_history"]
    rows: list[pd.DataFrame] = []
    for run in runs:
        summary = _slice_result(run["result"], args.eval_start)
        signal_coverage, avg_llm_abs_score = _signal_metrics(run["ranking_history"])
        summary["signal_coverage"] = signal_coverage
        summary["avg_llm_abs_score"] = avg_llm_abs_score
        summary["changed_rebalance_count"] = _changed_rebalance_count(
            base_history,
            run["ranking_history"],
        )
        summary.insert(0, "config_name", Path(str(run["config_path"])).stem)
        summary.insert(1, "config_path", str(Path(str(run["config_path"]))))
        rows.append(summary)

    comparison = pd.concat(rows, ignore_index=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    manifest = build_run_manifest(
        configs[0],
        experiment_name="compare_configs",
        extra={
            "configs": [str(Path(config_path).resolve()) for config_path in args.configs],
            "eval_start": args.eval_start,
            "market_data_source": market_data.provenance.get("source"),
            "market_data_manifest_path": market_data.provenance.get("manifest_path"),
            "market_data_manifest_sha256": market_data.provenance.get("manifest_sha256"),
        },
    )
    save_manifest(output_path.parent / "comparison_manifest.json", attach_output_files(manifest, {"comparison": output_path}))

    print(comparison.to_string(index=False))
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()
