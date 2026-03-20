from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from invest_ai_core.artifacts import (
    DataFrameArtifact,
    ensure_output_dir,
    write_dataframe_artifacts,
    write_manifest_with_outputs,
)
from invest_ai_core.evaluation import build_backtest_evaluation_row
from us_invest_ai.backtest import run_backtest
from us_invest_ai.config import load_config
from us_invest_ai.data import prepare_market_data_bundle
from us_invest_ai.dl_strategy import MLPModelConfig, generate_mlp_target_weights
from us_invest_ai.experiment_manifest import build_run_manifest
from us_invest_ai.features import build_features
from us_invest_ai.hybrid_sequence_strategy import (
    HybridSequenceModelConfig,
    generate_hybrid_sequence_target_weights,
)
from us_invest_ai.lstm_strategy import LSTMModelConfig, generate_lstm_target_weights
from us_invest_ai.ml_strategy import MLModelConfig, generate_ridge_walkforward_target_weights
from us_invest_ai.performance_report import _build_svg
from us_invest_ai.signals import load_llm_scores
from us_invest_ai.strategy import generate_target_weights
from us_invest_ai.tcn_strategy import TCNModelConfig, generate_tcn_target_weights
from us_invest_ai.transformer_report_support import build_transformer_report_config
from us_invest_ai.transformer_strategy import generate_transformer_target_weights
from us_invest_ai.tree_strategy import TreeModelConfig, generate_tree_target_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model stability across repeated out-of-sample windows, including the clipped-objective transformer."
    )
    parser.add_argument(
        "--config",
        default="us_stocks/config/soft_price_large_cap_60_dynamic_eligibility.yaml",
        help="Base config for market data, costs, and portfolio construction.",
    )
    parser.add_argument(
        "--window-trading-days",
        type=int,
        default=252,
        help="Trading-day length of each evaluation window.",
    )
    parser.add_argument(
        "--step-trading-days",
        type=int,
        default=252,
        help="Trading-day step between window end dates.",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=4,
        help="Maximum number of windows to evaluate.",
    )
    parser.add_argument(
        "--label-horizon-days",
        type=int,
        default=20,
        help="Forward return horizon used as the supervised label.",
    )
    parser.add_argument(
        "--validation-window-days",
        type=int,
        default=60,
        help="Number of pre-live trading dates reserved for walk-forward validation.",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=20,
        help="Trading-day embargo between training labels and the live rebalance date.",
    )
    parser.add_argument(
        "--min-training-samples",
        type=int,
        default=252,
        help="Minimum training samples for supervised models.",
    )
    parser.add_argument(
        "--min-validation-samples",
        type=int,
        default=120,
        help="Minimum validation samples for supervised models.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=8.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--tree-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for the gradient-boosted tree model.",
    )
    parser.add_argument(
        "--tree-max-iter",
        type=int,
        default=200,
        help="Maximum boosting iterations for the tree model.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=3,
        help="Maximum tree depth for each boosting stage.",
    )
    parser.add_argument(
        "--tree-min-samples-leaf",
        type=int,
        default=20,
        help="Minimum samples per leaf in the tree model.",
    )
    parser.add_argument(
        "--tree-l2-regularization",
        type=float,
        default=0.0,
        help="L2 regularization strength for the tree model.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden layer width for the MLP.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="MLP learning rate.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=250,
        help="Maximum number of MLP training epochs per rebalance.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for MLP training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay used in the MLP optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early-stopping patience for the MLP.",
    )
    parser.add_argument(
        "--sequence-lookback-window",
        type=int,
        default=20,
        help="Lookback window length for the temporal convolution baseline.",
    )
    parser.add_argument(
        "--sequence-kernel-size",
        type=int,
        default=5,
        help="Kernel size for the temporal convolution baseline.",
    )
    parser.add_argument(
        "--sequence-hidden-channels",
        type=int,
        default=8,
        help="Hidden channel count for the temporal convolution baseline.",
    )
    parser.add_argument(
        "--sequence-learning-rate",
        type=float,
        default=0.005,
        help="Learning rate for the temporal convolution baseline.",
    )
    parser.add_argument(
        "--sequence-max-epochs",
        type=int,
        default=120,
        help="Maximum number of training epochs for the temporal convolution baseline.",
    )
    parser.add_argument(
        "--sequence-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for the temporal convolution baseline.",
    )
    parser.add_argument(
        "--sequence-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay used in the temporal convolution baseline.",
    )
    parser.add_argument(
        "--sequence-patience",
        type=int,
        default=15,
        help="Early-stopping patience for the temporal convolution baseline.",
    )
    parser.add_argument(
        "--hybrid-static-hidden-dim",
        type=int,
        default=16,
        help="Hidden dimension for the static branch inside the hybrid sequence baseline.",
    )
    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=12,
        help="Hidden dimension for the LSTM baseline.",
    )
    parser.add_argument(
        "--transformer-model-dim",
        type=int,
        default=4,
        help="Model dimension for the clipped-objective transformer baseline.",
    )
    parser.add_argument(
        "--transformer-training-lookback-days",
        type=int,
        default=252,
        help="Rolling training-window length in trading days for the clipped-objective transformer baseline.",
    )
    parser.add_argument(
        "--transformer-sequence-lookback-window",
        type=int,
        default=20,
        help="Sequence lookback window for the clipped-objective transformer baseline.",
    )
    parser.add_argument(
        "--transformer-target-clip-quantile",
        type=float,
        default=0.95,
        help="Central target clip quantile for the clipped-objective transformer baseline.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital assumed inside each window.",
    )
    parser.add_argument(
        "--use-llm-feature",
        action="store_true",
        help="Use llm_score as an auxiliary feature when the config enables LLM.",
    )
    parser.add_argument(
        "--output-dir",
        default="us_stocks/artifacts/stability_report",
        help="Directory for per-window summaries and aggregate outputs.",
    )
    return parser.parse_args()


def _build_evaluation_windows(
    trading_dates: list[pd.Timestamp],
    window_trading_days: int,
    step_trading_days: int,
    window_count: int,
) -> list[dict[str, pd.Timestamp | str]]:
    if window_trading_days <= 0 or step_trading_days <= 0 or window_count <= 0:
        raise ValueError("Window length, step, and count must all be positive.")
    if len(trading_dates) < window_trading_days:
        raise ValueError("Not enough trading dates to build the requested window length.")

    windows: list[dict[str, pd.Timestamp | str]] = []
    last_index = len(trading_dates) - 1
    for offset in range(window_count):
        end_index = last_index - offset * step_trading_days
        start_index = end_index - window_trading_days + 1
        if start_index < 0:
            break

        eval_start = pd.Timestamp(trading_dates[start_index]).normalize()
        eval_end = pd.Timestamp(trading_dates[end_index]).normalize()
        windows.append(
            {
                "window_label": f"{eval_start.date().isoformat()}_to_{eval_end.date().isoformat()}",
                "eval_start": eval_start,
                "eval_end": eval_end,
            }
        )

    windows.reverse()
    if not windows:
        raise ValueError("Unable to build any evaluation windows from the available trading dates.")
    return windows

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
    trading_dates = sorted(pd.to_datetime(prices["date"].unique()).tolist())
    windows = _build_evaluation_windows(
        trading_dates=trading_dates,
        window_trading_days=args.window_trading_days,
        step_trading_days=args.step_trading_days,
        window_count=args.window_count,
    )
    earliest_eval_start = pd.Timestamp(windows[0]["eval_start"]).normalize()

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
    llm_scores = None
    if config.llm.enabled:
        llm_scores = load_llm_scores(config.llm.signal_path, config.llm.horizon_bucket)

    baseline_weights, baseline_history = generate_target_weights(features, config.strategy, llm_scores)
    baseline_result = run_backtest(
        prices=prices,
        target_weights=baseline_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )
    ridge_weights, ridge_history = generate_ridge_walkforward_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=MLModelConfig(
            label_horizon_days=args.label_horizon_days,
            ridge_alpha=args.ridge_alpha,
            min_training_samples=args.min_training_samples,
            validation_window_days=args.validation_window_days,
            embargo_days=args.embargo_days,
            min_validation_samples=args.min_validation_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
        ),
        eval_start=earliest_eval_start,
        llm_scores=llm_scores,
    )
    ridge_result = run_backtest(
        prices=prices,
        target_weights=ridge_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )
    tree_weights, tree_history = generate_tree_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=TreeModelConfig(
            label_horizon_days=args.label_horizon_days,
            validation_window_days=args.validation_window_days,
            embargo_days=args.embargo_days,
            min_training_samples=args.min_training_samples,
            min_validation_samples=args.min_validation_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
            learning_rate=args.tree_learning_rate,
            max_iter=args.tree_max_iter,
            max_depth=args.tree_max_depth,
            min_samples_leaf=args.tree_min_samples_leaf,
            l2_regularization=args.tree_l2_regularization,
        ),
        eval_start=earliest_eval_start,
        llm_scores=llm_scores,
    )
    tree_result = run_backtest(
        prices=prices,
        target_weights=tree_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )
    mlp_weights, mlp_history = generate_mlp_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=MLPModelConfig(
            label_horizon_days=args.label_horizon_days,
            validation_window_days=args.validation_window_days,
            embargo_days=args.embargo_days,
            min_training_samples=args.min_training_samples,
            min_validation_samples=args.min_validation_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            patience=args.patience,
        ),
        eval_start=earliest_eval_start,
        llm_scores=llm_scores,
    )
    mlp_result = run_backtest(
        prices=prices,
        target_weights=mlp_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )
    tcn_weights, tcn_history = generate_tcn_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=TCNModelConfig(
            label_horizon_days=args.label_horizon_days,
            validation_window_days=args.validation_window_days,
            embargo_days=args.embargo_days,
            min_training_samples=args.min_training_samples,
            min_validation_samples=args.min_validation_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
            lookback_window=args.sequence_lookback_window,
            kernel_size=args.sequence_kernel_size,
            hidden_channels=args.sequence_hidden_channels,
            learning_rate=args.sequence_learning_rate,
            max_epochs=args.sequence_max_epochs,
            batch_size=args.sequence_batch_size,
            weight_decay=args.sequence_weight_decay,
            patience=args.sequence_patience,
        ),
        eval_start=earliest_eval_start,
        llm_scores=llm_scores,
    )
    tcn_result = run_backtest(
        prices=prices,
        target_weights=tcn_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )
    hybrid_weights, hybrid_history = generate_hybrid_sequence_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=HybridSequenceModelConfig(
            label_horizon_days=args.label_horizon_days,
            validation_window_days=args.validation_window_days,
            embargo_days=args.embargo_days,
            min_training_samples=args.min_training_samples,
            min_validation_samples=args.min_validation_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
            lookback_window=args.sequence_lookback_window,
            kernel_size=args.sequence_kernel_size,
            sequence_hidden_channels=args.sequence_hidden_channels,
            static_hidden_dim=args.hybrid_static_hidden_dim,
            learning_rate=args.sequence_learning_rate,
            max_epochs=args.sequence_max_epochs,
            batch_size=args.sequence_batch_size,
            weight_decay=args.sequence_weight_decay,
            patience=args.sequence_patience,
        ),
        eval_start=earliest_eval_start,
        llm_scores=llm_scores,
    )
    hybrid_result = run_backtest(
        prices=prices,
        target_weights=hybrid_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )
    lstm_weights, lstm_history = generate_lstm_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=LSTMModelConfig(
            label_horizon_days=args.label_horizon_days,
            validation_window_days=args.validation_window_days,
            embargo_days=args.embargo_days,
            min_training_samples=args.min_training_samples,
            min_validation_samples=args.min_validation_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
            lookback_window=args.sequence_lookback_window,
            hidden_dim=args.lstm_hidden_dim,
            learning_rate=args.sequence_learning_rate,
            max_epochs=args.sequence_max_epochs,
            batch_size=args.sequence_batch_size,
            weight_decay=args.sequence_weight_decay,
            patience=args.sequence_patience,
        ),
        eval_start=earliest_eval_start,
        llm_scores=llm_scores,
    )
    lstm_result = run_backtest(
        prices=prices,
        target_weights=lstm_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )
    transformer_weights, transformer_history = generate_transformer_target_weights(
        features=features,
        strategy_config=config.strategy,
        model_config=build_transformer_report_config(
            label_horizon_days=args.label_horizon_days,
            validation_window_days=args.validation_window_days,
            embargo_days=args.embargo_days,
            min_training_samples=args.min_training_samples,
            min_validation_samples=args.min_validation_samples,
            use_llm_feature=args.use_llm_feature and config.llm.enabled,
            sequence_lookback_window=args.transformer_sequence_lookback_window,
            training_lookback_days=args.transformer_training_lookback_days,
            model_dim=args.transformer_model_dim,
            target_clip_quantile=args.transformer_target_clip_quantile,
        ),
        eval_start=earliest_eval_start,
        llm_scores=llm_scores,
    )
    transformer_result = run_backtest(
        prices=prices,
        target_weights=transformer_weights,
        backtest_config=config.backtest,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )

    model_runs = [
        ("configured_baseline", baseline_result, baseline_history),
        ("ridge_walkforward", ridge_result, ridge_history),
        ("tree_walkforward", tree_result, tree_history),
        ("mlp_walkforward", mlp_result, mlp_history),
        ("tcn_walkforward", tcn_result, tcn_history),
        ("hybrid_sequence_walkforward", hybrid_result, hybrid_history),
        ("lstm_walkforward", lstm_result, lstm_history),
        ("transformer_walkforward", transformer_result, transformer_history),
    ]

    window_rows: list[pd.DataFrame] = []
    for window in windows:
        for model_name, result, history in model_runs:
            eval_start = pd.Timestamp(window["eval_start"]).normalize()
            eval_end = pd.Timestamp(window["eval_end"]).normalize()
            window_rows.append(
                build_backtest_evaluation_row(
                    model_name=model_name,
                    result=result,
                    window_label=str(window["window_label"]),
                    eval_start=eval_start,
                    eval_end=eval_end,
                    history=history,
                    initial_capital=args.initial_capital,
                    include_rebalance_count=True,
                )
            )

    window_summary = pd.concat(window_rows, ignore_index=True)
    baseline_capital = (
        window_summary.loc[window_summary["model_name"] == "configured_baseline", ["window_label", "ending_capital"]]
        .rename(columns={"ending_capital": "baseline_ending_capital"})
        .reset_index(drop=True)
    )
    window_summary = window_summary.merge(baseline_capital, on="window_label", how="left")
    window_summary["ending_capital_gap_vs_baseline"] = (
        window_summary["ending_capital"] - window_summary["baseline_ending_capital"]
    )
    window_summary["beat_baseline"] = window_summary["ending_capital_gap_vs_baseline"] > 0
    window_summary = window_summary.sort_values(["eval_start", "ending_capital"], ascending=[True, False]).reset_index(drop=True)

    aggregate = (
        window_summary.groupby("model_name", as_index=False)
        .agg(
            window_count=("window_label", "nunique"),
            avg_ending_capital=("ending_capital", "mean"),
            median_ending_capital=("ending_capital", "median"),
            avg_profit_dollars=("profit_dollars", "mean"),
            avg_cagr=("cagr", "mean"),
            median_cagr=("cagr", "median"),
            avg_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            avg_sortino=("sortino", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_information_ratio=("information_ratio", "mean"),
            beat_baseline_windows=("beat_baseline", "sum"),
        )
        .sort_values(["avg_ending_capital", "avg_sharpe"], ascending=[False, False])
        .reset_index(drop=True)
    )

    chart_frame = window_summary.pivot_table(
        index="eval_end",
        columns="model_name",
        values="ending_capital",
        aggfunc="last",
    ).reset_index().rename(columns={"eval_end": "date"})
    ordered_chart_columns = [
        "date",
        "configured_baseline",
        "ridge_walkforward",
        "tree_walkforward",
        "mlp_walkforward",
        "tcn_walkforward",
        "hybrid_sequence_walkforward",
        "lstm_walkforward",
        "transformer_walkforward",
    ]
    chart_frame = chart_frame.reindex(columns=ordered_chart_columns)
    chart_frame = chart_frame.rename(columns={column: f"{column}_value" for column in chart_frame.columns if column != "date"})

    output_dir = ensure_output_dir(args.output_dir)
    output_files = write_dataframe_artifacts(
        output_dir,
        [
            DataFrameArtifact("window_summary", window_summary, "stability_window_summary.csv"),
            DataFrameArtifact("aggregate", aggregate, "stability_model_aggregate.csv"),
            DataFrameArtifact("window_end_values", chart_frame, "stability_window_end_values.csv"),
        ],
    )
    summary_path = output_files["window_summary"]
    aggregate_path = output_files["aggregate"]
    values_path = output_files["window_end_values"]
    chart_path = output_dir / "stability_window_end_values.svg"
    _build_svg(chart_frame, benchmark_name="configured_baseline_value", output_path=chart_path)

    manifest = build_run_manifest(
        config,
        experiment_name="stability_report",
        extra={
            "window_trading_days": args.window_trading_days,
            "step_trading_days": args.step_trading_days,
            "window_count": args.window_count,
            "label_horizon_days": args.label_horizon_days,
            "validation_window_days": args.validation_window_days,
            "embargo_days": args.embargo_days,
            "min_training_samples": args.min_training_samples,
            "min_validation_samples": args.min_validation_samples,
            "ridge_alpha": args.ridge_alpha,
            "tree_learning_rate": args.tree_learning_rate,
            "tree_max_iter": args.tree_max_iter,
            "tree_max_depth": args.tree_max_depth,
            "tree_min_samples_leaf": args.tree_min_samples_leaf,
            "tree_l2_regularization": args.tree_l2_regularization,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "sequence_lookback_window": args.sequence_lookback_window,
            "sequence_kernel_size": args.sequence_kernel_size,
            "sequence_hidden_channels": args.sequence_hidden_channels,
            "sequence_learning_rate": args.sequence_learning_rate,
            "sequence_max_epochs": args.sequence_max_epochs,
            "sequence_batch_size": args.sequence_batch_size,
            "sequence_weight_decay": args.sequence_weight_decay,
            "sequence_patience": args.sequence_patience,
            "hybrid_static_hidden_dim": args.hybrid_static_hidden_dim,
            "lstm_hidden_dim": args.lstm_hidden_dim,
            "transformer_model_dim": args.transformer_model_dim,
            "transformer_training_lookback_days": args.transformer_training_lookback_days,
            "transformer_sequence_lookback_window": args.transformer_sequence_lookback_window,
            "transformer_target_clip_quantile": args.transformer_target_clip_quantile,
            "use_llm_feature": args.use_llm_feature and config.llm.enabled,
            "window_labels": [str(window["window_label"]) for window in windows],
            "market_data_source": market_data.provenance.get("source"),
            "market_data_manifest_path": market_data.provenance.get("manifest_path"),
            "market_data_manifest_sha256": market_data.provenance.get("manifest_sha256"),
        },
    )
    write_manifest_with_outputs(
        output_dir,
        "stability_manifest.json",
        manifest,
        {
            "window_summary": summary_path,
            "aggregate": aggregate_path,
            "window_end_values": values_path,
            "chart": chart_path,
        },
    )

    print(window_summary.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(f"Saved window summary to: {summary_path}")
    print(f"Saved aggregate summary to: {aggregate_path}")
    print(f"Saved chart to: {chart_path}")


if __name__ == "__main__":
    main()
