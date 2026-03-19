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
from invest_ai_core.evaluation import evaluate_backtest_window
from invest_ai_core.reporting import (
    build_evaluation_row as _build_summary_row,
    build_value_curve as _build_value_curve,
)
from us_invest_ai.backtest import run_backtest
from us_invest_ai.config import load_config
from us_invest_ai.data import prepare_market_data_bundle
from us_invest_ai.dl_strategy import MLPModelConfig, generate_mlp_target_weights
from us_invest_ai.experiment_manifest import build_run_manifest
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
        description="Compare rules, ridge, tree, MLP, TCN, hybrid sequence, LSTM, and clipped-objective transformer walk-forward models on the last year."
    )
    parser.add_argument(
        "--config",
        default="us_stocks/config/soft_price_large_cap_60_dynamic_eligibility.yaml",
        help="Base config for market data, costs, and portfolio construction.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Calendar-day final test window ending on the latest trading day.",
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
        help="Trading-day embargo between the training labels and the live rebalance date.",
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
        default=40,
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
        help="Initial capital used for the value chart.",
    )
    parser.add_argument(
        "--use-llm-feature",
        action="store_true",
        help="Use llm_score as an auxiliary feature when the config enables LLM.",
    )
    parser.add_argument(
        "--output-dir",
        default="us_stocks/artifacts/deep_learning_last_year",
        help="Directory for summary, value curves, ranking histories, and chart.",
    )
    return parser.parse_args()


def _load_saved_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def _load_market_data(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = config.output.data_dir / "raw"
    prices_path = raw_dir / "prices.csv"
    benchmark_path = raw_dir / "benchmark.csv"
    if prices_path.exists() and benchmark_path.exists():
        return _load_saved_prices(prices_path), _load_saved_prices(benchmark_path)

    prices = download_ohlcv(
        tickers=config.data.tickers,
        start=config.data.start,
        end=config.data.end,
    )
    benchmark_prices = download_ohlcv(
        tickers=[config.data.benchmark],
        start=config.data.start,
        end=config.data.end,
    )
    return prices, benchmark_prices

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

    from us_invest_ai.features import build_features

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

    baseline_weights, baseline_history = generate_target_weights(features, config.strategy, llm_scores)
    baseline_result = run_backtest(
        prices=prices,
        target_weights=baseline_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
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
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    ridge_result = run_backtest(
        prices=prices,
        target_weights=ridge_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
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
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    tree_result = run_backtest(
        prices=prices,
        target_weights=tree_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
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
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    mlp_result = run_backtest(
        prices=prices,
        target_weights=mlp_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
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
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    tcn_result = run_backtest(
        prices=prices,
        target_weights=tcn_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
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
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    hybrid_result = run_backtest(
        prices=prices,
        target_weights=hybrid_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
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
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    lstm_result = run_backtest(
        prices=prices,
        target_weights=lstm_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
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
        eval_start=eval_start,
        llm_scores=llm_scores,
    )
    transformer_result = run_backtest(
        prices=prices,
        target_weights=transformer_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )

    baseline_returns = baseline_result.daily_returns.loc[baseline_result.daily_returns.index >= eval_start]
    ridge_returns = ridge_result.daily_returns.loc[ridge_result.daily_returns.index >= eval_start]
    tree_returns = tree_result.daily_returns.loc[tree_result.daily_returns.index >= eval_start]
    mlp_returns = mlp_result.daily_returns.loc[mlp_result.daily_returns.index >= eval_start]
    tcn_returns = tcn_result.daily_returns.loc[tcn_result.daily_returns.index >= eval_start]
    hybrid_returns = hybrid_result.daily_returns.loc[hybrid_result.daily_returns.index >= eval_start]
    lstm_returns = lstm_result.daily_returns.loc[lstm_result.daily_returns.index >= eval_start]
    transformer_returns = transformer_result.daily_returns.loc[transformer_result.daily_returns.index >= eval_start]
    eval_start_actual = pd.to_datetime(transformer_returns.index.min()).normalize()
    eval_end_actual = pd.to_datetime(transformer_returns.index.max()).normalize()

    benchmark_returns = None
    if baseline_result.benchmark_returns is not None:
        benchmark_returns = baseline_result.benchmark_returns.reindex(transformer_returns.index)

    baseline_evaluation = evaluate_backtest_window(baseline_result, eval_start_actual, initial_capital=args.initial_capital)
    ridge_evaluation = evaluate_backtest_window(ridge_result, eval_start_actual, initial_capital=args.initial_capital)
    tree_evaluation = evaluate_backtest_window(tree_result, eval_start_actual, initial_capital=args.initial_capital)
    mlp_evaluation = evaluate_backtest_window(mlp_result, eval_start_actual, initial_capital=args.initial_capital)
    tcn_evaluation = evaluate_backtest_window(tcn_result, eval_start_actual, initial_capital=args.initial_capital)
    hybrid_evaluation = evaluate_backtest_window(hybrid_result, eval_start_actual, initial_capital=args.initial_capital)
    lstm_evaluation = evaluate_backtest_window(lstm_result, eval_start_actual, initial_capital=args.initial_capital)
    transformer_evaluation = evaluate_backtest_window(
        transformer_result,
        eval_start_actual,
        initial_capital=args.initial_capital,
    )

    baseline_summary = baseline_evaluation.summary
    ridge_summary = ridge_evaluation.summary
    tree_summary = tree_evaluation.summary
    mlp_summary = mlp_evaluation.summary
    tcn_summary = tcn_evaluation.summary
    hybrid_summary = hybrid_evaluation.summary
    lstm_summary = lstm_evaluation.summary
    transformer_summary = transformer_evaluation.summary

    baseline_curve = baseline_evaluation.curve
    ridge_curve = ridge_evaluation.curve
    tree_curve = tree_evaluation.curve
    mlp_curve = mlp_evaluation.curve
    tcn_curve = tcn_evaluation.curve
    hybrid_curve = hybrid_evaluation.curve
    lstm_curve = lstm_evaluation.curve
    transformer_curve = transformer_evaluation.curve

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_frame = pd.concat(
        [
            _build_summary_row(
                "configured_baseline",
                baseline_summary,
                baseline_history.loc[baseline_history["date"] >= eval_start_actual],
                baseline_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
            _build_summary_row(
                "ridge_walkforward",
                ridge_summary,
                ridge_history.loc[ridge_history["date"] >= eval_start_actual],
                ridge_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
            _build_summary_row(
                "tree_walkforward",
                tree_summary,
                tree_history.loc[tree_history["date"] >= eval_start_actual],
                tree_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
            _build_summary_row(
                "mlp_walkforward",
                mlp_summary,
                mlp_history.loc[mlp_history["date"] >= eval_start_actual],
                mlp_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
            _build_summary_row(
                "tcn_walkforward",
                tcn_summary,
                tcn_history.loc[tcn_history["date"] >= eval_start_actual],
                tcn_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
            _build_summary_row(
                "hybrid_sequence_walkforward",
                hybrid_summary,
                hybrid_history.loc[hybrid_history["date"] >= eval_start_actual],
                hybrid_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
            _build_summary_row(
                "lstm_walkforward",
                lstm_summary,
                lstm_history.loc[lstm_history["date"] >= eval_start_actual],
                lstm_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
            _build_summary_row(
                "transformer_walkforward",
                transformer_summary,
                transformer_history.loc[transformer_history["date"] >= eval_start_actual],
                transformer_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
        ],
        ignore_index=True,
    ).sort_values(["ending_capital", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    values_frame = pd.DataFrame({"date": tcn_curve["date"]})
    values_frame["configured_baseline_value"] = baseline_curve["strategy_value"].to_numpy()
    values_frame["ridge_walkforward_value"] = ridge_curve["strategy_value"].to_numpy()
    values_frame["tree_walkforward_value"] = tree_curve["strategy_value"].to_numpy()
    values_frame["mlp_walkforward_value"] = mlp_curve["strategy_value"].to_numpy()
    values_frame["tcn_walkforward_value"] = tcn_curve["strategy_value"].to_numpy()
    values_frame["hybrid_sequence_walkforward_value"] = hybrid_curve["strategy_value"].to_numpy()
    values_frame["lstm_walkforward_value"] = lstm_curve["strategy_value"].to_numpy()
    values_frame["transformer_walkforward_value"] = transformer_curve["strategy_value"].to_numpy()
    if "benchmark_value" in tcn_curve.columns:
        values_frame["benchmark_value"] = tcn_curve["benchmark_value"].to_numpy()
    output_dir = ensure_output_dir(output_dir)
    output_files = write_dataframe_artifacts(
        output_dir,
        [
            DataFrameArtifact("summary", summary_frame, "deep_learning_summary_last_year.csv"),
            DataFrameArtifact("values", values_frame, "deep_learning_values_last_year.csv"),
            DataFrameArtifact(
                "baseline_history",
                baseline_history.loc[baseline_history["date"] >= eval_start_actual],
                "configured_baseline_history_last_year.csv",
            ),
            DataFrameArtifact(
                "ridge_history",
                ridge_history.loc[ridge_history["date"] >= eval_start_actual],
                "ridge_walkforward_history_last_year.csv",
            ),
            DataFrameArtifact(
                "tree_history",
                tree_history.loc[tree_history["date"] >= eval_start_actual],
                "tree_walkforward_history_last_year.csv",
            ),
            DataFrameArtifact(
                "mlp_history",
                mlp_history.loc[mlp_history["date"] >= eval_start_actual],
                "mlp_walkforward_history_last_year.csv",
            ),
            DataFrameArtifact(
                "tcn_history",
                tcn_history.loc[tcn_history["date"] >= eval_start_actual],
                "tcn_walkforward_history_last_year.csv",
            ),
            DataFrameArtifact(
                "hybrid_history",
                hybrid_history.loc[hybrid_history["date"] >= eval_start_actual],
                "hybrid_sequence_walkforward_history_last_year.csv",
            ),
            DataFrameArtifact(
                "lstm_history",
                lstm_history.loc[lstm_history["date"] >= eval_start_actual],
                "lstm_walkforward_history_last_year.csv",
            ),
            DataFrameArtifact(
                "transformer_history",
                transformer_history.loc[transformer_history["date"] >= eval_start_actual],
                "transformer_walkforward_history_last_year.csv",
            ),
        ],
    )
    summary_path = output_files["summary"]
    values_path = output_files["values"]

    chart_path = output_dir / "deep_learning_values_last_year.svg"
    benchmark_name = "benchmark_value" if "benchmark_value" in values_frame.columns else "configured_baseline_value"
    _build_svg(values_frame, benchmark_name=benchmark_name, output_path=chart_path)
    manifest = build_run_manifest(
        config,
        experiment_name="deep_learning_report",
        extra={
            "lookback_days": args.lookback_days,
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
            **output_files,
            "chart": chart_path,
        },
    )

    print(summary_frame.to_string(index=False))
    print(f"Latest market date: {latest_date.date().isoformat()}")
    print(f"Evaluation window: {eval_start_actual.date().isoformat()} to {eval_end_actual.date().isoformat()}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved values to: {values_path}")
    print(f"Saved chart to: {chart_path}")


if __name__ == "__main__":
    main()
