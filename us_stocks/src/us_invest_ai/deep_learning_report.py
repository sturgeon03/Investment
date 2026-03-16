from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from us_invest_ai.backtest import build_summary, run_backtest
from us_invest_ai.config import load_config
from us_invest_ai.data import download_ohlcv
from us_invest_ai.dl_strategy import MLPModelConfig, generate_mlp_target_weights
from us_invest_ai.ml_strategy import MLModelConfig, generate_ridge_walkforward_target_weights
from us_invest_ai.performance_report import _build_svg
from us_invest_ai.signals import load_llm_scores
from us_invest_ai.strategy import generate_target_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rules, ridge walk-forward, and MLP walk-forward on the last year."
    )
    parser.add_argument(
        "--config",
        default="us_stocks/config/soft_price.yaml",
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


def _build_value_curve(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    initial_capital: float,
) -> pd.DataFrame:
    strategy_growth = (1.0 + daily_returns).cumprod()
    strategy_value = initial_capital * (strategy_growth / strategy_growth.iloc[0])
    frame = pd.DataFrame({"date": strategy_value.index, "strategy_value": strategy_value.to_numpy()})
    if benchmark_returns is not None:
        benchmark_growth = (1.0 + benchmark_returns).cumprod()
        benchmark_value = initial_capital * (benchmark_growth / benchmark_growth.iloc[0])
        frame["benchmark_value"] = benchmark_value.reindex(strategy_value.index).to_numpy()
    return frame


def _build_summary_row(
    model_name: str,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    curve: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    initial_capital: float,
) -> pd.DataFrame:
    row = summary.copy()
    row.insert(0, "model_name", model_name)
    row["eval_start"] = eval_start.date().isoformat()
    row["eval_end"] = eval_end.date().isoformat()
    row["starting_capital"] = initial_capital
    row["ending_capital"] = float(curve["strategy_value"].iloc[-1])
    row["profit_dollars"] = float(curve["strategy_value"].iloc[-1] - initial_capital)
    selected = history.loc[history["selected"]] if "selected" in history.columns else pd.DataFrame()
    row["avg_train_sample_count"] = (
        float(selected["train_sample_count"].mean())
        if not selected.empty and "train_sample_count" in selected.columns
        else np.nan
    )
    row["avg_validation_sample_count"] = (
        float(selected["validation_sample_count"].mean())
        if not selected.empty and "validation_sample_count" in selected.columns
        else np.nan
    )
    return row


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    prices, benchmark_prices = _load_market_data(config)

    from us_invest_ai.features import build_features

    features = build_features(prices)
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
    )

    baseline_returns = baseline_result.daily_returns.loc[baseline_result.daily_returns.index >= eval_start]
    ridge_returns = ridge_result.daily_returns.loc[ridge_result.daily_returns.index >= eval_start]
    mlp_returns = mlp_result.daily_returns.loc[mlp_result.daily_returns.index >= eval_start]
    eval_start_actual = pd.to_datetime(mlp_returns.index.min()).normalize()
    eval_end_actual = pd.to_datetime(mlp_returns.index.max()).normalize()

    benchmark_returns = None
    if baseline_result.benchmark_returns is not None:
        benchmark_returns = baseline_result.benchmark_returns.reindex(mlp_returns.index)

    baseline_summary = build_summary(
        baseline_returns,
        baseline_result.turnover.reindex(baseline_returns.index),
        baseline_result.benchmark_returns.reindex(baseline_returns.index)
        if baseline_result.benchmark_returns is not None
        else None,
    )
    ridge_summary = build_summary(
        ridge_returns,
        ridge_result.turnover.reindex(ridge_returns.index),
        ridge_result.benchmark_returns.reindex(ridge_returns.index) if ridge_result.benchmark_returns is not None else None,
    )
    mlp_summary = build_summary(
        mlp_returns,
        mlp_result.turnover.reindex(mlp_returns.index),
        mlp_result.benchmark_returns.reindex(mlp_returns.index) if mlp_result.benchmark_returns is not None else None,
    )

    baseline_curve = _build_value_curve(baseline_returns, benchmark_returns, args.initial_capital)
    ridge_curve = _build_value_curve(ridge_returns, benchmark_returns, args.initial_capital)
    mlp_curve = _build_value_curve(mlp_returns, benchmark_returns, args.initial_capital)

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
                "mlp_walkforward",
                mlp_summary,
                mlp_history.loc[mlp_history["date"] >= eval_start_actual],
                mlp_curve,
                eval_start_actual,
                eval_end_actual,
                args.initial_capital,
            ),
        ],
        ignore_index=True,
    ).sort_values(["ending_capital", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    summary_path = output_dir / "deep_learning_summary_last_year.csv"
    summary_frame.to_csv(summary_path, index=False)

    values_frame = pd.DataFrame({"date": mlp_curve["date"]})
    values_frame["configured_baseline_value"] = baseline_curve["strategy_value"].to_numpy()
    values_frame["ridge_walkforward_value"] = ridge_curve["strategy_value"].to_numpy()
    values_frame["mlp_walkforward_value"] = mlp_curve["strategy_value"].to_numpy()
    if "benchmark_value" in mlp_curve.columns:
        values_frame["benchmark_value"] = mlp_curve["benchmark_value"].to_numpy()
    values_path = output_dir / "deep_learning_values_last_year.csv"
    values_frame.to_csv(values_path, index=False)

    chart_path = output_dir / "deep_learning_values_last_year.svg"
    benchmark_name = "benchmark_value" if "benchmark_value" in values_frame.columns else "configured_baseline_value"
    _build_svg(values_frame, benchmark_name=benchmark_name, output_path=chart_path)

    baseline_history.loc[baseline_history["date"] >= eval_start_actual].to_csv(
        output_dir / "configured_baseline_history_last_year.csv",
        index=False,
    )
    ridge_history.loc[ridge_history["date"] >= eval_start_actual].to_csv(
        output_dir / "ridge_walkforward_history_last_year.csv",
        index=False,
    )
    mlp_history.loc[mlp_history["date"] >= eval_start_actual].to_csv(
        output_dir / "mlp_walkforward_history_last_year.csv",
        index=False,
    )

    print(summary_frame.to_string(index=False))
    print(f"Latest market date: {latest_date.date().isoformat()}")
    print(f"Evaluation window: {eval_start_actual.date().isoformat()} to {eval_end_actual.date().isoformat()}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved values to: {values_path}")
    print(f"Saved chart to: {chart_path}")


if __name__ == "__main__":
    main()
