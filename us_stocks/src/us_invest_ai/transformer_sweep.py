from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from us_invest_ai.backtest import build_summary, run_backtest
from us_invest_ai.config import load_config
from us_invest_ai.data import prepare_market_data_bundle
from us_invest_ai.experiment_manifest import attach_output_files, build_run_manifest, save_manifest
from us_invest_ai.features import build_features
from us_invest_ai.performance_report import _build_svg
from us_invest_ai.stability_report import _build_evaluation_windows
from us_invest_ai.strategy import generate_target_weights
from us_invest_ai.transformer_strategy import TransformerModelConfig, generate_transformer_target_weights


def parse_int_grid(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",")]
    parsed = [int(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer value is required.")
    return parsed


def parse_optional_float_grid(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "raw"}:
            parsed.append(None)
            continue
        parsed.append(float(token))
    if not parsed:
        raise argparse.ArgumentTypeError("At least one float or 'none' value is required.")
    return parsed


def format_objective_name(target_clip_quantile: float | None) -> str:
    if target_clip_quantile is None:
        return "raw"
    percentage = f"{target_clip_quantile * 100:.1f}".rstrip("0").rstrip(".")
    return f"clip_q{percentage.replace('.', 'p')}"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a focused baseline-vs-transformer robustness sweep on the strict dynamic-universe lane."
    )
    parser.add_argument(
        "--config",
        default="us_stocks/config/soft_price_large_cap_60_dynamic_eligibility.yaml",
        help="Base config for the dynamic-universe control lane.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Calendar-day latest evaluation window ending on the latest market date.",
    )
    parser.add_argument(
        "--window-trading-days",
        type=int,
        default=252,
        help="Trading-day length for repeated OOS windows.",
    )
    parser.add_argument(
        "--step-trading-days",
        type=int,
        default=252,
        help="Trading-day step between repeated OOS windows.",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=4,
        help="Number of repeated OOS windows.",
    )
    parser.add_argument(
        "--transformer-model-dims",
        type=parse_int_grid,
        default=[4, 8],
        help="Comma-separated transformer model dimensions.",
    )
    parser.add_argument(
        "--transformer-training-lookback-days",
        type=parse_int_grid,
        default=[126, 252, 378],
        help="Comma-separated rolling training-window lengths in trading days.",
    )
    parser.add_argument(
        "--sequence-lookback-window",
        type=int,
        default=None,
        help="Single sequence lookback override.",
    )
    parser.add_argument(
        "--sequence-lookback-windows",
        type=parse_int_grid,
        default=[20],
        help="Comma-separated sequence lookback windows.",
    )
    parser.add_argument(
        "--target-clip-quantiles",
        type=parse_optional_float_grid,
        default=[None],
        help="Comma-separated central target clip quantiles, or 'none' for raw targets.",
    )
    parser.add_argument(
        "--label-horizon-days",
        type=int,
        default=20,
        help="Forward return label horizon.",
    )
    parser.add_argument(
        "--validation-window-days",
        type=int,
        default=60,
        help="Validation window length in trading days.",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=20,
        help="Embargo gap in trading days.",
    )
    parser.add_argument(
        "--min-training-samples",
        type=int,
        default=252,
        help="Minimum training samples.",
    )
    parser.add_argument(
        "--min-validation-samples",
        type=int,
        default=120,
        help="Minimum validation samples.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital used in value curves.",
    )
    parser.add_argument(
        "--output-dir",
        default="us_stocks/artifacts/transformer_sweep",
        help="Directory for sweep outputs.",
    )
    return parser.parse_args()


def _last_year_row(
    model_name: str,
    result,
    history: pd.DataFrame,
    eval_start: pd.Timestamp,
    initial_capital: float,
    extra: dict[str, object] | None = None,
) -> pd.DataFrame:
    returns = result.daily_returns.loc[result.daily_returns.index >= eval_start]
    turnover = result.turnover.reindex(returns.index)
    benchmark_returns = (
        result.benchmark_returns.reindex(returns.index)
        if result.benchmark_returns is not None
        else None
    )
    summary = build_summary(returns, turnover, benchmark_returns)
    curve = _build_value_curve(returns, benchmark_returns, initial_capital)
    row = summary.copy()
    row.insert(0, "model_name", model_name)
    row["eval_start"] = pd.Timestamp(returns.index.min()).date().isoformat()
    row["eval_end"] = pd.Timestamp(returns.index.max()).date().isoformat()
    row["starting_capital"] = initial_capital
    row["ending_capital"] = float(curve["strategy_value"].iloc[-1])
    row["profit_dollars"] = float(curve["strategy_value"].iloc[-1] - initial_capital)
    history_slice = history.loc[pd.to_datetime(history["date"]).dt.normalize() >= eval_start].copy()
    selected = history_slice.loc[history_slice["selected"]] if "selected" in history_slice.columns else pd.DataFrame()
    row["avg_train_sample_count"] = (
        float(selected["train_sample_count"].mean())
        if not selected.empty and "train_sample_count" in selected.columns
        else float("nan")
    )
    row["avg_validation_sample_count"] = (
        float(selected["validation_sample_count"].mean())
        if not selected.empty and "validation_sample_count" in selected.columns
        else float("nan")
    )
    row["avg_validation_mse"] = (
        float(selected["validation_mse"].mean())
        if not selected.empty and "validation_mse" in selected.columns
        else float("nan")
    )
    if extra:
        for key, value in extra.items():
            row[key] = value
    return row


def _window_rows(
    model_name: str,
    result,
    history: pd.DataFrame,
    windows: list[dict[str, pd.Timestamp | str]],
    initial_capital: float,
    extra: dict[str, object] | None = None,
) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for window in windows:
        eval_start = pd.Timestamp(window["eval_start"]).normalize()
        eval_end = pd.Timestamp(window["eval_end"]).normalize()
        returns = result.daily_returns.loc[(result.daily_returns.index >= eval_start) & (result.daily_returns.index <= eval_end)]
        turnover = result.turnover.reindex(returns.index)
        benchmark_returns = (
            result.benchmark_returns.reindex(returns.index)
            if result.benchmark_returns is not None
            else None
        )
        summary = build_summary(returns, turnover, benchmark_returns)
        curve = _build_value_curve(returns, benchmark_returns, initial_capital)
        row = summary.copy()
        row.insert(0, "model_name", model_name)
        row.insert(1, "window_label", str(window["window_label"]))
        row["eval_start"] = eval_start.date().isoformat()
        row["eval_end"] = eval_end.date().isoformat()
        row["starting_capital"] = initial_capital
        row["ending_capital"] = float(curve["strategy_value"].iloc[-1])
        row["profit_dollars"] = float(curve["strategy_value"].iloc[-1] - initial_capital)
        history_slice = history.loc[
            (pd.to_datetime(history["date"]).dt.normalize() >= eval_start)
            & (pd.to_datetime(history["date"]).dt.normalize() <= eval_end)
        ].copy()
        selected = history_slice.loc[history_slice["selected"]] if "selected" in history_slice.columns else pd.DataFrame()
        row["rebalance_count"] = int(history_slice["date"].nunique()) if not history_slice.empty else 0
        row["avg_train_sample_count"] = (
            float(selected["train_sample_count"].mean())
            if not selected.empty and "train_sample_count" in selected.columns
            else float("nan")
        )
        row["avg_validation_sample_count"] = (
            float(selected["validation_sample_count"].mean())
            if not selected.empty and "validation_sample_count" in selected.columns
            else float("nan")
        )
        row["avg_validation_mse"] = (
            float(selected["validation_mse"].mean())
            if not selected.empty and "validation_mse" in selected.columns
            else float("nan")
        )
        if extra:
            for key, value in extra.items():
                row[key] = value
        rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    sequence_lookback_windows = (
        [args.sequence_lookback_window]
        if args.sequence_lookback_window is not None
        else args.sequence_lookback_windows
    )

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
    latest_eval_start = latest_date - pd.Timedelta(days=args.lookback_days)
    trading_dates = sorted(pd.to_datetime(prices["date"].unique()).tolist())
    windows = _build_evaluation_windows(
        trading_dates=trading_dates,
        window_trading_days=args.window_trading_days,
        step_trading_days=args.step_trading_days,
        window_count=args.window_count,
    )
    earliest_eval_start = pd.Timestamp(windows[0]["eval_start"]).normalize()

    baseline_weights, baseline_history = generate_target_weights(features, config.strategy, None)
    baseline_result = run_backtest(
        prices=prices,
        target_weights=baseline_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        benchmark_prices=benchmark_prices,
        risk_config=config.risk,
    )

    last_year_rows = [
        _last_year_row(
            "configured_baseline",
            baseline_result,
            baseline_history,
            latest_eval_start,
            args.initial_capital,
        )
    ]
    window_rows = _window_rows(
        "configured_baseline",
        baseline_result,
        baseline_history,
        windows,
        args.initial_capital,
    )

    for model_dim in args.transformer_model_dims:
        for training_lookback_days in args.transformer_training_lookback_days:
            for sequence_lookback_window in sequence_lookback_windows:
                for target_clip_quantile in args.target_clip_quantiles:
                    objective_name = format_objective_name(target_clip_quantile)
                    model_label = (
                        f"transformer_d{model_dim}_lb{training_lookback_days}"
                        f"_seq{sequence_lookback_window}_{objective_name}"
                    )
                    model_config = TransformerModelConfig(
                        label_horizon_days=args.label_horizon_days,
                        validation_window_days=args.validation_window_days,
                        embargo_days=args.embargo_days,
                        min_training_samples=args.min_training_samples,
                        min_validation_samples=args.min_validation_samples,
                        lookback_window=sequence_lookback_window,
                        training_lookback_days=training_lookback_days,
                        target_clip_quantile=target_clip_quantile,
                        model_dim=model_dim,
                    )
                    weights, history = generate_transformer_target_weights(
                        features=features,
                        strategy_config=config.strategy,
                        model_config=model_config,
                        eval_start=earliest_eval_start,
                    )
                    result = run_backtest(
                        prices=prices,
                        target_weights=weights,
                        transaction_cost_bps=config.backtest.transaction_cost_bps,
                        benchmark_prices=benchmark_prices,
                        risk_config=config.risk,
                    )

                    extra = {
                        "model_dim": model_dim,
                        "training_lookback_days": training_lookback_days,
                        "sequence_lookback_window": sequence_lookback_window,
                        "target_clip_quantile": target_clip_quantile,
                        "objective_name": objective_name,
                    }
                    last_year_rows.append(
                        _last_year_row(model_label, result, history, latest_eval_start, args.initial_capital, extra=extra)
                    )
                    window_rows.extend(
                        _window_rows(model_label, result, history, windows, args.initial_capital, extra=extra)
                    )

    last_year_summary = pd.concat(last_year_rows, ignore_index=True)
    baseline_ending_capital = float(
        last_year_summary.loc[last_year_summary["model_name"] == "configured_baseline", "ending_capital"].iloc[0]
    )
    last_year_summary["ending_capital_gap_vs_baseline"] = last_year_summary["ending_capital"] - baseline_ending_capital
    last_year_summary["beat_baseline"] = last_year_summary["ending_capital_gap_vs_baseline"] > 0
    last_year_summary = last_year_summary.sort_values(["ending_capital", "sharpe"], ascending=[False, False]).reset_index(drop=True)

    window_summary = pd.concat(window_rows, ignore_index=True)
    baseline_capital = (
        window_summary.loc[window_summary["model_name"] == "configured_baseline", ["window_label", "ending_capital"]]
        .rename(columns={"ending_capital": "baseline_ending_capital"})
        .reset_index(drop=True)
    )
    window_summary = window_summary.merge(baseline_capital, on="window_label", how="left")
    window_summary["ending_capital_gap_vs_baseline"] = window_summary["ending_capital"] - window_summary["baseline_ending_capital"]
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
            model_dim=("model_dim", "first"),
            training_lookback_days=("training_lookback_days", "first"),
            sequence_lookback_window=("sequence_lookback_window", "first"),
            target_clip_quantile=("target_clip_quantile", "first"),
            objective_name=("objective_name", "first"),
        )
        .sort_values(["avg_ending_capital", "avg_sharpe"], ascending=[False, False])
        .reset_index(drop=True)
    )

    focused_chart = aggregate.loc[aggregate["model_name"] != "configured_baseline", [
        "model_name",
        "avg_ending_capital",
        "training_lookback_days",
        "model_dim",
        "sequence_lookback_window",
        "target_clip_quantile",
        "objective_name",
        "beat_baseline_windows",
    ]].copy()
    focused_chart = focused_chart.rename(columns={"avg_ending_capital": "avg_transformer_ending_capital"})
    best_transformer_name = str(focused_chart.iloc[0]["model_name"])
    chart_rows = window_summary.loc[
        window_summary["model_name"].isin(["configured_baseline", best_transformer_name]),
        ["eval_end", "model_name", "ending_capital"],
    ].copy()
    chart_frame = (
        chart_rows.pivot_table(index="eval_end", columns="model_name", values="ending_capital", aggfunc="last")
        .reset_index()
        .rename(columns={"eval_end": "date"})
    )
    chart_frame = chart_frame.rename(columns={column: f"{column}_value" for column in chart_frame.columns if column != "date"})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_year_path = output_dir / "transformer_last_year_sweep.csv"
    window_path = output_dir / "transformer_window_summary.csv"
    aggregate_path = output_dir / "transformer_stability_sweep.csv"
    focused_chart_path = output_dir / "transformer_stability_focus.csv"

    last_year_summary.to_csv(last_year_path, index=False)
    window_summary.to_csv(window_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    focused_chart.to_csv(focused_chart_path, index=False)

    chart_svg_path = output_dir / "transformer_stability_focus.svg"
    _build_svg(chart_frame, benchmark_name="configured_baseline_value", output_path=chart_svg_path)

    manifest = build_run_manifest(
        config,
        experiment_name="transformer_sweep",
        extra={
            "lookback_days": args.lookback_days,
            "window_trading_days": args.window_trading_days,
            "step_trading_days": args.step_trading_days,
            "window_count": args.window_count,
            "label_horizon_days": args.label_horizon_days,
            "validation_window_days": args.validation_window_days,
            "embargo_days": args.embargo_days,
            "min_training_samples": args.min_training_samples,
            "min_validation_samples": args.min_validation_samples,
            "sequence_lookback_windows": sequence_lookback_windows,
            "transformer_model_dims": args.transformer_model_dims,
            "transformer_training_lookback_days": args.transformer_training_lookback_days,
            "target_clip_quantiles": args.target_clip_quantiles,
            "best_transformer_name": best_transformer_name,
            "latest_market_date": latest_date.date().isoformat(),
            "market_data_source": market_data.provenance.get("source"),
            "market_data_manifest_path": market_data.provenance.get("manifest_path"),
            "market_data_manifest_sha256": market_data.provenance.get("manifest_sha256"),
        },
    )
    save_manifest(
        output_dir / "transformer_sweep_manifest.json",
        attach_output_files(
            manifest,
            {
                "last_year_sweep": last_year_path,
                "window_summary": window_path,
                "stability_sweep": aggregate_path,
                "focus_table": focused_chart_path,
                "focus_chart": chart_svg_path,
            },
        ),
    )

    print(last_year_summary.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
