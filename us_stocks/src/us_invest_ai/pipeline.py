from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from us_invest_ai.backtest import BacktestResult, run_backtest
from us_invest_ai.config import RunConfig
from us_invest_ai.data import download_ohlcv, save_prices
from us_invest_ai.features import build_features
from us_invest_ai.router_dataset import build_router_training_frame
from us_invest_ai.signals import load_llm_scores
from us_invest_ai.strategy import generate_target_weights


@dataclass(slots=True)
class ResearchRun:
    prices: pd.DataFrame
    benchmark_prices: pd.DataFrame
    features: pd.DataFrame
    ranking_history: pd.DataFrame
    target_weights: pd.DataFrame
    backtest_result: BacktestResult
    llm_scores_used: pd.DataFrame | None
    all_llm_scores: pd.DataFrame | None
    router_training_frame: pd.DataFrame | None


def run_research_pipeline(config: RunConfig) -> ResearchRun:
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
    features = build_features(prices)

    llm_scores_used = (
        load_llm_scores(config.llm.signal_path, config.llm.horizon_bucket)
        if config.llm.enabled
        else None
    )
    all_llm_scores = (
        load_llm_scores(config.llm.signal_path, horizon_bucket=None)
        if config.llm.enabled
        else None
    )

    target_weights, ranking_history = generate_target_weights(features, config.strategy, llm_scores_used)
    backtest_result = run_backtest(
        prices=prices,
        target_weights=target_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        benchmark_prices=benchmark_prices,
    )
    router_training_frame = (
        build_router_training_frame(features, all_llm_scores)
        if all_llm_scores is not None
        else None
    )

    return ResearchRun(
        prices=prices,
        benchmark_prices=benchmark_prices,
        features=features,
        ranking_history=ranking_history,
        target_weights=target_weights,
        backtest_result=backtest_result,
        llm_scores_used=llm_scores_used,
        all_llm_scores=all_llm_scores,
        router_training_frame=router_training_frame,
    )


def save_research_outputs(
    run: ResearchRun,
    output_dir: Path,
    data_dir: Path,
    target_portfolio: pd.DataFrame | None = None,
    recommended_orders: pd.DataFrame | None = None,
    next_positions: pd.DataFrame | None = None,
) -> None:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_prices(run.prices, raw_dir / "prices.csv")
    save_prices(run.benchmark_prices, raw_dir / "benchmark.csv")
    run.features.to_csv(processed_dir / "features.csv", index=False)
    run.ranking_history.to_csv(output_dir / "ranking_history.csv", index=False)
    run.target_weights.to_csv(output_dir / "target_weights.csv", index_label="date")
    run.backtest_result.equity_curve.to_csv(output_dir / "equity_curve.csv", index_label="date")
    run.backtest_result.summary.to_csv(output_dir / "summary.csv", index=False)

    if run.llm_scores_used is not None:
        run.llm_scores_used.to_csv(output_dir / "llm_scores_used.csv", index=False)
    if run.router_training_frame is not None:
        run.router_training_frame.to_csv(output_dir / "router_training_frame.csv", index=False)
    if target_portfolio is not None:
        target_portfolio.to_csv(output_dir / "target_portfolio.csv", index=False)
    if recommended_orders is not None:
        recommended_orders.to_csv(output_dir / "recommended_orders.csv", index=False)
    if next_positions is not None:
        next_positions.to_csv(output_dir / "next_positions_preview.csv", index=False)
