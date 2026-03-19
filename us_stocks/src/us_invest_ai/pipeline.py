from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from invest_ai_core.artifacts import DataFrameArtifact
from invest_ai_core.runtime import ResearchArtifactBundle, write_research_artifact_bundle
from us_invest_ai.backtest import BacktestResult, run_backtest
from us_invest_ai.config import RunConfig
from us_invest_ai.data import MarketDataBundle, prepare_market_data_bundle, save_prices
from us_invest_ai.experiment_manifest import attach_output_files, save_manifest
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
    market_data_provenance: dict[str, Any] | None


def run_research_pipeline(config: RunConfig) -> ResearchRun:
    market_data: MarketDataBundle = prepare_market_data_bundle(
        data_dir=config.output.data_dir,
        tickers=config.data.tickers,
        benchmark=config.data.benchmark,
        start=config.data.start,
        end=config.data.end,
        tickers_file=config.data.tickers_file,
        metadata_file=config.data.metadata_file,
        universe_snapshots_file=config.data.universe_snapshots_file,
    )
    features = build_features(
        market_data.prices,
        market_data.benchmark_prices,
        market_data.ticker_metadata,
        market_data.universe_snapshots,
        {
            "min_close_price": config.eligibility.min_close_price,
            "min_dollar_volume_20": config.eligibility.min_dollar_volume_20,
            "min_universe_age_days": config.eligibility.min_universe_age_days,
        },
    )

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
        prices=market_data.prices,
        target_weights=target_weights,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        benchmark_prices=market_data.benchmark_prices,
        risk_config=config.risk,
    )
    router_training_frame = (
        build_router_training_frame(features, all_llm_scores)
        if all_llm_scores is not None
        else None
    )

    return ResearchRun(
        prices=market_data.prices,
        benchmark_prices=market_data.benchmark_prices,
        features=features,
        ranking_history=ranking_history,
        target_weights=target_weights,
        backtest_result=backtest_result,
        llm_scores_used=llm_scores_used,
        all_llm_scores=all_llm_scores,
        router_training_frame=router_training_frame,
        market_data_provenance=market_data.provenance,
    )


def save_research_outputs(
    run: ResearchRun,
    output_dir: Path,
    data_dir: Path,
    target_portfolio: pd.DataFrame | None = None,
    recommended_orders: pd.DataFrame | None = None,
    next_positions: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
) -> None:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, Path] = {
        "prices": raw_dir / "prices.csv",
        "benchmark": raw_dir / "benchmark.csv",
        "market_data_manifest": raw_dir / "market_data_manifest.json",
        "features": processed_dir / "features.csv",
        "ranking_history": output_dir / "ranking_history.csv",
        "target_weights": output_dir / "target_weights.csv",
        "equity_curve": output_dir / "equity_curve.csv",
        "summary": output_dir / "summary.csv",
    }

    save_prices(run.prices, output_files["prices"])
    save_prices(run.benchmark_prices, output_files["benchmark"])
    shared_output_files = write_research_artifact_bundle(
        output_dir,
        ResearchArtifactBundle(
            features=run.features,
            ranking_history=run.ranking_history,
            target_weights=run.target_weights,
            equity_curve=run.backtest_result.equity_curve,
            summary=run.backtest_result.summary,
        ),
        extra_artifacts=[
            *(
                [DataFrameArtifact("llm_scores_used", run.llm_scores_used, "llm_scores_used.csv")]
                if run.llm_scores_used is not None
                else []
            ),
            *(
                [DataFrameArtifact("router_training_frame", run.router_training_frame, "router_training_frame.csv")]
                if run.router_training_frame is not None
                else []
            ),
            *(
                [DataFrameArtifact("target_portfolio", target_portfolio, "target_portfolio.csv")]
                if target_portfolio is not None
                else []
            ),
            *(
                [DataFrameArtifact("recommended_orders", recommended_orders, "recommended_orders.csv")]
                if recommended_orders is not None
                else []
            ),
            *(
                [DataFrameArtifact("next_positions_preview", next_positions, "next_positions_preview.csv")]
                if next_positions is not None
                else []
            ),
        ],
    )
    output_files.update(shared_output_files)

    if manifest is not None:
        save_manifest(output_dir / "run_manifest.json", attach_output_files(manifest, output_files))
