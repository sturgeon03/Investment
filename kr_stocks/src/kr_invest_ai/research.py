from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from invest_ai_core.artifacts import DataFrameArtifact, write_manifest_with_outputs
from invest_ai_core.backtest import BacktestResult, run_backtest
from invest_ai_core.config import BacktestConfig
from invest_ai_core.manifest import sha256_file, sha256_payload
from invest_ai_core.runtime import ResearchArtifactBundle, write_research_artifact_bundle
from kr_invest_ai.data_bundle import KRResearchDataRequest
from kr_invest_ai.features import build_kr_feature_frame
from kr_invest_ai.pipeline import KRPipelineRun, load_corp_codes_csv, run_kr_data_pipeline
from kr_invest_ai.strategy import KRStrategyConfig, generate_target_weights


@dataclass(slots=True)
class KRResearchRun:
    prices: pd.DataFrame
    features: pd.DataFrame
    ranking_history: pd.DataFrame
    target_weights: pd.DataFrame
    backtest_result: BacktestResult
    benchmark_prices: pd.DataFrame
    raw_run: KRPipelineRun | None
    manifest: dict[str, Any]


def _resolve_backtest_config(
    *,
    transaction_cost_bps: float,
    spread_cost_bps: float,
    market_impact_bps: float,
    market_impact_exponent: float,
    liquidity_lookback_days: int,
    backtest_config: BacktestConfig | None,
) -> BacktestConfig:
    if backtest_config is not None:
        return backtest_config
    return BacktestConfig(
        transaction_cost_bps=transaction_cost_bps,
        spread_cost_bps=spread_cost_bps,
        market_impact_bps=market_impact_bps,
        market_impact_exponent=market_impact_exponent,
        liquidity_lookback_days=liquidity_lookback_days,
    )


def run_kr_research_backtest(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None,
    filings: pd.DataFrame | None,
    *,
    strategy_config: KRStrategyConfig | None = None,
    transaction_cost_bps: float = 10.0,
    spread_cost_bps: float = 0.0,
    market_impact_bps: float = 0.0,
    market_impact_exponent: float = 0.5,
    liquidity_lookback_days: int = 20,
    backtest_config: BacktestConfig | None = None,
) -> KRResearchRun:
    resolved_strategy = strategy_config or KRStrategyConfig()
    resolved_backtest = _resolve_backtest_config(
        transaction_cost_bps=transaction_cost_bps,
        spread_cost_bps=spread_cost_bps,
        market_impact_bps=market_impact_bps,
        market_impact_exponent=market_impact_exponent,
        liquidity_lookback_days=liquidity_lookback_days,
        backtest_config=backtest_config,
    )
    features = build_kr_feature_frame(prices, filings, benchmark_prices=benchmark_prices)
    target_weights, ranking_history = generate_target_weights(features, resolved_strategy)
    backtest_result = run_backtest(
        prices=prices,
        target_weights=target_weights,
        transaction_cost_bps=resolved_backtest.transaction_cost_bps,
        benchmark_prices=benchmark_prices,
        spread_cost_bps=resolved_backtest.spread_cost_bps,
        market_impact_bps=resolved_backtest.market_impact_bps,
        market_impact_exponent=resolved_backtest.market_impact_exponent,
        liquidity_lookback_days=resolved_backtest.liquidity_lookback_days,
    )
    manifest = {
        "pipeline": "kr_research_backtest",
        "transaction_cost_bps": resolved_backtest.transaction_cost_bps,
        "spread_cost_bps": resolved_backtest.spread_cost_bps,
        "market_impact_bps": resolved_backtest.market_impact_bps,
        "market_impact_exponent": resolved_backtest.market_impact_exponent,
        "liquidity_lookback_days": resolved_backtest.liquidity_lookback_days,
        "strategy_config": asdict(resolved_strategy),
        "feature_rows": len(features),
        "rebalance_rows": int(ranking_history["selected"].fillna(False).sum()) if "selected" in ranking_history else 0,
    }
    return KRResearchRun(
        prices=prices,
        features=features,
        ranking_history=ranking_history,
        target_weights=target_weights,
        backtest_result=backtest_result,
        benchmark_prices=benchmark_prices if benchmark_prices is not None else pd.DataFrame(),
        raw_run=None,
        manifest=manifest,
    )


def run_kr_research_pipeline(
    request: KRResearchDataRequest,
    *,
    data_dir: str | Path = "kr_stocks/data",
    corp_code_map_csv: str | Path | None = None,
    use_dart: bool = False,
    strategy_config: KRStrategyConfig | None = None,
    transaction_cost_bps: float = 10.0,
    spread_cost_bps: float = 0.0,
    market_impact_bps: float = 0.0,
    market_impact_exponent: float = 0.5,
    liquidity_lookback_days: int = 20,
    backtest_config: BacktestConfig | None = None,
) -> KRResearchRun:
    from kr_invest_ai.dart_client import DARTOpenAPIClient

    corp_codes = load_corp_codes_csv(corp_code_map_csv)
    dart_client = DARTOpenAPIClient() if use_dart else None
    raw_run = run_kr_data_pipeline(
        request,
        data_dir=data_dir,
        corp_codes_by_ticker=corp_codes,
        dart_client=dart_client,
    )
    research_run = run_kr_research_backtest(
        raw_run.bundle.prices,
        raw_run.bundle.benchmark_prices,
        raw_run.bundle.filings,
        strategy_config=strategy_config,
        transaction_cost_bps=transaction_cost_bps,
        spread_cost_bps=spread_cost_bps,
        market_impact_bps=market_impact_bps,
        market_impact_exponent=market_impact_exponent,
        liquidity_lookback_days=liquidity_lookback_days,
        backtest_config=backtest_config,
    )
    research_run.raw_run = raw_run
    research_run.manifest.update(
        {
            "request_signature": raw_run.manifest.get("request_signature"),
            "raw_manifest_path": str(raw_run.manifest_path),
            "raw_manifest_sha256": sha256_file(raw_run.manifest_path),
            "corp_code_map_csv": str(Path(corp_code_map_csv).resolve()) if corp_code_map_csv else None,
            "corp_code_map_sha256": sha256_file(corp_code_map_csv),
        }
    )
    return research_run


def save_kr_research_outputs(
    run: KRResearchRun,
    *,
    output_dir: str | Path = "kr_stocks/artifacts/research",
) -> Path:
    output_files = write_research_artifact_bundle(
        output_dir,
        ResearchArtifactBundle(
            features=run.features,
            ranking_history=run.ranking_history,
            target_weights=run.target_weights,
            equity_curve=run.backtest_result.equity_curve,
            summary=run.backtest_result.summary,
        ),
        extra_artifacts=(
            [DataFrameArtifact("benchmark", run.benchmark_prices, "benchmark.csv")]
            if run.benchmark_prices is not None and not run.benchmark_prices.empty
            else []
        ),
    )
    manifest = dict(run.manifest)
    manifest["output_signature"] = sha256_payload(
        {
            "summary": run.backtest_result.summary.to_dict(orient="records"),
            "features_rows": len(run.features),
            "ranking_history_rows": len(run.ranking_history),
        }
    )
    write_manifest_with_outputs(
        output_dir,
        "run_manifest.json",
        manifest,
        output_files,
    )
    return next(iter(output_files.values())).parent
