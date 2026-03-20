from __future__ import annotations

from pathlib import Path

from us_invest_ai.config import (
    BacktestConfig,
    DataConfig,
    EligibilityConfig,
    LLMConfig,
    OutputConfig,
    RiskConfig,
    RunConfig,
    ScoringConfig,
    StrategyConfig,
    WorkflowConfig,
)
from us_invest_ai.refresh_market_data import refresh_market_data


def _build_config(root: Path) -> RunConfig:
    return RunConfig(
        config_path=root / "config.yaml",
        project_root=root,
        data=DataConfig(
            tickers=["AAPL", "MSFT"],
            benchmark="SPY",
            start="2024-01-01",
            end=None,
            tickers_file=None,
            metadata_file=None,
            universe_snapshots_file=None,
        ),
        strategy=StrategyConfig(
            rebalance="monthly",
            top_n=4,
            min_history_days=200,
            trend_filter_mode="soft",
            trend_penalty=0.35,
            momentum_20_weight=0.45,
            momentum_60_weight=0.35,
            volatility_weight=-0.2,
            llm_weight=0.0,
        ),
        eligibility=EligibilityConfig(
            min_close_price=5.0,
            min_dollar_volume_20=50_000_000.0,
            min_universe_age_days=120,
        ),
        backtest=BacktestConfig(transaction_cost_bps=10.0),
        llm=LLMConfig(enabled=False, signal_path=root / "signals.csv", horizon_bucket="swing"),
        scoring=ScoringConfig(
            provider="heuristic",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            timeout_seconds=60,
            temperature=0.0,
            env_file=None,
        ),
        risk=RiskConfig(
            capital_base=100_000.0,
            cash_buffer=0.04,
            max_position_weight=0.12,
            min_trade_notional=250.0,
            allow_fractional_shares=True,
        ),
        workflow=WorkflowConfig(
            forms=["10-K", "10-Q", "8-K"],
            start_date_lookback_days=180,
            limit_per_ticker=1,
            pause_seconds=0.25,
            max_chars=200_000,
            min_section_chars=250,
            positions_path=root / "paper" / "current_positions.csv",
            output_root=root / "runs",
            apply_paper_orders=False,
        ),
        output=OutputConfig(
            data_dir=root / "data",
            artifacts_dir=root / "artifacts",
        ),
    )


def test_refresh_market_data_forces_fresh_download_by_default(tmp_path, monkeypatch) -> None:
    config = _build_config(tmp_path)
    captured: dict[str, object] = {}

    class DummyBundle:
        provenance = {
            "source": "download",
            "prices_summary": {"end_date": "2026-03-19", "rows": 123, "ticker_count": 2},
            "benchmark_summary": {"rows": 61, "ticker_count": 1},
        }

    def fake_prepare_market_data_bundle(**kwargs):
        captured["prepare_kwargs"] = kwargs
        return DummyBundle()

    def fake_build_run_manifest(config_arg, experiment_name, extra):
        captured["build_run_manifest"] = {
            "config": config_arg,
            "experiment_name": experiment_name,
            "extra": extra,
        }
        return {"experiment_name": experiment_name, "extra": extra}

    def fake_save_manifest(path, manifest):
        captured["save_manifest"] = {"path": path, "manifest": manifest}

    monkeypatch.setattr(
        "us_invest_ai.refresh_market_data.prepare_market_data_bundle",
        fake_prepare_market_data_bundle,
    )
    monkeypatch.setattr(
        "us_invest_ai.refresh_market_data.build_run_manifest",
        fake_build_run_manifest,
    )
    monkeypatch.setattr("us_invest_ai.refresh_market_data.save_manifest", fake_save_manifest)

    _, manifest, manifest_path = refresh_market_data(config)

    assert captured["prepare_kwargs"]["prefer_cache"] is False
    assert captured["build_run_manifest"]["experiment_name"] == "refresh_market_data"
    assert captured["build_run_manifest"]["extra"]["latest_market_date"] == "2026-03-19"
    assert manifest["output_files"]["market_data_manifest"]["path"].endswith("market_data_manifest.json")
    assert manifest_path == tmp_path / "data" / "raw" / "refresh_run_manifest.json"
    assert captured["save_manifest"]["path"] == manifest_path
