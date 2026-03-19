from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class DataConfig:
    tickers: list[str]
    benchmark: str
    start: str
    end: str | None
    tickers_file: Path | None = None
    metadata_file: Path | None = None
    universe_snapshots_file: Path | None = None


@dataclass(slots=True)
class StrategyConfig:
    rebalance: str
    top_n: int
    min_history_days: int
    trend_filter_mode: str
    trend_penalty: float
    momentum_20_weight: float
    momentum_60_weight: float
    volatility_weight: float
    llm_weight: float


@dataclass(slots=True)
class EligibilityConfig:
    min_close_price: float
    min_dollar_volume_20: float
    min_universe_age_days: int


@dataclass(slots=True)
class BacktestConfig:
    transaction_cost_bps: float


@dataclass(slots=True)
class LLMConfig:
    enabled: bool
    signal_path: Path
    horizon_bucket: str


@dataclass(slots=True)
class ScoringConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: int
    temperature: float
    env_file: Path | None


@dataclass(slots=True)
class OutputConfig:
    data_dir: Path
    artifacts_dir: Path


@dataclass(slots=True)
class RiskConfig:
    capital_base: float
    cash_buffer: float
    max_position_weight: float
    min_trade_notional: float
    allow_fractional_shares: bool


@dataclass(slots=True)
class WorkflowConfig:
    forms: list[str]
    start_date_lookback_days: int
    limit_per_ticker: int
    pause_seconds: float
    max_chars: int
    min_section_chars: int
    positions_path: Path
    output_root: Path
    apply_paper_orders: bool


@dataclass(slots=True)
class RunConfig:
    config_path: Path
    project_root: Path
    data: DataConfig
    strategy: StrategyConfig
    eligibility: EligibilityConfig
    backtest: BacktestConfig
    llm: LLMConfig
    scoring: ScoringConfig
    risk: RiskConfig
    workflow: WorkflowConfig
    output: OutputConfig


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _load_ticker_file(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in raw.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        tokens.extend(part.strip().upper() for part in stripped.replace(",", " ").split())

    tickers = [ticker for ticker in tokens if ticker]
    if not tickers:
        raise ValueError(f"No tickers found in ticker file: {path}")
    return tickers


def _load_snapshot_tickers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Universe snapshots file is empty: {path}")

        required = {"effective_date", "ticker"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Universe snapshots file is missing columns {sorted(missing)}: {path}"
            )

        tickers: list[str] = []
        seen: set[str] = set()
        for row in reader:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)

    if not tickers:
        raise ValueError(f"No tickers found in universe snapshots file: {path}")
    return tickers


def load_config(config_path: str | Path) -> RunConfig:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    data_payload = payload["data"]
    tickers_file = data_payload.get("tickers_file")
    resolved_tickers_file = _resolve_path(project_root, tickers_file) if tickers_file else None
    universe_snapshots_file = data_payload.get("universe_snapshots_file")
    resolved_universe_snapshots_file = (
        _resolve_path(project_root, universe_snapshots_file)
        if universe_snapshots_file
        else None
    )
    tickers = data_payload.get("tickers")
    if resolved_universe_snapshots_file is not None:
        tickers = _load_snapshot_tickers(resolved_universe_snapshots_file)
    if resolved_tickers_file is not None:
        tickers = _load_ticker_file(resolved_tickers_file)
    if not tickers:
        raise ValueError(
            "Config data section must define tickers, tickers_file, or universe_snapshots_file."
        )

    data = DataConfig(
        tickers=[str(ticker).upper() for ticker in tickers],
        benchmark=str(data_payload["benchmark"]).upper(),
        start=str(data_payload["start"]),
        end=data_payload.get("end"),
        tickers_file=resolved_tickers_file,
        metadata_file=_resolve_path(project_root, data_payload["metadata_file"])
        if data_payload.get("metadata_file")
        else None,
        universe_snapshots_file=resolved_universe_snapshots_file,
    )
    strategy = StrategyConfig(**payload["strategy"])
    eligibility = EligibilityConfig(
        min_close_price=float(payload.get("eligibility", {}).get("min_close_price", 0.0)),
        min_dollar_volume_20=float(payload.get("eligibility", {}).get("min_dollar_volume_20", 0.0)),
        min_universe_age_days=int(payload.get("eligibility", {}).get("min_universe_age_days", 0)),
    )
    backtest = BacktestConfig(**payload["backtest"])
    llm = LLMConfig(
        enabled=payload["llm"]["enabled"],
        signal_path=_resolve_path(project_root, payload["llm"]["signal_path"]),
        horizon_bucket=payload["llm"].get("horizon_bucket", "swing"),
    )
    scoring_payload = payload.get("scoring", {})
    env_file = scoring_payload.get("env_file")
    scoring = ScoringConfig(
        provider=str(scoring_payload.get("provider", "heuristic")),
        base_url=str(scoring_payload.get("base_url", "https://api.deepseek.com")),
        model=str(scoring_payload.get("model", "deepseek-chat")),
        api_key_env=str(scoring_payload.get("api_key_env", "DEEPSEEK_API_KEY")),
        timeout_seconds=int(scoring_payload.get("timeout_seconds", 60)),
        temperature=float(scoring_payload.get("temperature", 0.0)),
        env_file=_resolve_path(project_root, env_file) if env_file else None,
    )
    risk = RiskConfig(
        capital_base=float(payload.get("risk", {}).get("capital_base", 100_000.0)),
        cash_buffer=float(payload.get("risk", {}).get("cash_buffer", 0.05)),
        max_position_weight=float(payload.get("risk", {}).get("max_position_weight", 0.25)),
        min_trade_notional=float(payload.get("risk", {}).get("min_trade_notional", 250.0)),
        allow_fractional_shares=bool(payload.get("risk", {}).get("allow_fractional_shares", True)),
    )
    workflow_payload = payload.get("workflow", {})
    workflow = WorkflowConfig(
        forms=[str(form).upper() for form in workflow_payload.get("forms", ["10-K", "10-Q", "8-K"])],
        start_date_lookback_days=int(workflow_payload.get("start_date_lookback_days", 180)),
        limit_per_ticker=int(workflow_payload.get("limit_per_ticker", 2)),
        pause_seconds=float(workflow_payload.get("pause_seconds", 0.25)),
        max_chars=int(workflow_payload.get("max_chars", 200_000)),
        min_section_chars=int(workflow_payload.get("min_section_chars", 250)),
        positions_path=_resolve_path(
            project_root,
            str(workflow_payload.get("positions_path", "paper/current_positions.csv")),
        ),
        output_root=_resolve_path(project_root, str(workflow_payload.get("output_root", "runs"))),
        apply_paper_orders=bool(workflow_payload.get("apply_paper_orders", False)),
    )
    output = OutputConfig(
        data_dir=_resolve_path(project_root, payload["output"]["data_dir"]),
        artifacts_dir=_resolve_path(project_root, payload["output"]["artifacts_dir"]),
    )

    return RunConfig(
        config_path=config_file,
        project_root=project_root,
        data=data,
        strategy=strategy,
        eligibility=eligibility,
        backtest=backtest,
        llm=llm,
        scoring=scoring,
        risk=risk,
        workflow=workflow,
        output=output,
    )


__all__ = [
    "BacktestConfig",
    "DataConfig",
    "EligibilityConfig",
    "LLMConfig",
    "OutputConfig",
    "RiskConfig",
    "RunConfig",
    "ScoringConfig",
    "StrategyConfig",
    "WorkflowConfig",
    "load_config",
]
