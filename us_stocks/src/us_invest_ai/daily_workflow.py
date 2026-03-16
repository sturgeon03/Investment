from __future__ import annotations

import argparse
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from us_invest_ai.config import RunConfig, load_config
from us_invest_ai.env_utils import load_env_file
from us_invest_ai.llm_scoring import (
    HeuristicScorer,
    OpenAICompatibleConfig,
    OpenAICompatibleScorer,
    aggregate_document_scores,
    save_score_outputs,
    score_documents,
)
from us_invest_ai.pipeline import run_research_pipeline, save_research_outputs
from us_invest_ai.portfolio import (
    build_next_positions,
    build_rebalance_orders,
    build_target_portfolio,
    latest_prices_by_ticker,
    load_current_positions,
    save_table,
)
from us_invest_ai.sec_filings import (
    DEFAULT_FORMS,
    build_sec_session,
    extract_recent_filings,
    extract_scoring_documents,
    fetch_company_tickers,
    fetch_filing_documents,
    fetch_submissions,
    lookup_companies,
    resolve_user_agent,
    save_documents,
)


def _default_start_date(lookback_days: int) -> str:
    return (datetime.now().date() - timedelta(days=lookback_days)).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    return slug.strip("-_") or "run"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SEC-to-paper-trading workflow for the US stocks research stack."
    )
    parser.add_argument(
        "--config",
        default="us_stocks/config/with_llm_swing.yaml",
        help="Config file used for the research and paper portfolio step.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional explicit ticker list. Defaults to the config universe.",
    )
    parser.add_argument(
        "--forms",
        default=None,
        help="Optional comma-separated SEC forms to fetch. Defaults to the config workflow forms.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional filing date floor in YYYY-MM-DD format. Defaults to the config workflow lookback.",
    )
    parser.add_argument(
        "--limit-per-ticker",
        type=int,
        default=None,
        help="Optional SEC filing cap per ticker. Defaults to the config workflow value.",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help="Optional SEC user-agent string. Defaults to SEC_USER_AGENT.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=None,
        help="Optional pause between filing downloads. Defaults to the config workflow value.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional maximum characters kept per filing. Defaults to the config workflow value.",
    )
    parser.add_argument(
        "--min-section-chars",
        type=int,
        default=None,
        help="Optional minimum extracted section length. Defaults to the config workflow value.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["heuristic", "openai-compatible"],
        help="Optional scoring backend. Defaults to the config scoring provider.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional base URL for the OpenAI-compatible chat completion API.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name for the scoring API.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Optional environment variable containing the API key.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Optional HTTP timeout for each scoring request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature for the scoring request.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional KEY=VALUE env file loaded before SEC/API setup.",
    )
    parser.add_argument(
        "--positions-csv",
        default=None,
        help="Optional current paper positions CSV. Defaults to the config workflow path.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional directory where dated workflow runs are stored. Defaults to the config workflow path.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional suffix for the run directory.",
    )
    parser.add_argument(
        "--apply-paper-orders",
        action="store_true",
        help="Write the next positions preview back to the configured paper positions file.",
    )
    return parser.parse_args()


def _build_scorer(
    provider: str,
    base_url: str,
    model: str,
    api_key_env: str,
    timeout_seconds: int,
    temperature: float,
) -> HeuristicScorer | OpenAICompatibleScorer:
    if provider == "heuristic":
        return HeuristicScorer()
    return OpenAICompatibleScorer(
        OpenAICompatibleConfig(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    )


def _merge_aggregated_history(existing_path: Path, new_scores: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    required_columns = [
        "date",
        "ticker",
        "horizon_bucket",
        "llm_score",
        "document_count",
        "section_count",
        "avg_confidence",
        "avg_risk_flag",
    ]
    frames: list[pd.DataFrame] = []

    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        if not existing.empty:
            frames.append(existing)
    if not new_scores.empty:
        frames.append(new_scores.copy())

    if not frames:
        merged = pd.DataFrame(columns=required_columns)
    else:
        merged = pd.concat(frames, ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
        merged["ticker"] = merged["ticker"].astype(str).str.upper()
        if "horizon_bucket" not in merged.columns:
            merged["horizon_bucket"] = "swing"
        merged = merged.sort_values(["date", "ticker", "horizon_bucket"]).drop_duplicates(
            subset=["date", "ticker", "horizon_bucket"],
            keep="last",
        )
        merged = merged[required_columns].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def _build_workflow_config(config: RunConfig, signal_path: Path, run_dir: Path, enable_llm: bool) -> RunConfig:
    return replace(
        config,
        llm=replace(config.llm, enabled=enable_llm, signal_path=signal_path),
        output=replace(config.output, data_dir=run_dir / "data", artifacts_dir=run_dir / "research"),
    )


def main() -> None:
    args = _parse_args()
    base_config = load_config(args.config)
    env_file = args.env_file or (str(base_config.scoring.env_file) if base_config.scoring.env_file else None)
    if env_file:
        load_env_file(env_file)

    tickers = args.tickers or base_config.data.tickers
    forms = (
        tuple(form.strip().upper() for form in args.forms.split(",") if form.strip())
        if args.forms
        else tuple(base_config.workflow.forms)
    ) or DEFAULT_FORMS
    start_date = args.start_date or _default_start_date(base_config.workflow.start_date_lookback_days)
    limit_per_ticker = args.limit_per_ticker or base_config.workflow.limit_per_ticker
    pause_seconds = args.pause_seconds if args.pause_seconds is not None else base_config.workflow.pause_seconds
    max_chars = args.max_chars if args.max_chars is not None else base_config.workflow.max_chars
    min_section_chars = (
        args.min_section_chars
        if args.min_section_chars is not None
        else base_config.workflow.min_section_chars
    )
    provider = args.provider or base_config.scoring.provider
    base_url = args.base_url or base_config.scoring.base_url
    model = args.model or base_config.scoring.model
    api_key_env = args.api_key_env or base_config.scoring.api_key_env
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else base_config.scoring.timeout_seconds
    )
    temperature = args.temperature if args.temperature is not None else base_config.scoring.temperature
    positions_path = Path(args.positions_csv).resolve() if args.positions_csv else base_config.workflow.positions_path
    output_root = Path(args.output_root).resolve() if args.output_root else base_config.workflow.output_root
    apply_paper_orders = args.apply_paper_orders or base_config.workflow.apply_paper_orders

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{_slugify(args.run_label)}" if args.run_label else ""
    run_dir = output_root / f"{timestamp}{suffix}"

    documents_dir = run_dir / "documents"
    signals_dir = run_dir / "signals"
    paper_dir = run_dir / "paper"
    raw_filings_path = documents_dir / "sec_filings_recent.csv"
    sections_path = documents_dir / "sec_sections.csv"
    detailed_scores_path = signals_dir / "llm_document_scores.csv"
    aggregated_scores_path = signals_dir / "llm_scores.generated.csv"
    merged_signal_store_path = signals_dir / "llm_scores.merged.csv"

    user_agent = resolve_user_agent(args.user_agent)
    session = build_sec_session(user_agent)
    companies = fetch_company_tickers(session)
    matched_companies = lookup_companies(companies, tickers)

    filings = []
    for _, company in matched_companies.iterrows():
        submissions = fetch_submissions(session, int(company["cik_str"]))
        filings.extend(
            extract_recent_filings(
                submissions=submissions,
                ticker=company["ticker"],
                forms=forms,
                start_date=start_date,
                limit_per_ticker=limit_per_ticker,
            )
        )
    filings = sorted(filings, key=lambda filing: (filing.filing_date, filing.ticker), reverse=True)

    raw_filings, errors = fetch_filing_documents(
        session=session,
        filings=filings,
        max_chars=max_chars,
        pause_seconds=pause_seconds,
    )
    save_documents(raw_filings, raw_filings_path)

    sections = extract_scoring_documents(raw_filings, min_section_chars=min_section_chars)
    save_documents(sections, sections_path)

    scored_documents = score_documents(
        sections,
        _build_scorer(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        ),
    )
    aggregated_scores = aggregate_document_scores(scored_documents)
    save_score_outputs(
        scored_documents=scored_documents,
        aggregated_scores=aggregated_scores,
        detailed_output_path=detailed_scores_path,
        aggregated_output_path=aggregated_scores_path,
    )

    merged_signal_store = _merge_aggregated_history(
        existing_path=base_config.llm.signal_path,
        new_scores=aggregated_scores,
        output_path=merged_signal_store_path,
    )
    llm_enabled = base_config.llm.enabled and not merged_signal_store.empty
    workflow_config = _build_workflow_config(
        config=base_config,
        signal_path=merged_signal_store_path,
        run_dir=run_dir,
        enable_llm=llm_enabled,
    )

    run = run_research_pipeline(workflow_config)
    target_portfolio = build_target_portfolio(
        target_weights=run.target_weights,
        prices=run.prices,
        risk=workflow_config.risk,
    )
    current_positions = load_current_positions(positions_path)
    latest_prices = latest_prices_by_ticker(run.prices)
    recommended_orders = build_rebalance_orders(
        target_portfolio,
        current_positions,
        workflow_config.risk,
        latest_prices=latest_prices,
    )
    next_positions = build_next_positions(target_portfolio)

    save_research_outputs(
        run=run,
        output_dir=workflow_config.output.artifacts_dir,
        data_dir=workflow_config.output.data_dir,
        target_portfolio=target_portfolio,
        recommended_orders=recommended_orders,
        next_positions=next_positions,
    )
    save_table(target_portfolio, paper_dir / "target_portfolio.csv")
    save_table(recommended_orders, paper_dir / "recommended_orders.csv")
    save_table(next_positions, paper_dir / "next_positions_preview.csv")
    if apply_paper_orders:
        save_table(next_positions, positions_path)

    print(f"Run directory: {run_dir}")
    if env_file:
        print(f"Loaded env file: {env_file}")
    print(f"Scoring provider: {provider}")
    if provider == "openai-compatible":
        print(f"Scoring model: {model}")
    print(f"Fetched filings: {len(raw_filings)}")
    print(f"Extracted sections: {len(sections)}")
    print(f"Scored section-horizon rows: {len(scored_documents)}")
    print(f"Merged signal rows: {len(merged_signal_store)}")
    print(f"LLM enabled in research run: {llm_enabled}")
    print(f"Paper positions source: {positions_path}")
    print(f"Applied paper orders to state: {apply_paper_orders}")
    if errors:
        print(f"Skipped {len(errors)} filings due to errors.")
    print(run.backtest_result.summary.to_string(index=False))
    print(f"Paper portfolio: {paper_dir}")


if __name__ == "__main__":
    main()
