from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from invest_ai_core.manifest import normalize_for_json
from us_invest_ai.portfolio import save_table


def _latest_market_date(market_data_provenance: dict[str, Any] | None) -> str | None:
    if not market_data_provenance:
        return None
    prices_summary = market_data_provenance.get("prices_summary")
    if isinstance(prices_summary, dict):
        return prices_summary.get("end_date")
    return None


def _market_data_manifest_path(market_data_provenance: dict[str, Any] | None) -> str | None:
    if not market_data_provenance:
        return None
    manifest_path = market_data_provenance.get("manifest_path")
    return str(manifest_path) if manifest_path is not None else None


def _portfolio_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "tickers": [],
            "total_target_weight": 0.0,
            "total_target_notional_after_rounding": 0.0,
        }
    summary = {
        "row_count": int(len(frame)),
        "tickers": sorted(frame["ticker"].astype(str).str.upper().tolist()),
        "total_target_weight": 0.0,
        "total_target_notional_after_rounding": 0.0,
    }
    if "target_weight" in frame.columns:
        summary["total_target_weight"] = float(pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0).sum())
    if "target_notional_after_rounding" in frame.columns:
        summary["total_target_notional_after_rounding"] = float(
            pd.to_numeric(frame["target_notional_after_rounding"], errors="coerce").fillna(0.0).sum()
        )
    return summary


def _orders_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "order_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "total_trade_notional": 0.0,
            "tickers": [],
        }
    side_counts = frame["side"].astype(str).value_counts()
    return {
        "order_count": int(len(frame)),
        "buy_count": int(side_counts.get("BUY", 0)),
        "sell_count": int(side_counts.get("SELL", 0)),
        "total_trade_notional": float(pd.to_numeric(frame["trade_notional"], errors="coerce").fillna(0.0).sum()),
        "tickers": sorted(frame["ticker"].astype(str).str.upper().tolist()),
    }


def _positions_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"row_count": 0, "tickers": [], "total_shares": 0.0}
    return {
        "row_count": int(len(frame)),
        "tickers": sorted(frame["ticker"].astype(str).str.upper().tolist()),
        "total_shares": float(pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0).sum()),
    }


def write_paper_runtime_state(
    *,
    run_dir: Path,
    positions_path: Path,
    positions_existed_before_run: bool,
    apply_paper_orders: bool,
    provider: str,
    llm_enabled: bool,
    target_portfolio: pd.DataFrame,
    recommended_orders: pd.DataFrame,
    next_positions: pd.DataFrame,
    current_positions_after_run: pd.DataFrame,
    market_data_provenance: dict[str, Any] | None,
    workflow_manifest_path: Path,
) -> dict[str, Any]:
    runtime_root = positions_path.parent / "runtime"
    ledger_path = runtime_root / "ledger" / "paper_run_ledger.jsonl"
    latest_status_path = runtime_root / "latest_status.json"
    latest_target_path = runtime_root / "latest_target_portfolio.csv"
    latest_orders_path = runtime_root / "latest_recommended_orders.csv"
    latest_next_positions_path = runtime_root / "latest_next_positions_preview.csv"
    latest_positions_state_path = runtime_root / "latest_current_positions.csv"

    save_table(target_portfolio, latest_target_path)
    save_table(recommended_orders, latest_orders_path)
    save_table(next_positions, latest_next_positions_path)
    save_table(current_positions_after_run, latest_positions_state_path)

    paper_state_advanced = bool(apply_paper_orders and positions_path.exists())
    paper_state_bootstrapped = bool(paper_state_advanced and not positions_existed_before_run)
    state_mode = (
        "bootstrapped"
        if paper_state_bootstrapped
        else "advanced"
        if paper_state_advanced
        else "preview_only"
    )

    status = normalize_for_json(
        {
            "job_name": "daily_workflow_paper_runtime",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(run_dir),
            "workflow_manifest_path": str(workflow_manifest_path),
            "provider": provider,
            "llm_enabled_in_research_run": llm_enabled,
            "positions_path": str(positions_path),
            "positions_existed_before_run": positions_existed_before_run,
            "paper_state_advanced": paper_state_advanced,
            "paper_state_bootstrapped": paper_state_bootstrapped,
            "paper_state_mode": state_mode,
            "latest_market_date": _latest_market_date(market_data_provenance),
            "market_data_manifest_path": _market_data_manifest_path(market_data_provenance),
            "target_portfolio": _portfolio_summary(target_portfolio),
            "recommended_orders": _orders_summary(recommended_orders),
            "current_positions_after_run": _positions_summary(current_positions_after_run),
            "latest_runtime_files": {
                "target_portfolio": str(latest_target_path),
                "recommended_orders": str(latest_orders_path),
                "next_positions_preview": str(latest_next_positions_path),
                "current_positions_state": str(latest_positions_state_path),
                "ledger": str(ledger_path),
            },
            "next_recommended_action": (
                "Inspect the latest recommended orders before promoting this paper flow toward broker-backed paper execution."
                if not recommended_orders.empty
                else "No rebalance orders were generated; review freshness and wait for the next paper cycle."
            ),
        }
    )

    latest_status_path.parent.mkdir(parents=True, exist_ok=True)
    latest_status_path.write_text(json.dumps(status, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status, sort_keys=True, ensure_ascii=True) + "\n")

    return {
        "latest_status_path": latest_status_path,
        "ledger_path": ledger_path,
        "latest_target_path": latest_target_path,
        "latest_orders_path": latest_orders_path,
        "latest_next_positions_path": latest_next_positions_path,
        "latest_positions_state_path": latest_positions_state_path,
        "status": status,
    }

