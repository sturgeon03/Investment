from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from invest_ai_core.manifest import normalize_for_json
from us_invest_ai.paper_runtime_status import build_runtime_summary
from us_invest_ai.portfolio import load_current_positions


def _load_latest_account_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _share_deltas(current_positions: pd.DataFrame, broker_positions: pd.DataFrame) -> list[dict[str, Any]]:
    current = current_positions.rename(columns={"shares": "current_shares"}).copy()
    broker = broker_positions.rename(columns={"shares": "broker_shares"}).copy()
    merged = current.merge(broker, on="ticker", how="outer")
    if merged.empty:
        return []
    merged["current_shares"] = pd.to_numeric(merged["current_shares"], errors="coerce").fillna(0.0)
    merged["broker_shares"] = pd.to_numeric(merged["broker_shares"], errors="coerce").fillna(0.0)
    merged["share_delta"] = merged["current_shares"] - merged["broker_shares"]
    deltas = merged.loc[merged["share_delta"].abs() > 1e-6].copy()
    if deltas.empty:
        return []
    return normalize_for_json(
        deltas[["ticker", "current_shares", "broker_shares", "share_delta"]]
        .sort_values("ticker")
        .to_dict(orient="records")
    )


def reconcile_paper_broker_state(
    *,
    positions_path: Path,
    broker_root: Path,
    runtime_status_path: Path | None = None,
    market_data_manifest_path: Path | None = None,
) -> dict[str, Any]:
    latest_account_state_path = broker_root / "latest_account_state.json"
    latest_positions_path = broker_root / "latest_positions.csv"
    account_state = _load_latest_account_state(latest_account_state_path)
    runtime_summary = build_runtime_summary(
        runtime_status_path or (positions_path.parent / "runtime" / "latest_status.json"),
        market_data_manifest_path,
    )

    current_positions = load_current_positions(positions_path)
    broker_positions = load_current_positions(latest_positions_path)
    deltas = _share_deltas(current_positions, broker_positions)
    positions_match = len(deltas) == 0

    broker_market_date = account_state.get("market_date") if account_state is not None else None
    runtime_market_date = runtime_summary.get("paper_market_date")
    market_date_match = (
        broker_market_date == runtime_market_date
        if broker_market_date is not None and runtime_market_date is not None
        else None
    )

    return normalize_for_json(
        {
            "job_name": "paper_broker_reconciliation",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "positions_path": str(positions_path),
            "broker_root": str(broker_root),
            "latest_account_state_path": str(latest_account_state_path),
            "latest_broker_positions_path": str(latest_positions_path),
            "runtime_status_path": str(runtime_status_path or (positions_path.parent / "runtime" / "latest_status.json")),
            "market_data_manifest_path": str(market_data_manifest_path) if market_data_manifest_path is not None else None,
            "account_state_exists": account_state is not None,
            "broker_positions_exist": latest_positions_path.exists(),
            "runtime_status_exists": runtime_summary.get("status_exists", False),
            "runtime_stale": runtime_summary.get("stale"),
            "broker_backend": account_state.get("broker_backend") if account_state is not None else None,
            "broker_market_date": broker_market_date,
            "runtime_market_date": runtime_market_date,
            "market_date_match": market_date_match,
            "positions_match": positions_match,
            "share_delta_count": len(deltas),
            "share_deltas": deltas,
            "ok": bool(
                account_state is not None
                and latest_positions_path.exists()
                and runtime_summary.get("status_exists")
                and positions_match
                and runtime_summary.get("stale") is False
                and (market_date_match is not False)
            ),
        }
    )


def save_paper_broker_reconciliation(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_for_json(summary), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_paper_broker_reconciliation_ledger(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalize_for_json(summary), sort_keys=True, ensure_ascii=True) + "\n")
