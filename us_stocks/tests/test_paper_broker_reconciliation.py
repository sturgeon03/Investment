from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from us_invest_ai.paper_broker_reconciliation import reconcile_paper_broker_state


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_runtime_status(path: Path, market_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "latest_market_date": market_date,
                "paper_state_mode": "broker_submitted",
                "paper_state_advanced": True,
                "recommended_orders": {"order_count": 1},
                "positions_path": str(path.parent.parent / "current_positions.csv"),
            }
        ),
        encoding="utf-8",
    )


def test_reconciliation_passes_when_positions_and_dates_match() -> None:
    root = _fresh_dir("paper_broker_reconciliation_match")
    positions_path = root / "paper" / "current_positions.csv"
    broker_root = root / "paper" / "broker"
    runtime_status_path = root / "paper" / "runtime" / "latest_status.json"
    market_manifest_path = root / "data" / "raw" / "market_data_manifest.json"

    positions_path.parent.mkdir(parents=True, exist_ok=True)
    broker_root.mkdir(parents=True, exist_ok=True)
    market_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": ["AAPL"], "shares": [1.5]}).to_csv(positions_path, index=False)
    pd.DataFrame({"ticker": ["AAPL"], "shares": [1.5]}).to_csv(broker_root / "latest_positions.csv", index=False)
    (broker_root / "latest_account_state.json").write_text(
        json.dumps({"broker_backend": "alpaca", "market_date": "2026-03-25"}),
        encoding="utf-8",
    )
    _write_runtime_status(runtime_status_path, "2026-03-25")
    market_manifest_path.write_text(
        json.dumps({"prices_summary": {"end_date": "2026-03-25"}}),
        encoding="utf-8",
    )

    summary = reconcile_paper_broker_state(
        positions_path=positions_path,
        broker_root=broker_root,
        runtime_status_path=runtime_status_path,
        market_data_manifest_path=market_manifest_path,
    )

    assert summary["positions_match"] is True
    assert summary["market_date_match"] is True
    assert summary["runtime_stale"] is False
    assert summary["ok"] is True


def test_reconciliation_detects_share_mismatch() -> None:
    root = _fresh_dir("paper_broker_reconciliation_mismatch")
    positions_path = root / "paper" / "current_positions.csv"
    broker_root = root / "paper" / "broker"
    runtime_status_path = root / "paper" / "runtime" / "latest_status.json"
    market_manifest_path = root / "data" / "raw" / "market_data_manifest.json"

    positions_path.parent.mkdir(parents=True, exist_ok=True)
    broker_root.mkdir(parents=True, exist_ok=True)
    market_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": ["AAPL"], "shares": [2.0]}).to_csv(positions_path, index=False)
    pd.DataFrame({"ticker": ["AAPL"], "shares": [1.5]}).to_csv(broker_root / "latest_positions.csv", index=False)
    (broker_root / "latest_account_state.json").write_text(
        json.dumps({"broker_backend": "alpaca", "market_date": "2026-03-25"}),
        encoding="utf-8",
    )
    _write_runtime_status(runtime_status_path, "2026-03-25")
    market_manifest_path.write_text(
        json.dumps({"prices_summary": {"end_date": "2026-03-25"}}),
        encoding="utf-8",
    )

    summary = reconcile_paper_broker_state(
        positions_path=positions_path,
        broker_root=broker_root,
        runtime_status_path=runtime_status_path,
        market_data_manifest_path=market_manifest_path,
    )

    assert summary["positions_match"] is False
    assert summary["share_delta_count"] == 1
    assert summary["ok"] is False
