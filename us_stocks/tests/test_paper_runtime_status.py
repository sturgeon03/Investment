from __future__ import annotations

import json
import shutil
from pathlib import Path

from us_invest_ai.paper_runtime_status import build_runtime_summary


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_build_runtime_summary_marks_missing_status_as_stale() -> None:
    root = _fresh_dir("paper_runtime_status_missing")
    summary = build_runtime_summary(root / "latest_status.json")
    assert summary["status_exists"] is False
    assert summary["stale"] is True


def test_build_runtime_summary_compares_paper_date_to_market_date() -> None:
    root = _fresh_dir("paper_runtime_status_compare")
    latest_status_path = root / "latest_status.json"
    market_manifest_path = root / "market_data_manifest.json"

    latest_status_path.write_text(
        json.dumps(
            {
                "latest_market_date": "2026-03-25",
                "paper_state_mode": "advanced",
                "paper_state_advanced": True,
                "recommended_orders": {"order_count": 3},
                "paper_broker": {
                    "broker_backend": "alpaca",
                    "order_count_submitted": 2,
                    "fill_count": 2,
                    "ending_cash": 1250.0,
                    "ending_total_equity": 100250.0,
                },
                "paper_broker_guardrails": {
                    "violation_count": 0,
                    "ok_to_submit": True,
                },
                "paper_broker_reconciliation": {
                    "ok": True,
                    "share_delta_count": 0,
                },
                "run_directory": "runs/20260326_010203",
                "positions_path": "paper/current_positions.csv",
            }
        ),
        encoding="utf-8",
    )
    market_manifest_path.write_text(
        json.dumps({"prices_summary": {"end_date": "2026-03-26"}}),
        encoding="utf-8",
    )

    summary = build_runtime_summary(latest_status_path, market_manifest_path)
    assert summary["status_exists"] is True
    assert summary["latest_market_date"] == "2026-03-26"
    assert summary["paper_market_date"] == "2026-03-25"
    assert summary["stale"] is True
    assert summary["recommended_order_count"] == 3
    assert summary["paper_broker_backend"] == "alpaca"
    assert summary["paper_broker_order_count"] == 2
    assert summary["paper_broker_fill_count"] == 2
    assert summary["paper_broker_cash"] == 1250.0
    assert summary["paper_broker_guardrail_violation_count"] == 0
    assert summary["paper_broker_ok_to_submit"] is True
    assert summary["paper_broker_reconciliation_ok"] is True
    assert summary["paper_broker_share_delta_count"] == 0
