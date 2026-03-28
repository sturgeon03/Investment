from __future__ import annotations

import json
import shutil
from pathlib import Path

from us_invest_ai.paper_runtime_incidents import build_paper_runtime_incident_summary


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_runtime_incidents_report_ok_when_no_issues_exist() -> None:
    root = _fresh_dir("paper_runtime_incidents_ok")
    latest_status_path = root / "latest_status.json"
    market_manifest_path = root / "market_data_manifest.json"
    latest_status_path.write_text(
        json.dumps(
            {
                "latest_market_date": "2026-03-27",
                "paper_state_mode": "broker_submitted",
                "paper_state_advanced": True,
                "next_recommended_action": "Inspect the broker ledger.",
                "paper_broker": {"broker_backend": "alpaca", "order_count_submitted": 1},
                "paper_broker_kill_switch": {"active": False, "reason": None},
                "paper_broker_readiness": {"ready": True, "connectivity_ok": True},
                "paper_broker_guardrails": {"violation_count": 0, "ok_to_submit": True},
                "paper_broker_reconciliation": {"ok": True, "share_delta_count": 0},
            }
        ),
        encoding="utf-8",
    )
    market_manifest_path.write_text(json.dumps({"prices_summary": {"end_date": "2026-03-27"}}), encoding="utf-8")

    summary = build_paper_runtime_incident_summary(latest_status_path, market_manifest_path)

    assert summary["ok"] is True
    assert summary["incident_count"] == 0
    assert summary["highest_severity"] == "ok"


def test_runtime_incidents_collect_errors_and_warnings() -> None:
    root = _fresh_dir("paper_runtime_incidents_bad")
    latest_status_path = root / "latest_status.json"
    market_manifest_path = root / "market_data_manifest.json"
    latest_status_path.write_text(
        json.dumps(
            {
                "latest_market_date": "2026-03-26",
                "paper_state_mode": "broker_blocked",
                "paper_state_advanced": False,
                "next_recommended_action": "Resolve the broker readiness failure.",
                "paper_broker": {"broker_backend": "alpaca"},
                "paper_broker_kill_switch": {"active": True, "reason": "manual stop"},
                "paper_broker_readiness": {"ready": False, "connectivity_ok": False},
                "paper_broker_guardrails": {"violation_count": 2, "ok_to_submit": False},
                "paper_broker_reconciliation": {"ok": False, "share_delta_count": 1},
            }
        ),
        encoding="utf-8",
    )
    market_manifest_path.write_text(json.dumps({"prices_summary": {"end_date": "2026-03-27"}}), encoding="utf-8")

    summary = build_paper_runtime_incident_summary(latest_status_path, market_manifest_path)

    assert summary["ok"] is False
    assert summary["error_count"] >= 3
    assert summary["warning_count"] == 1
    assert summary["highest_severity"] == "error"
    codes = {incident["code"] for incident in summary["incidents"]}
    assert "paper_runtime_stale" in codes
    assert "paper_broker_kill_switch_active" in codes
    assert "paper_broker_not_ready" in codes
    assert "paper_broker_guardrails_blocked" in codes
    assert "paper_broker_reconciliation_failed" in codes
