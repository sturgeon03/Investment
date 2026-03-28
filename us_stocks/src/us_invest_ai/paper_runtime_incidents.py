from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invest_ai_core.manifest import normalize_for_json
from us_invest_ai.paper_runtime_status import build_runtime_summary, load_runtime_status


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_paper_runtime_incident_summary(
    latest_status_path: Path,
    market_data_manifest_path: Path | None = None,
) -> dict[str, Any]:
    runtime_status = load_runtime_status(latest_status_path) or {}
    summary = build_runtime_summary(latest_status_path, market_data_manifest_path)
    incidents: list[dict[str, Any]] = []

    def add_incident(code: str, severity: str, message: str) -> None:
        incidents.append({"code": code, "severity": severity, "message": message})

    if not summary["status_exists"]:
        add_incident("paper_runtime_missing", "error", "Paper runtime status does not exist yet.")
    if summary.get("stale"):
        add_incident(
            "paper_runtime_stale",
            "error",
            (
                f"Paper runtime market date {summary.get('paper_market_date')} does not match the latest "
                f"market date {summary.get('latest_market_date')}."
            ),
        )
    if summary.get("paper_broker_kill_switch_active"):
        add_incident(
            "paper_broker_kill_switch_active",
            "error",
            f"Broker submission is stopped by the kill switch: {summary.get('paper_broker_kill_switch_reason')}.",
        )
    if summary.get("paper_broker_readiness_ready") is False:
        add_incident(
            "paper_broker_not_ready",
            "error",
            "Broker readiness checks are failing; credentials or live connectivity are not ready.",
        )
    if (summary.get("paper_broker_guardrail_violation_count") or 0) > 0:
        add_incident(
            "paper_broker_guardrails_blocked",
            "warning",
            (
                f"Broker submission is currently blocked by "
                f"{summary.get('paper_broker_guardrail_violation_count')} guardrail violation(s)."
            ),
        )
    if summary.get("paper_broker_reconciliation_ok") is False:
        add_incident(
            "paper_broker_reconciliation_failed",
            "error",
            (
                "Broker reconciliation failed; current positions, broker positions, or paper runtime freshness "
                "do not agree."
            ),
        )

    error_count = sum(1 for incident in incidents if incident["severity"] == "error")
    warning_count = sum(1 for incident in incidents if incident["severity"] == "warning")
    if error_count:
        highest_severity = "error"
    elif warning_count:
        highest_severity = "warning"
    else:
        highest_severity = "ok"

    return normalize_for_json(
        {
            "job_name": "paper_runtime_incidents",
            "generated_at_utc": _utc_now(),
            "latest_status_path": str(latest_status_path),
            "market_data_manifest_path": str(market_data_manifest_path) if market_data_manifest_path else None,
            "paper_state_mode": summary.get("paper_state_mode"),
            "paper_broker_backend": summary.get("paper_broker_backend"),
            "incident_count": len(incidents),
            "error_count": error_count,
            "warning_count": warning_count,
            "highest_severity": highest_severity,
            "ok": len(incidents) == 0,
            "incidents": incidents,
            "next_recommended_action": runtime_status.get("next_recommended_action"),
        }
    )


def save_paper_runtime_incident_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_for_json(summary), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_paper_runtime_incident_ledger(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalize_for_json(summary), sort_keys=True, ensure_ascii=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize incidents from the latest paper runtime status.")
    parser.add_argument(
        "--positions-path",
        default="us_stocks/paper/current_positions.csv",
        help="Paper positions CSV used to locate the runtime status directory.",
    )
    parser.add_argument(
        "--market-data-manifest",
        default=None,
        help="Optional explicit market_data_manifest.json path for freshness comparison.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with status 1 when any warning or error incidents exist.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with status 1 when any error incidents exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    positions_path = Path(args.positions_path).resolve()
    runtime_root = positions_path.parent / "runtime"
    latest_status_path = runtime_root / "latest_status.json"
    latest_incidents_path = runtime_root / "latest_incidents.json"
    incidents_ledger_path = runtime_root / "ledger" / "incidents.jsonl"
    market_data_manifest_path = Path(args.market_data_manifest).resolve() if args.market_data_manifest else None

    summary = build_paper_runtime_incident_summary(latest_status_path, market_data_manifest_path)
    save_paper_runtime_incident_summary(latest_incidents_path, summary)
    append_paper_runtime_incident_ledger(incidents_ledger_path, summary)

    print(f"Paper runtime incidents file: {latest_incidents_path}")
    print(f"Paper runtime incident count: {summary['incident_count']}")
    print(f"Paper runtime highest severity: {summary['highest_severity']}")
    print(f"Paper runtime ok: {summary['ok']}")
    if summary.get("next_recommended_action"):
        print(f"Next recommended action: {summary['next_recommended_action']}")
    for incident in summary["incidents"]:
        print(f"[{incident['severity']}] {incident['code']}: {incident['message']}")

    if args.fail_on_warning and summary["incident_count"] > 0:
        return 1
    if args.fail_on_error and summary["error_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
