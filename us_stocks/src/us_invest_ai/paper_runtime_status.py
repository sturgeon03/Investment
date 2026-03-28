from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_runtime_status(latest_status_path: Path) -> dict[str, Any] | None:
    if not latest_status_path.exists():
        return None
    return json.loads(latest_status_path.read_text(encoding="utf-8"))


def latest_market_date_from_manifest(manifest_path: Path | None) -> str | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    prices_summary = payload.get("prices_summary")
    if isinstance(prices_summary, dict):
        return prices_summary.get("end_date")
    return None


def build_runtime_summary(latest_status_path: Path, market_data_manifest_path: Path | None = None) -> dict[str, Any]:
    status = load_runtime_status(latest_status_path)
    if status is None:
        return {
            "status_exists": False,
            "latest_status_path": str(latest_status_path),
            "stale": True,
            "latest_market_date": None,
            "paper_market_date": None,
        }

    manifest_path = market_data_manifest_path
    if manifest_path is None:
        manifest_value = status.get("market_data_manifest_path")
        manifest_path = Path(manifest_value) if manifest_value else None

    latest_market_date = latest_market_date_from_manifest(manifest_path)
    paper_market_date = status.get("latest_market_date")
    stale = bool(latest_market_date and latest_market_date != paper_market_date)
    paper_broker = status.get("paper_broker") or {}
    paper_broker_kill_switch = status.get("paper_broker_kill_switch") or {}
    paper_broker_readiness = status.get("paper_broker_readiness") or {}
    paper_broker_guardrails = status.get("paper_broker_guardrails") or {}
    paper_broker_reconciliation = status.get("paper_broker_reconciliation") or {}

    return {
        "status_exists": True,
        "latest_status_path": str(latest_status_path),
        "market_data_manifest_path": str(manifest_path) if manifest_path is not None else None,
        "latest_market_date": latest_market_date,
        "paper_market_date": paper_market_date,
        "stale": stale,
        "paper_state_mode": status.get("paper_state_mode"),
        "paper_state_advanced": status.get("paper_state_advanced"),
        "next_recommended_action": status.get("next_recommended_action"),
        "recommended_order_count": ((status.get("recommended_orders") or {}).get("order_count")),
        "run_directory": status.get("run_directory"),
        "positions_path": status.get("positions_path"),
        "paper_broker_backend": paper_broker.get("broker_backend"),
        "paper_broker_order_count": paper_broker.get("order_count_submitted"),
        "paper_broker_fill_count": paper_broker.get("fill_count"),
        "paper_broker_cash": paper_broker.get("ending_cash"),
        "paper_broker_total_equity": paper_broker.get("ending_total_equity"),
        "paper_broker_kill_switch_active": paper_broker_kill_switch.get("active"),
        "paper_broker_kill_switch_reason": paper_broker_kill_switch.get("reason"),
        "paper_broker_readiness_ready": paper_broker_readiness.get("ready"),
        "paper_broker_connectivity_ok": paper_broker_readiness.get("connectivity_ok"),
        "paper_broker_guardrail_violation_count": paper_broker_guardrails.get("violation_count"),
        "paper_broker_ok_to_submit": paper_broker_guardrails.get("ok_to_submit"),
        "paper_broker_reconciliation_ok": paper_broker_reconciliation.get("ok"),
        "paper_broker_share_delta_count": paper_broker_reconciliation.get("share_delta_count"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the current paper-runtime state and whether it is stale.")
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
        "--fail-if-stale",
        action="store_true",
        help="Exit with status 1 when the latest paper runtime state is missing or stale.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    positions_path = Path(args.positions_path).resolve()
    latest_status_path = positions_path.parent / "runtime" / "latest_status.json"
    market_data_manifest_path = Path(args.market_data_manifest).resolve() if args.market_data_manifest else None
    summary = build_runtime_summary(latest_status_path, market_data_manifest_path)

    print(f"Paper runtime status file: {summary['latest_status_path']}")
    print(f"Paper runtime status exists: {summary['status_exists']}")
    print(f"Latest market date: {summary['latest_market_date']}")
    print(f"Paper market date: {summary['paper_market_date']}")
    print(f"Paper state mode: {summary.get('paper_state_mode')}")
    print(f"Paper state advanced: {summary.get('paper_state_advanced')}")
    print(f"Next recommended action: {summary.get('next_recommended_action')}")
    print(f"Recommended order count: {summary.get('recommended_order_count')}")
    print(f"Paper broker backend: {summary.get('paper_broker_backend')}")
    print(f"Paper broker submitted orders: {summary.get('paper_broker_order_count')}")
    print(f"Paper broker fill count: {summary.get('paper_broker_fill_count')}")
    print(f"Paper broker ending cash: {summary.get('paper_broker_cash')}")
    print(f"Paper broker ending total equity: {summary.get('paper_broker_total_equity')}")
    print(f"Paper broker kill switch active: {summary.get('paper_broker_kill_switch_active')}")
    print(f"Paper broker kill switch reason: {summary.get('paper_broker_kill_switch_reason')}")
    print(f"Paper broker readiness ok: {summary.get('paper_broker_readiness_ready')}")
    print(f"Paper broker connectivity ok: {summary.get('paper_broker_connectivity_ok')}")
    print(f"Paper broker guardrail violations: {summary.get('paper_broker_guardrail_violation_count')}")
    print(f"Paper broker ok to submit: {summary.get('paper_broker_ok_to_submit')}")
    print(f"Paper broker reconciliation ok: {summary.get('paper_broker_reconciliation_ok')}")
    print(f"Paper broker share delta count: {summary.get('paper_broker_share_delta_count')}")
    print(f"Paper runtime stale: {summary['stale']}")
    if summary.get("run_directory"):
        print(f"Latest paper run: {summary['run_directory']}")

    if args.fail_if_stale and (summary["stale"] or not summary["status_exists"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
