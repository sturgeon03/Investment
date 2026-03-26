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

    return {
        "status_exists": True,
        "latest_status_path": str(latest_status_path),
        "market_data_manifest_path": str(manifest_path) if manifest_path is not None else None,
        "latest_market_date": latest_market_date,
        "paper_market_date": paper_market_date,
        "stale": stale,
        "paper_state_mode": status.get("paper_state_mode"),
        "paper_state_advanced": status.get("paper_state_advanced"),
        "recommended_order_count": ((status.get("recommended_orders") or {}).get("order_count")),
        "run_directory": status.get("run_directory"),
        "positions_path": status.get("positions_path"),
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
    print(f"Recommended order count: {summary.get('recommended_order_count')}")
    print(f"Paper runtime stale: {summary['stale']}")
    if summary.get("run_directory"):
        print(f"Latest paper run: {summary['run_directory']}")

    if args.fail_if_stale and (summary["stale"] or not summary["status_exists"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
