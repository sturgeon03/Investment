from __future__ import annotations

import argparse
from pathlib import Path

from us_invest_ai.paper_broker_kill_switch import (
    append_paper_broker_kill_switch_event,
    evaluate_paper_broker_kill_switch,
    paper_broker_kill_switch_path,
    set_paper_broker_kill_switch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the broker-backed paper OMS kill switch.")
    parser.add_argument(
        "--broker-root",
        default="us_stocks/paper/broker",
        help="Broker root that contains kill_switch.json and the OMS ledgers.",
    )
    parser.add_argument(
        "--positions-path",
        default="us_stocks/paper/current_positions.csv",
        help="Positions CSV used for status context output.",
    )
    parser.add_argument(
        "--broker-backend",
        default="local",
        help="Backend name shown in the status output.",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="Reason stored when the kill switch state is changed.",
    )
    parser.add_argument(
        "--source",
        default="manual",
        help="Source string recorded on kill switch state changes.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--activate", action="store_true", help="Activate the kill switch.")
    mode.add_argument("--deactivate", action="store_true", help="Deactivate the kill switch.")
    mode.add_argument("--status", action="store_true", help="Print the current kill switch state.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    broker_root = Path(args.broker_root).resolve()
    positions_path = Path(args.positions_path).resolve()
    control_path = paper_broker_kill_switch_path(broker_root)
    ledger_path = broker_root / "ledger" / "kill_switch_events.jsonl"

    if args.activate or args.deactivate:
        payload = set_paper_broker_kill_switch(
            path=control_path,
            active=args.activate,
            reason=args.reason or None,
            source=args.source,
        )
        append_paper_broker_kill_switch_event(ledger_path, payload)

    summary = evaluate_paper_broker_kill_switch(
        broker_root=broker_root,
        broker_backend=args.broker_backend,
        positions_path=positions_path,
    )
    print(f"Paper broker root: {broker_root}")
    print(f"Kill switch path: {summary['kill_switch_path']}")
    print(f"Kill switch exists: {summary['kill_switch_exists']}")
    print(f"Kill switch active: {summary['active']}")
    print(f"Kill switch reason: {summary.get('reason')}")
    print(f"Kill switch source: {summary.get('source')}")
    print(f"Kill switch ok to submit: {summary['ok_to_submit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
