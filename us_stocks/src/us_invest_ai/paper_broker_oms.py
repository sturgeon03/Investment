from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from us_invest_ai.paper_broker import submit_orders_to_paper_broker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit recommended orders into the local broker-shaped paper OMS."
    )
    parser.add_argument(
        "--orders-csv",
        required=True,
        help="CSV of recommended orders, typically paper/recommended_orders.csv from daily_workflow.",
    )
    parser.add_argument(
        "--latest-prices-csv",
        required=True,
        help="CSV with latest ticker closes, typically exported from the research run.",
    )
    parser.add_argument(
        "--positions-path",
        default="us_stocks/paper/current_positions.csv",
        help="Paper positions CSV updated by the OMS.",
    )
    parser.add_argument(
        "--paper-broker-root",
        default=None,
        help="Optional root directory for OMS ledgers and latest state files.",
    )
    parser.add_argument(
        "--capital-base",
        type=float,
        default=100000.0,
        help="Initial equity used when bootstrapping a missing paper account state.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="Simulated transaction cost deducted from fills.",
    )
    parser.add_argument(
        "--allow-fractional-shares",
        action="store_true",
        help="Allow fractional share fills when cash is insufficient for a whole share.",
    )
    parser.add_argument(
        "--run-directory",
        default=None,
        help="Optional workflow run directory recorded in OMS ledgers.",
    )
    parser.add_argument(
        "--workflow-manifest-path",
        default=None,
        help="Optional workflow manifest path recorded in OMS ledgers.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    orders_csv = Path(args.orders_csv).resolve()
    latest_prices_csv = Path(args.latest_prices_csv).resolve()
    positions_path = Path(args.positions_path).resolve()
    broker_root = (
        Path(args.paper_broker_root).resolve()
        if args.paper_broker_root
        else positions_path.parent / "broker"
    )

    orders = pd.read_csv(orders_csv) if orders_csv.exists() else pd.DataFrame()
    latest_prices = pd.read_csv(latest_prices_csv) if latest_prices_csv.exists() else pd.DataFrame()
    outputs = submit_orders_to_paper_broker(
        orders=orders,
        latest_prices=latest_prices,
        positions_path=positions_path,
        broker_root=broker_root,
        capital_base=args.capital_base,
        allow_fractional_shares=args.allow_fractional_shares,
        transaction_cost_bps=args.transaction_cost_bps,
        run_dir=Path(args.run_directory).resolve() if args.run_directory else None,
        workflow_manifest_path=Path(args.workflow_manifest_path).resolve() if args.workflow_manifest_path else None,
    )
    summary = outputs["summary"]

    print(f"Paper broker account state: {outputs['latest_account_state_path']}")
    print(f"Submitted orders: {summary['order_count_submitted']}")
    print(f"Filled orders: {summary['filled_order_count']}")
    print(f"Fill count: {summary['fill_count']}")
    print(f"Ending cash: {summary['ending_cash']:.2f}")
    print(f"Ending total equity: {summary['ending_total_equity']:.2f}")
    print(f"Updated positions: {outputs['latest_positions_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
