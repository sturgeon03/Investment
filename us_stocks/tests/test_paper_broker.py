from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from us_invest_ai.paper_broker import submit_orders_to_paper_broker


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_submit_orders_to_paper_broker_processes_sells_before_buys() -> None:
    root = _fresh_dir("paper_broker_sell_then_buy")
    positions_path = root / "paper" / "current_positions.csv"
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": ["AAA"], "shares": [2.0]}).to_csv(positions_path, index=False)

    orders = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25"), pd.Timestamp("2026-03-25")],
            "ticker": ["BBB", "AAA"],
            "side": ["BUY", "SELL"],
            "order_shares": [3.0, -2.0],
            "close": [100.0, 50.0],
            "trade_notional": [300.0, 100.0],
        }
    )
    latest_prices = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25"), pd.Timestamp("2026-03-25")],
            "ticker": ["AAA", "BBB"],
            "close": [50.0, 100.0],
        }
    )

    outputs = submit_orders_to_paper_broker(
        orders=orders,
        latest_prices=latest_prices,
        positions_path=positions_path,
        broker_root=root / "paper" / "broker",
        capital_base=300.0,
        allow_fractional_shares=False,
        transaction_cost_bps=0.0,
    )

    summary = outputs["summary"]
    positions = pd.read_csv(positions_path)
    account_state = json.loads(outputs["latest_account_state_path"].read_text(encoding="utf-8"))

    assert summary["filled_order_count"] == 2
    assert summary["ending_cash"] == 0.0
    assert summary["ending_total_equity"] == 300.0
    assert positions.to_dict(orient="records") == [{"ticker": "BBB", "shares": 3.0}]
    assert account_state["fill_count"] == 2


def test_submit_orders_to_paper_broker_partially_fills_when_cash_is_insufficient() -> None:
    root = _fresh_dir("paper_broker_partial_buy")
    positions_path = root / "paper" / "current_positions.csv"
    positions_path.parent.mkdir(parents=True, exist_ok=True)

    orders = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25")],
            "ticker": ["BBB"],
            "side": ["BUY"],
            "order_shares": [3.0],
            "close": [100.0],
            "trade_notional": [300.0],
        }
    )
    latest_prices = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25")],
            "ticker": ["BBB"],
            "close": [100.0],
        }
    )

    outputs = submit_orders_to_paper_broker(
        orders=orders,
        latest_prices=latest_prices,
        positions_path=positions_path,
        broker_root=root / "paper" / "broker",
        capital_base=250.0,
        allow_fractional_shares=False,
        transaction_cost_bps=0.0,
    )

    order_frame = outputs["orders_frame"]
    positions = pd.read_csv(positions_path)

    assert order_frame.loc[0, "status"] == "PARTIALLY_FILLED"
    assert order_frame.loc[0, "filled_shares"] == 2.0
    assert order_frame.loc[0, "remaining_shares"] == 1.0
    assert positions.to_dict(orient="records") == [{"ticker": "BBB", "shares": 2.0}]
    assert outputs["summary"]["ending_cash"] == 50.0
