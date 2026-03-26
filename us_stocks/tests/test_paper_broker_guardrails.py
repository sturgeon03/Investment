from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from us_invest_ai.paper_broker_guardrails import evaluate_paper_broker_guardrails


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_guardrails_block_duplicate_market_date_submission() -> None:
    root = _fresh_dir("paper_broker_guardrails_duplicate")
    positions_path = root / "paper" / "current_positions.csv"
    broker_root = root / "paper" / "broker"
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    broker_root.mkdir(parents=True, exist_ok=True)
    (broker_root / "latest_account_state.json").write_text(
        json.dumps(
            {
                "market_date": "2026-03-25",
                "broker_backend": "local",
                "positions_path": str(positions_path),
            }
        ),
        encoding="utf-8",
    )

    orders = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25")],
            "ticker": ["AAPL"],
            "side": ["BUY"],
            "order_shares": [1.0],
            "close": [100.0],
            "trade_notional": [100.0],
        }
    )

    summary = evaluate_paper_broker_guardrails(
        orders=orders,
        broker_root=broker_root,
        broker_backend="local",
        positions_path=positions_path,
        allow_duplicate_market_date=False,
    )

    assert summary["ok_to_submit"] is False
    assert summary["violation_count"] == 1
    assert summary["violations"][0]["code"] == "duplicate_market_date_submission_blocked"


def test_guardrails_enforce_order_count_and_notional_limits() -> None:
    root = _fresh_dir("paper_broker_guardrails_limits")
    positions_path = root / "paper" / "current_positions.csv"
    broker_root = root / "paper" / "broker"
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    broker_root.mkdir(parents=True, exist_ok=True)

    orders = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25"), pd.Timestamp("2026-03-25")],
            "ticker": ["AAPL", "MSFT"],
            "side": ["BUY", "BUY"],
            "order_shares": [1.0, 2.0],
            "close": [100.0, 150.0],
            "trade_notional": [100.0, 300.0],
        }
    )

    summary = evaluate_paper_broker_guardrails(
        orders=orders,
        broker_root=broker_root,
        broker_backend="local",
        positions_path=positions_path,
        max_order_count=1,
        max_total_trade_notional=250.0,
        max_single_order_notional=200.0,
    )

    assert summary["ok_to_submit"] is False
    assert summary["violation_count"] == 3
