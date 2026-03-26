from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from us_invest_ai.paper_runtime import write_paper_runtime_state


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_write_paper_runtime_state_records_bootstrap_and_latest_files() -> None:
    root = _fresh_dir("paper_runtime_bootstrap")
    run_dir = root / "runs" / "20260326_120000"
    run_dir.mkdir(parents=True, exist_ok=True)
    positions_path = root / "paper" / "current_positions.csv"
    positions_path.parent.mkdir(parents=True, exist_ok=True)

    target_portfolio = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25"), pd.Timestamp("2026-03-25")],
            "ticker": ["AAA", "BBB"],
            "target_weight": [0.25, 0.20],
            "target_notional_after_rounding": [25_000.0, 20_000.0],
        }
    )
    recommended_orders = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25"), pd.Timestamp("2026-03-25")],
            "ticker": ["AAA", "BBB"],
            "side": ["BUY", "SELL"],
            "trade_notional": [25_000.0, 5_000.0],
        }
    )
    next_positions = pd.DataFrame({"ticker": ["AAA", "BBB"], "shares": [10.0, 5.0]})
    next_positions.to_csv(positions_path, index=False)

    outputs = write_paper_runtime_state(
        run_dir=run_dir,
        positions_path=positions_path,
        positions_existed_before_run=False,
        apply_paper_orders=True,
        provider="heuristic",
        llm_enabled=False,
        target_portfolio=target_portfolio,
        recommended_orders=recommended_orders,
        next_positions=next_positions,
        current_positions_after_run=next_positions,
        market_data_provenance={
            "manifest_path": str(root / "data" / "raw" / "market_data_manifest.json"),
            "prices_summary": {"end_date": "2026-03-25"},
        },
        workflow_manifest_path=run_dir / "workflow_manifest.json",
    )

    status = json.loads(outputs["latest_status_path"].read_text(encoding="utf-8"))
    assert status["paper_state_mode"] == "bootstrapped"
    assert status["paper_state_bootstrapped"] is True
    assert status["recommended_orders"]["order_count"] == 2
    assert status["recommended_orders"]["buy_count"] == 1
    assert status["latest_market_date"] == "2026-03-25"
    assert outputs["ledger_path"].exists()
    assert outputs["latest_target_path"].exists()
    assert outputs["latest_orders_path"].exists()
    assert outputs["latest_next_positions_path"].exists()
    assert outputs["latest_positions_state_path"].exists()


def test_write_paper_runtime_state_records_preview_only_when_orders_not_applied() -> None:
    root = _fresh_dir("paper_runtime_preview")
    run_dir = root / "runs" / "20260326_130000"
    run_dir.mkdir(parents=True, exist_ok=True)
    positions_path = root / "paper" / "current_positions.csv"
    positions_path.parent.mkdir(parents=True, exist_ok=True)

    empty_positions = pd.DataFrame(columns=["ticker", "shares"])
    outputs = write_paper_runtime_state(
        run_dir=run_dir,
        positions_path=positions_path,
        positions_existed_before_run=False,
        apply_paper_orders=False,
        provider="heuristic",
        llm_enabled=True,
        target_portfolio=pd.DataFrame(columns=["ticker", "target_weight", "target_notional_after_rounding"]),
        recommended_orders=pd.DataFrame(columns=["ticker", "side", "trade_notional"]),
        next_positions=empty_positions,
        current_positions_after_run=empty_positions,
        market_data_provenance=None,
        workflow_manifest_path=run_dir / "workflow_manifest.json",
    )

    status = json.loads(outputs["latest_status_path"].read_text(encoding="utf-8"))
    assert status["paper_state_mode"] == "preview_only"
    assert status["paper_state_advanced"] is False
    assert status["current_positions_after_run"]["row_count"] == 0
