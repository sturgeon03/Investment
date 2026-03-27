from __future__ import annotations

import json
import shutil
from pathlib import Path

from us_invest_ai.paper_broker_kill_switch import (
    evaluate_paper_broker_kill_switch,
    paper_broker_kill_switch_path,
    set_paper_broker_kill_switch,
)


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_kill_switch_defaults_to_inactive_when_absent() -> None:
    root = _fresh_dir("paper_broker_kill_switch_missing")
    summary = evaluate_paper_broker_kill_switch(
        broker_root=root / "paper" / "broker",
        broker_backend="local",
        positions_path=root / "paper" / "current_positions.csv",
    )

    assert summary["active"] is False
    assert summary["ok_to_submit"] is True


def test_kill_switch_activate_and_evaluate() -> None:
    root = _fresh_dir("paper_broker_kill_switch_activate")
    broker_root = root / "paper" / "broker"
    path = paper_broker_kill_switch_path(broker_root)
    payload = set_paper_broker_kill_switch(
        path=path,
        active=True,
        reason="operator stop",
        source="test",
    )
    summary = evaluate_paper_broker_kill_switch(
        broker_root=broker_root,
        broker_backend="alpaca",
        positions_path=root / "paper" / "current_positions.csv",
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert payload["active"] is True
    assert persisted["reason"] == "operator stop"
    assert summary["active"] is True
    assert summary["ok_to_submit"] is False
    assert summary["reason"] == "operator stop"
