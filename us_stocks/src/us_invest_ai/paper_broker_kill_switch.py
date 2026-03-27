from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invest_ai_core.manifest import normalize_for_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def paper_broker_kill_switch_path(broker_root: Path) -> Path:
    return broker_root / "kill_switch.json"


def load_paper_broker_kill_switch(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def set_paper_broker_kill_switch(
    *,
    path: Path,
    active: bool,
    reason: str | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    existing = load_paper_broker_kill_switch(path)
    now_utc = _utc_now()

    payload = normalize_for_json(
        {
            "job_name": "paper_broker_kill_switch",
            "active": bool(active),
            "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None,
            "source": source,
            "updated_at_utc": now_utc,
            "activated_at_utc": (
                existing.get("activated_at_utc")
                if existing and existing.get("active") and active
                else now_utc
                if active
                else existing.get("activated_at_utc")
                if existing
                else None
            ),
            "deactivated_at_utc": now_utc if not active else None,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def append_paper_broker_kill_switch_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalize_for_json(payload), sort_keys=True, ensure_ascii=True) + "\n")


def evaluate_paper_broker_kill_switch(
    *,
    broker_root: Path,
    broker_backend: str,
    positions_path: Path,
) -> dict[str, Any]:
    control_path = paper_broker_kill_switch_path(broker_root)
    control = load_paper_broker_kill_switch(control_path)
    active = bool(control.get("active")) if control is not None else False
    return normalize_for_json(
        {
            "job_name": "paper_broker_kill_switch_status",
            "generated_at_utc": _utc_now(),
            "broker_root": str(broker_root),
            "broker_backend": broker_backend,
            "positions_path": str(positions_path),
            "kill_switch_path": str(control_path),
            "kill_switch_exists": control is not None,
            "active": active,
            "reason": control.get("reason") if control is not None else None,
            "source": control.get("source") if control is not None else None,
            "activated_at_utc": control.get("activated_at_utc") if control is not None else None,
            "deactivated_at_utc": control.get("deactivated_at_utc") if control is not None else None,
            "ok_to_submit": not active,
        }
    )


def save_paper_broker_kill_switch_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_for_json(summary), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
