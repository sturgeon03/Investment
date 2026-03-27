from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invest_ai_core.manifest import normalize_for_json
from us_invest_ai.paper_broker_alpaca import AlpacaPaperClient, load_alpaca_paper_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_paper_broker_readiness(
    *,
    broker_backend: str,
    broker_root: Path,
    env_file: str | Path | None = None,
    live_check: bool = False,
    session: Any | None = None,
) -> dict[str, Any]:
    normalized_backend = str(broker_backend).strip().lower()
    error: str | None = None
    credentials_present: bool | None = None
    connectivity_ok: bool | None = None
    broker_account_id: str | None = None
    broker_account_status: str | None = None
    broker_cash: float | None = None
    broker_equity: float | None = None
    broker_buying_power: float | None = None
    broker_position_count: int | None = None

    if normalized_backend == "local":
        ready = True
    elif normalized_backend == "alpaca":
        try:
            config = load_alpaca_paper_config(env_file)
            credentials_present = True
        except Exception as exc:  # pragma: no cover - exact exception varies by env source
            config = None
            credentials_present = False
            error = str(exc)

        ready = bool(credentials_present)
        if ready and live_check and config is not None:
            try:
                client = AlpacaPaperClient(config, session=session)
                account = client.get_account()
                positions = client.get_positions()
                connectivity_ok = True
                broker_account_id = str(account.get("account_number") or account.get("id") or "")
                broker_account_status = str(account.get("status") or "")
                broker_cash = _safe_float_or_none(account.get("cash"))
                broker_equity = _safe_float_or_none(account.get("equity"))
                broker_buying_power = _safe_float_or_none(account.get("buying_power"))
                broker_position_count = len(positions)
            except Exception as exc:  # pragma: no cover - network/provider errors depend on runtime
                connectivity_ok = False
                ready = False
                error = str(exc)
    else:
        ready = False
        error = f"Unsupported paper broker backend: {broker_backend}"

    return normalize_for_json(
        {
            "job_name": "paper_broker_readiness",
            "generated_at_utc": _utc_now(),
            "broker_backend": normalized_backend,
            "broker_root": str(broker_root),
            "env_file": str(Path(env_file).resolve()) if env_file else None,
            "live_check_requested": bool(live_check),
            "credentials_present": credentials_present,
            "connectivity_ok": connectivity_ok,
            "broker_account_id": broker_account_id or None,
            "broker_account_status": broker_account_status or None,
            "broker_cash": broker_cash,
            "broker_equity": broker_equity,
            "broker_buying_power": broker_buying_power,
            "broker_position_count": broker_position_count,
            "error": error,
            "ready": bool(ready),
        }
    )


def save_paper_broker_readiness(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_for_json(summary), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_paper_broker_readiness_ledger(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalize_for_json(summary), sort_keys=True, ensure_ascii=True) + "\n")
