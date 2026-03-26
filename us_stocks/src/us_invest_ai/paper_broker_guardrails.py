from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from invest_ai_core.manifest import normalize_for_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_market_date(orders: pd.DataFrame) -> str | None:
    if orders.empty or "date" not in orders.columns:
        return None
    values = pd.to_datetime(orders["date"], errors="coerce").dropna()
    if values.empty:
        return None
    return values.max().date().isoformat()


def evaluate_paper_broker_guardrails(
    *,
    orders: pd.DataFrame,
    broker_root: Path,
    broker_backend: str,
    positions_path: Path,
    max_order_count: int | None = None,
    max_total_trade_notional: float | None = None,
    max_single_order_notional: float | None = None,
    allow_duplicate_market_date: bool = False,
) -> dict[str, Any]:
    normalized_orders = orders.copy()
    if not normalized_orders.empty:
        normalized_orders["trade_notional"] = pd.to_numeric(
            normalized_orders.get("trade_notional"), errors="coerce"
        ).fillna(0.0)
        normalized_orders["ticker"] = normalized_orders["ticker"].astype(str).str.upper()

    market_date = _resolve_market_date(normalized_orders)
    latest_account_state = _load_json(broker_root / "latest_account_state.json")
    order_count = int(len(normalized_orders))
    total_trade_notional = float(normalized_orders["trade_notional"].sum()) if order_count else 0.0
    max_observed_notional = float(normalized_orders["trade_notional"].max()) if order_count else 0.0

    violations: list[dict[str, Any]] = []
    if max_order_count is not None and order_count > int(max_order_count):
        violations.append(
            {
                "code": "max_order_count_exceeded",
                "message": f"Recommended order count {order_count} exceeds guardrail {int(max_order_count)}.",
            }
        )
    if max_total_trade_notional is not None and total_trade_notional > float(max_total_trade_notional):
        violations.append(
            {
                "code": "max_total_trade_notional_exceeded",
                "message": (
                    f"Total trade notional {total_trade_notional:.2f} exceeds guardrail "
                    f"{float(max_total_trade_notional):.2f}."
                ),
            }
        )
    if max_single_order_notional is not None and max_observed_notional > float(max_single_order_notional):
        violations.append(
            {
                "code": "max_single_order_notional_exceeded",
                "message": (
                    f"Largest single-order notional {max_observed_notional:.2f} exceeds guardrail "
                    f"{float(max_single_order_notional):.2f}."
                ),
            }
        )
    if (
        not allow_duplicate_market_date
        and latest_account_state is not None
        and latest_account_state.get("market_date")
        and latest_account_state.get("market_date") == market_date
        and str(latest_account_state.get("broker_backend", "")).lower() == str(broker_backend).lower()
        and str(latest_account_state.get("positions_path", "")) == str(positions_path)
    ):
        violations.append(
            {
                "code": "duplicate_market_date_submission_blocked",
                "message": (
                    f"Latest broker snapshot already covers market date {market_date} for backend "
                    f"{broker_backend}; refusing a duplicate submission."
                ),
            }
        )

    return normalize_for_json(
        {
            "job_name": "paper_broker_guardrails",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "broker_backend": broker_backend,
            "broker_root": str(broker_root),
            "positions_path": str(positions_path),
            "market_date": market_date,
            "order_count": order_count,
            "total_trade_notional": total_trade_notional,
            "max_single_order_notional_observed": max_observed_notional,
            "configured_max_order_count": int(max_order_count) if max_order_count is not None else None,
            "configured_max_total_trade_notional": (
                float(max_total_trade_notional) if max_total_trade_notional is not None else None
            ),
            "configured_max_single_order_notional": (
                float(max_single_order_notional) if max_single_order_notional is not None else None
            ),
            "allow_duplicate_market_date": allow_duplicate_market_date,
            "latest_account_market_date": (
                latest_account_state.get("market_date") if latest_account_state is not None else None
            ),
            "latest_account_backend": (
                latest_account_state.get("broker_backend") if latest_account_state is not None else None
            ),
            "violations": violations,
            "violation_count": len(violations),
            "ok_to_submit": len(violations) == 0,
        }
    )


def save_paper_broker_guardrails(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_for_json(summary), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
