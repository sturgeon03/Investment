from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from invest_ai_core.manifest import normalize_for_json
from us_invest_ai.env_utils import load_env_file
from us_invest_ai.paper_broker import _append_jsonl, _latest_market_date
from us_invest_ai.portfolio import save_table


@dataclass(slots=True)
class AlpacaPaperConfig:
    api_key_id: str
    api_secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_qty(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _safe_float(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


def _safe_float_or_none(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def load_alpaca_paper_config(env_file: str | Path | None = None) -> AlpacaPaperConfig:
    if env_file:
        load_env_file(env_file)

    api_key_id = os.getenv("APCA_API_KEY_ID", "").strip()
    api_secret_key = os.getenv("APCA_API_SECRET_KEY", "").strip()
    base_url = os.getenv("APCA_PAPER_BASE_URL", "").strip() or "https://paper-api.alpaca.markets"
    if not api_key_id or not api_secret_key:
        raise ValueError(
            "Alpaca paper credentials are missing. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY or provide --paper-broker-env-file."
        )
    return AlpacaPaperConfig(
        api_key_id=api_key_id,
        api_secret_key=api_secret_key,
        base_url=base_url.rstrip("/"),
    )


def build_alpaca_order_payload(
    order: dict[str, Any],
    *,
    client_order_id: str,
    order_type: str = "market",
    time_in_force: str = "day",
) -> dict[str, Any]:
    side = str(order["side"]).lower()
    qty = abs(float(order["order_shares"]))
    if qty <= 0.0:
        raise ValueError("Alpaca order qty must be positive.")
    payload = {
        "symbol": str(order["ticker"]).upper(),
        "qty": _format_qty(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
        "client_order_id": client_order_id,
    }
    return payload


class AlpacaPaperClient:
    def __init__(self, config: AlpacaPaperConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, *, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.config.base_url}{path}"
        response = self.session.request(
            method=method,
            url=url,
            headers={
                "APCA-API-KEY-ID": self.config.api_key_id,
                "APCA-API-SECRET-KEY": self.config.api_secret_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        if response.content:
            return response.json()
        return None

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/v2/account")

    def get_positions(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/v2/positions")
        return payload if isinstance(payload, list) else []

    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._request("POST", "/v2/orders", payload=payload)
        return result if isinstance(result, dict) else {}


def _positions_frame_from_alpaca(positions_payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in positions_payload:
        symbol = str(item.get("symbol", "")).strip().upper()
        qty = pd.to_numeric(item.get("qty"), errors="coerce")
        if not symbol or pd.isna(qty):
            continue
        rows.append({"ticker": symbol, "shares": float(qty)})
    return pd.DataFrame(rows, columns=["ticker", "shares"])


def _orders_frame_from_alpaca(
    responses: list[dict[str, Any]],
    request_payloads: list[dict[str, Any]],
    *,
    submitted_at_utc: str,
    run_dir: Path | None,
    workflow_manifest_path: Path | None,
    market_date: str | None,
    positions_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload, response in zip(request_payloads, responses):
        requested_qty = _safe_float(payload.get("qty"))
        filled_qty = _safe_float(response.get("filled_qty"))
        rows.append(
            {
                "submitted_at_utc": submitted_at_utc,
                "order_id": response.get("id"),
                "run_directory": str(run_dir) if run_dir is not None else None,
                "workflow_manifest_path": str(workflow_manifest_path) if workflow_manifest_path is not None else None,
                "market_date": market_date,
                "ticker": response.get("symbol", payload.get("symbol")),
                "side": str(response.get("side", payload.get("side", ""))).upper(),
                "requested_shares": requested_qty,
                "filled_shares": filled_qty,
                "remaining_shares": max(0.0, requested_qty - filled_qty),
                "requested_notional": None,
                "filled_notional": _safe_float(response.get("filled_avg_price")) * filled_qty,
                "fill_price": _safe_float_or_none(response.get("filled_avg_price")),
                "fees_paid": None,
                "status": response.get("status"),
                "reject_reason": response.get("reject_reason"),
                "positions_path": str(positions_path),
                "client_order_id": response.get("client_order_id", payload.get("client_order_id")),
            }
        )
    return pd.DataFrame(rows)


def _fills_frame_from_alpaca(order_frame: pd.DataFrame) -> pd.DataFrame:
    if order_frame.empty:
        return pd.DataFrame(
            columns=[
                "filled_at_utc",
                "fill_id",
                "order_id",
                "market_date",
                "ticker",
                "side",
                "filled_shares",
                "fill_price",
                "filled_notional",
                "fees_paid",
            ]
        )
    filled = order_frame.loc[pd.to_numeric(order_frame["filled_shares"], errors="coerce").fillna(0.0) > 0.0].copy()
    if filled.empty:
        return pd.DataFrame(
            columns=[
                "filled_at_utc",
                "fill_id",
                "order_id",
                "market_date",
                "ticker",
                "side",
                "filled_shares",
                "fill_price",
                "filled_notional",
                "fees_paid",
            ]
        )
    filled["filled_at_utc"] = filled["submitted_at_utc"]
    filled["fill_id"] = filled["order_id"].astype(str) + "-broker-fill"
    filled["fees_paid"] = 0.0
    return filled[
        [
            "filled_at_utc",
            "fill_id",
            "order_id",
            "market_date",
            "ticker",
            "side",
            "filled_shares",
            "fill_price",
            "filled_notional",
            "fees_paid",
        ]
    ].reset_index(drop=True)


def submit_orders_to_alpaca_paper(
    *,
    orders: pd.DataFrame,
    latest_prices: pd.DataFrame,
    positions_path: Path,
    broker_root: Path,
    capital_base: float,
    allow_fractional_shares: bool,
    transaction_cost_bps: float,
    env_file: str | Path | None = None,
    run_dir: Path | None = None,
    workflow_manifest_path: Path | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    config = load_alpaca_paper_config(env_file)
    client = AlpacaPaperClient(config, session=session)

    normalized_orders = orders.copy()
    if not normalized_orders.empty:
        normalized_orders["ticker"] = normalized_orders["ticker"].astype(str).str.upper()
        normalized_orders["side"] = normalized_orders["side"].astype(str).str.upper()
        normalized_orders["order_shares"] = pd.to_numeric(
            normalized_orders["order_shares"], errors="coerce"
        ).fillna(0.0)
        normalized_orders = normalized_orders.loc[
            normalized_orders["side"].isin(["BUY", "SELL"]) & (normalized_orders["order_shares"].abs() > 0.0)
        ].copy()
        normalized_orders = normalized_orders.sort_values(["ticker"]).reset_index(drop=True)

    submitted_at_utc = _utc_now()
    market_date = _latest_market_date(normalized_orders, latest_prices)

    request_payloads: list[dict[str, Any]] = []
    order_responses: list[dict[str, Any]] = []
    for index, row in normalized_orders.iterrows():
        payload = build_alpaca_order_payload(
            row.to_dict(),
            client_order_id=f"paper-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{index + 1:03d}",
        )
        request_payloads.append(payload)
        order_responses.append(client.submit_order(payload))

    account_payload = client.get_account()
    positions_payload = client.get_positions()
    positions_frame = _positions_frame_from_alpaca(positions_payload)
    save_table(positions_frame, positions_path)

    orders_frame = _orders_frame_from_alpaca(
        order_responses,
        request_payloads,
        submitted_at_utc=submitted_at_utc,
        run_dir=run_dir,
        workflow_manifest_path=workflow_manifest_path,
        market_date=market_date,
        positions_path=positions_path,
    )
    fills_frame = _fills_frame_from_alpaca(orders_frame)

    latest_account_state_path = broker_root / "latest_account_state.json"
    latest_orders_path = broker_root / "latest_orders.csv"
    latest_fills_path = broker_root / "latest_fills.csv"
    latest_positions_path = broker_root / "latest_positions.csv"
    orders_ledger_path = broker_root / "ledger" / "orders.jsonl"
    fills_ledger_path = broker_root / "ledger" / "fills.jsonl"
    account_ledger_path = broker_root / "ledger" / "account_snapshots.jsonl"

    save_table(orders_frame, latest_orders_path)
    save_table(fills_frame, latest_fills_path)
    save_table(positions_frame, latest_positions_path)

    status_counts = orders_frame["status"].astype(str).value_counts().to_dict() if not orders_frame.empty else {}
    account_summary = normalize_for_json(
        {
            "job_name": "paper_broker_oms",
            "broker_backend": "alpaca",
            "generated_at_utc": submitted_at_utc,
            "market_date": market_date,
            "broker_root": str(broker_root),
            "positions_path": str(positions_path),
            "run_directory": str(run_dir) if run_dir is not None else None,
            "workflow_manifest_path": str(workflow_manifest_path) if workflow_manifest_path is not None else None,
            "capital_base": float(capital_base),
            "transaction_cost_bps": float(transaction_cost_bps),
            "allow_fractional_shares": bool(allow_fractional_shares),
            "broker_account_id": account_payload.get("account_number"),
            "broker_account_status": account_payload.get("status"),
            "starting_cash": None,
            "starting_positions_market_value": None,
            "starting_total_equity": None,
            "ending_cash": _safe_float_or_none(account_payload.get("cash")),
            "ending_positions_market_value": _safe_float_or_none(account_payload.get("long_market_value")),
            "ending_total_equity": _safe_float_or_none(account_payload.get("equity")),
            "order_count_submitted": int(len(request_payloads)),
            "filled_order_count": int(status_counts.get("filled", 0)),
            "partial_order_count": int(status_counts.get("partially_filled", 0)),
            "fill_count": int(len(fills_frame)),
            "total_fees_paid": None,
            "positions_row_count": int(len(positions_frame)),
            "positions_updated": True,
            "broker_order_status_counts": status_counts,
            "latest_files": {
                "account_state": str(latest_account_state_path),
                "orders": str(latest_orders_path),
                "fills": str(latest_fills_path),
                "positions": str(latest_positions_path),
                "orders_ledger": str(orders_ledger_path),
                "fills_ledger": str(fills_ledger_path),
                "account_ledger": str(account_ledger_path),
            },
        }
    )

    latest_account_state_path.parent.mkdir(parents=True, exist_ok=True)
    latest_account_state_path.write_text(
        json.dumps(account_summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _append_jsonl(orders_ledger_path, orders_frame.to_dict(orient="records"))
    _append_jsonl(fills_ledger_path, fills_frame.to_dict(orient="records"))
    _append_jsonl(account_ledger_path, [account_summary])

    return {
        "summary": account_summary,
        "latest_account_state_path": latest_account_state_path,
        "latest_orders_path": latest_orders_path,
        "latest_fills_path": latest_fills_path,
        "latest_positions_path": latest_positions_path,
        "orders_ledger_path": orders_ledger_path,
        "fills_ledger_path": fills_ledger_path,
        "account_ledger_path": account_ledger_path,
        "orders_frame": orders_frame,
        "fills_frame": fills_frame,
    }
