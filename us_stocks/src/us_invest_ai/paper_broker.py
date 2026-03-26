from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from invest_ai_core.manifest import normalize_for_json
from us_invest_ai.portfolio import load_current_positions, save_table


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_shares(value: float, allow_fractional_shares: bool) -> float:
    rounded = max(0.0, float(value))
    if allow_fractional_shares:
        return round(rounded, 6)
    return float(math.floor(rounded + 1e-12))


def _latest_market_date(orders: pd.DataFrame, latest_prices: pd.DataFrame) -> str | None:
    if "date" in orders.columns and not orders.empty:
        order_dates = pd.to_datetime(orders["date"], errors="coerce").dropna()
        if not order_dates.empty:
            return order_dates.max().date().isoformat()
    if "date" in latest_prices.columns and not latest_prices.empty:
        price_dates = pd.to_datetime(latest_prices["date"], errors="coerce").dropna()
        if not price_dates.empty:
            return price_dates.max().date().isoformat()
    return None


def _load_latest_account_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _positions_market_value(positions: pd.DataFrame, price_map: dict[str, float]) -> float:
    if positions.empty:
        return 0.0
    missing = sorted(set(positions["ticker"]) - set(price_map))
    if missing:
        raise ValueError(f"Missing latest prices for existing paper positions: {missing}")
    values = positions["ticker"].map(price_map) * positions["shares"].astype(float)
    return float(values.sum())


def _bootstrap_cash(
    positions: pd.DataFrame,
    price_map: dict[str, float],
    capital_base: float,
) -> tuple[float, float, bool]:
    positions_value = _positions_market_value(positions, price_map)
    if positions_value <= capital_base:
        return float(capital_base - positions_value), positions_value, False
    return 0.0, positions_value, True


def _normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "side", "order_shares", "close"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Paper broker orders are missing columns: {sorted(missing)}")

    normalized = orders.copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    else:
        normalized["date"] = pd.NaT
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["side"] = normalized["side"].astype(str).str.upper()
    normalized["order_shares"] = pd.to_numeric(normalized["order_shares"], errors="coerce").fillna(0.0)
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce").fillna(0.0)
    if "trade_notional" not in normalized.columns:
        normalized["trade_notional"] = normalized["order_shares"].abs() * normalized["close"]
    normalized["trade_notional"] = pd.to_numeric(normalized["trade_notional"], errors="coerce").fillna(0.0)
    normalized = normalized.loc[normalized["side"].isin(["BUY", "SELL"])].copy()
    normalized["side_priority"] = normalized["side"].map({"SELL": 0, "BUY": 1}).fillna(9)
    return normalized.sort_values(["side_priority", "ticker"]).drop(columns=["side_priority"]).reset_index(drop=True)


def _positions_to_frame(position_map: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"ticker": ticker, "shares": round(shares, 6)}
        for ticker, shares in sorted(position_map.items())
        if shares > 0
    ]
    return pd.DataFrame(rows, columns=["ticker", "shares"])


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(normalize_for_json(record), sort_keys=True, ensure_ascii=True) + "\n")


def submit_orders_to_paper_broker(
    *,
    orders: pd.DataFrame,
    latest_prices: pd.DataFrame,
    positions_path: Path,
    broker_root: Path,
    capital_base: float,
    allow_fractional_shares: bool,
    transaction_cost_bps: float,
    run_dir: Path | None = None,
    workflow_manifest_path: Path | None = None,
) -> dict[str, Any]:
    normalized_orders = _normalize_orders(orders)
    latest_prices_frame = latest_prices.copy()
    if not latest_prices_frame.empty:
        latest_prices_frame["ticker"] = latest_prices_frame["ticker"].astype(str).str.upper()
        latest_prices_frame["close"] = pd.to_numeric(latest_prices_frame["close"], errors="coerce").fillna(0.0)
    price_map = {
        str(row["ticker"]).upper(): float(row["close"])
        for _, row in latest_prices_frame.iterrows()
        if float(row["close"]) > 0.0
    }

    current_positions = load_current_positions(positions_path)
    positions_map = {
        str(row["ticker"]).upper(): float(row["shares"])
        for _, row in current_positions.iterrows()
        if float(row["shares"]) > 0.0
    }

    latest_account_state_path = broker_root / "latest_account_state.json"
    existing_account_state = _load_latest_account_state(latest_account_state_path)
    cash_clipped_by_positions = False
    if existing_account_state is not None:
        starting_cash = float(existing_account_state.get("cash", capital_base))
        starting_positions_value = _positions_market_value(current_positions, price_map) if not current_positions.empty else 0.0
    else:
        starting_cash, starting_positions_value, cash_clipped_by_positions = _bootstrap_cash(
            current_positions,
            price_map,
            capital_base,
        )
    starting_total_equity = float(starting_cash + starting_positions_value)

    now_utc = _utc_now()
    fee_rate = max(0.0, float(transaction_cost_bps)) / 10_000.0
    market_date = _latest_market_date(normalized_orders, latest_prices_frame)

    order_records: list[dict[str, Any]] = []
    fill_records: list[dict[str, Any]] = []
    cash = float(starting_cash)
    total_fees = 0.0
    filled_order_count = 0
    partial_order_count = 0

    for index, row in normalized_orders.iterrows():
        ticker = str(row["ticker"]).upper()
        side = str(row["side"]).upper()
        requested_shares = abs(float(row["order_shares"]))
        price = float(row["close"]) if float(row["close"]) > 0.0 else float(price_map.get(ticker, 0.0))
        requested_notional = float(requested_shares * price) if price > 0.0 else 0.0
        order_id = f"paper-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{index + 1:03d}"

        fill_shares = 0.0
        reject_reason = None
        if price <= 0.0:
            reject_reason = "missing_price"
        elif side == "SELL":
            available_shares = float(positions_map.get(ticker, 0.0))
            fill_shares = _round_shares(min(requested_shares, available_shares), allow_fractional_shares)
            if fill_shares <= 0.0:
                reject_reason = "insufficient_shares"
        elif side == "BUY":
            affordable_shares = cash / (price * (1.0 + fee_rate)) if price > 0.0 else 0.0
            fill_shares = _round_shares(min(requested_shares, affordable_shares), allow_fractional_shares)
            if fill_shares <= 0.0:
                reject_reason = "insufficient_cash"
        else:
            reject_reason = "unsupported_side"

        filled_notional = float(fill_shares * price) if fill_shares > 0.0 else 0.0
        fees_paid = float(filled_notional * fee_rate)
        remaining_shares = float(max(0.0, requested_shares - fill_shares))

        if fill_shares > 0.0:
            total_fees += fees_paid
            if side == "BUY":
                cash -= filled_notional + fees_paid
                positions_map[ticker] = positions_map.get(ticker, 0.0) + fill_shares
            else:
                cash += filled_notional - fees_paid
                updated_shares = positions_map.get(ticker, 0.0) - fill_shares
                if updated_shares > 0.0:
                    positions_map[ticker] = updated_shares
                else:
                    positions_map.pop(ticker, None)

            status = "FILLED" if remaining_shares <= 0.0 else "PARTIALLY_FILLED"
            if status == "FILLED":
                filled_order_count += 1
            else:
                partial_order_count += 1
            fill_record = {
                "filled_at_utc": now_utc,
                "fill_id": f"{order_id}-fill-1",
                "order_id": order_id,
                "market_date": market_date,
                "ticker": ticker,
                "side": side,
                "filled_shares": fill_shares,
                "fill_price": price,
                "filled_notional": filled_notional,
                "fees_paid": fees_paid,
            }
            fill_records.append(fill_record)
        else:
            status = "REJECTED"

        order_record = {
            "submitted_at_utc": now_utc,
            "order_id": order_id,
            "run_directory": str(run_dir) if run_dir is not None else None,
            "workflow_manifest_path": str(workflow_manifest_path) if workflow_manifest_path is not None else None,
            "market_date": market_date,
            "ticker": ticker,
            "side": side,
            "requested_shares": requested_shares,
            "filled_shares": fill_shares,
            "remaining_shares": remaining_shares,
            "requested_notional": requested_notional,
            "filled_notional": filled_notional,
            "fill_price": price if fill_shares > 0.0 else None,
            "fees_paid": fees_paid,
            "status": status,
            "reject_reason": reject_reason,
            "positions_path": str(positions_path),
        }
        order_records.append(order_record)

    ending_positions = _positions_to_frame(positions_map)
    save_table(ending_positions, positions_path)

    positions_value = _positions_market_value(ending_positions, price_map) if not ending_positions.empty else 0.0
    total_equity = float(cash + positions_value)

    latest_orders_path = broker_root / "latest_orders.csv"
    latest_fills_path = broker_root / "latest_fills.csv"
    latest_positions_path = broker_root / "latest_positions.csv"
    account_ledger_path = broker_root / "ledger" / "account_snapshots.jsonl"
    orders_ledger_path = broker_root / "ledger" / "orders.jsonl"
    fills_ledger_path = broker_root / "ledger" / "fills.jsonl"

    orders_frame = pd.DataFrame(order_records).reindex(
        columns=[
            "submitted_at_utc",
            "order_id",
            "run_directory",
            "workflow_manifest_path",
            "market_date",
            "ticker",
            "side",
            "requested_shares",
            "filled_shares",
            "remaining_shares",
            "requested_notional",
            "filled_notional",
            "fill_price",
            "fees_paid",
            "status",
            "reject_reason",
            "positions_path",
        ]
    )
    fills_frame = pd.DataFrame(fill_records).reindex(
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
    save_table(orders_frame, latest_orders_path)
    save_table(fills_frame, latest_fills_path)
    save_table(ending_positions, latest_positions_path)

    account_summary = normalize_for_json(
        {
            "job_name": "paper_broker_oms",
            "broker_backend": "local",
            "generated_at_utc": now_utc,
            "market_date": market_date,
            "broker_root": str(broker_root),
            "positions_path": str(positions_path),
            "run_directory": str(run_dir) if run_dir is not None else None,
            "workflow_manifest_path": str(workflow_manifest_path) if workflow_manifest_path is not None else None,
            "capital_base": float(capital_base),
            "transaction_cost_bps": float(transaction_cost_bps),
            "allow_fractional_shares": bool(allow_fractional_shares),
            "bootstrapped_from_existing_account": existing_account_state is not None,
            "bootstrap_cash_clipped_by_positions": cash_clipped_by_positions,
            "starting_cash": float(starting_cash),
            "starting_positions_market_value": float(starting_positions_value),
            "starting_total_equity": float(starting_total_equity),
            "ending_cash": float(cash),
            "ending_positions_market_value": float(positions_value),
            "ending_total_equity": float(total_equity),
            "order_count_submitted": int(len(order_records)),
            "filled_order_count": int(filled_order_count),
            "partial_order_count": int(partial_order_count),
            "fill_count": int(len(fill_records)),
            "total_fees_paid": float(total_fees),
            "positions_row_count": int(len(ending_positions)),
            "positions_updated": True,
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

    _append_jsonl(orders_ledger_path, order_records)
    _append_jsonl(fills_ledger_path, fill_records)
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
