from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from us_invest_ai.paper_broker_alpaca import build_alpaca_order_payload, submit_orders_to_alpaca_paper


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeAlpacaSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self._order_counter = 0

    def request(self, method: str, url: str, headers=None, json=None, timeout=None):
        self.calls.append((method, url, json))
        if method == "POST" and url.endswith("/v2/orders"):
            self._order_counter += 1
            qty = str(json["qty"])
            return _FakeResponse(
                {
                    "id": f"order-{self._order_counter}",
                    "client_order_id": json["client_order_id"],
                    "symbol": json["symbol"],
                    "side": json["side"],
                    "status": "filled",
                    "filled_qty": qty,
                    "filled_avg_price": "100.0",
                }
            )
        if method == "GET" and url.endswith("/v2/account"):
            return _FakeResponse(
                {
                    "account_number": "PA-123",
                    "status": "ACTIVE",
                    "cash": "850.0",
                    "equity": "1000.0",
                    "long_market_value": "150.0",
                }
            )
        if method == "GET" and url.endswith("/v2/positions"):
            return _FakeResponse(
                [
                    {
                        "symbol": "AAPL",
                        "qty": "1.5",
                    }
                ]
            )
        raise AssertionError(f"Unexpected request: {method} {url}")


def test_build_alpaca_order_payload_formats_market_order() -> None:
    payload = build_alpaca_order_payload(
        {"ticker": "aapl", "side": "BUY", "order_shares": 1.5},
        client_order_id="paper-001",
    )

    assert payload == {
        "symbol": "AAPL",
        "qty": "1.5",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": "paper-001",
    }


def test_submit_orders_to_alpaca_paper_writes_synced_state(monkeypatch) -> None:
    root = _fresh_dir("paper_broker_alpaca")
    positions_path = root / "paper" / "current_positions.csv"
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    session = _FakeAlpacaSession()

    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")

    orders = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25")],
            "ticker": ["AAPL"],
            "side": ["BUY"],
            "order_shares": [1.5],
            "close": [100.0],
            "trade_notional": [150.0],
        }
    )
    latest_prices = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-25")],
            "ticker": ["AAPL"],
            "close": [100.0],
        }
    )

    outputs = submit_orders_to_alpaca_paper(
        orders=orders,
        latest_prices=latest_prices,
        positions_path=positions_path,
        broker_root=root / "paper" / "broker",
        capital_base=1000.0,
        allow_fractional_shares=True,
        transaction_cost_bps=10.0,
        session=session,
    )

    summary = outputs["summary"]
    positions = pd.read_csv(positions_path)
    account_state = json.loads(outputs["latest_account_state_path"].read_text(encoding="utf-8"))

    assert summary["broker_backend"] == "alpaca"
    assert summary["order_count_submitted"] == 1
    assert summary["filled_order_count"] == 1
    assert summary["fill_count"] == 1
    assert summary["ending_cash"] == 850.0
    assert summary["ending_total_equity"] == 1000.0
    assert positions.to_dict(orient="records") == [{"ticker": "AAPL", "shares": 1.5}]
    assert account_state["broker_account_id"] == "PA-123"
    assert any(call[0] == "POST" and call[1].endswith("/v2/orders") for call in session.calls)
