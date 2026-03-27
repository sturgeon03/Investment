from __future__ import annotations

import json
import shutil
from pathlib import Path

from us_invest_ai.paper_broker_readiness import evaluate_paper_broker_readiness


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


class _FakeReadinessSession:
    def request(self, method: str, url: str, headers=None, json=None, timeout=None):
        if method == "GET" and url.endswith("/v2/account"):
            return _FakeResponse(
                {
                    "account_number": "PA-321",
                    "status": "ACTIVE",
                    "cash": "1000.0",
                    "equity": "1000.0",
                    "buying_power": "2000.0",
                }
            )
        if method == "GET" and url.endswith("/v2/positions"):
            return _FakeResponse([{"symbol": "AAPL", "qty": "1"}])
        raise AssertionError(f"Unexpected request: {method} {url}")


def test_readiness_for_local_backend_is_immediately_ready() -> None:
    root = _fresh_dir("paper_broker_readiness_local")
    summary = evaluate_paper_broker_readiness(
        broker_backend="local",
        broker_root=root / "paper" / "broker",
    )

    assert summary["ready"] is True
    assert summary["connectivity_ok"] is None


def test_readiness_for_alpaca_live_check_uses_credentials_and_account_probe(monkeypatch) -> None:
    root = _fresh_dir("paper_broker_readiness_alpaca")
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")

    summary = evaluate_paper_broker_readiness(
        broker_backend="alpaca",
        broker_root=root / "paper" / "broker",
        live_check=True,
        session=_FakeReadinessSession(),
    )

    assert summary["ready"] is True
    assert summary["credentials_present"] is True
    assert summary["connectivity_ok"] is True
    assert summary["broker_account_id"] == "PA-321"
    assert summary["broker_position_count"] == 1
