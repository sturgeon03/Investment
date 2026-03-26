from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from us_invest_ai.paper_broker import submit_orders_to_paper_broker
from us_invest_ai.paper_broker_alpaca import submit_orders_to_alpaca_paper


def submit_orders_via_paper_broker_backend(
    *,
    backend: str,
    orders: pd.DataFrame,
    latest_prices: pd.DataFrame,
    positions_path: Path,
    broker_root: Path,
    capital_base: float,
    allow_fractional_shares: bool,
    transaction_cost_bps: float,
    run_dir: Path | None = None,
    workflow_manifest_path: Path | None = None,
    env_file: str | Path | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "local":
        return submit_orders_to_paper_broker(
            orders=orders,
            latest_prices=latest_prices,
            positions_path=positions_path,
            broker_root=broker_root,
            capital_base=capital_base,
            allow_fractional_shares=allow_fractional_shares,
            transaction_cost_bps=transaction_cost_bps,
            run_dir=run_dir,
            workflow_manifest_path=workflow_manifest_path,
        )
    if normalized_backend == "alpaca":
        return submit_orders_to_alpaca_paper(
            orders=orders,
            latest_prices=latest_prices,
            positions_path=positions_path,
            broker_root=broker_root,
            capital_base=capital_base,
            allow_fractional_shares=allow_fractional_shares,
            transaction_cost_bps=transaction_cost_bps,
            env_file=env_file,
            run_dir=run_dir,
            workflow_manifest_path=workflow_manifest_path,
            session=session,
        )
    raise ValueError(f"Unsupported paper broker backend: {backend}")
