from __future__ import annotations

import pandas as pd

from invest_ai_core.backtest import BacktestResult, run_backtest as _run_backtest
from invest_ai_core.performance import build_summary
from us_invest_ai.config import RiskConfig
from us_invest_ai.portfolio import apply_risk_limits


def _apply_risk_limits_frame(target_weights: pd.DataFrame, risk_config: RiskConfig) -> pd.DataFrame:
    limited = target_weights.fillna(0.0).apply(
        lambda row: apply_risk_limits(row.astype(float), risk_config),
        axis=1,
    )
    limited.index = target_weights.index
    limited.columns = target_weights.columns
    return limited.fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float,
    benchmark_prices: pd.DataFrame | None = None,
    risk_config: RiskConfig | None = None,
) -> BacktestResult:
    weight_limiter = None
    if risk_config is not None:
        weight_limiter = lambda row: apply_risk_limits(row.astype(float), risk_config)

    return _run_backtest(
        prices=prices,
        target_weights=target_weights,
        transaction_cost_bps=transaction_cost_bps,
        benchmark_prices=benchmark_prices,
        weight_limiter=weight_limiter,
    )
