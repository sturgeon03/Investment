from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from invest_ai_core.performance import build_summary


@dataclass(slots=True)
class BacktestResult:
    summary: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_returns: pd.Series
    turnover: pd.Series
    benchmark_returns: pd.Series | None


WeightLimiter = Callable[[pd.Series], pd.Series]


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float,
    benchmark_prices: pd.DataFrame | None = None,
    weight_limiter: WeightLimiter | None = None,
) -> BacktestResult:
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    asset_returns = closes.pct_change().fillna(0.0)
    aligned_weights = target_weights.reindex(closes.index).ffill().fillna(0.0)
    if weight_limiter is not None:
        aligned_weights = aligned_weights.fillna(0.0).apply(
            lambda row: weight_limiter(row.astype(float)),
            axis=1,
        )
        aligned_weights.index = closes.index
        aligned_weights.columns = target_weights.columns
        aligned_weights = aligned_weights.fillna(0.0)
    live_weights = aligned_weights.shift(1).fillna(0.0)

    gross_returns = (live_weights * asset_returns).sum(axis=1)
    turnover = aligned_weights.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = aligned_weights.iloc[0].abs().sum()
    turnover = turnover.fillna(0.0)
    transaction_cost = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = gross_returns - transaction_cost

    equity_curve = pd.DataFrame({"strategy": (1.0 + net_returns).cumprod()}, index=closes.index)

    benchmark_returns = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_series = (
            benchmark_prices.sort_values("date")
            .drop_duplicates(subset=["date"])
            .set_index("date")["close"]
            .reindex(closes.index)
            .ffill()
        )
        benchmark_returns = benchmark_series.pct_change().fillna(0.0)
        equity_curve["benchmark"] = (1.0 + benchmark_returns).cumprod()

    summary = build_summary(net_returns, turnover, benchmark_returns)
    return BacktestResult(
        summary=summary,
        equity_curve=equity_curve,
        daily_returns=net_returns,
        turnover=turnover,
        benchmark_returns=benchmark_returns,
    )
