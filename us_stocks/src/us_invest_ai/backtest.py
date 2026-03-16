from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    summary: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_returns: pd.Series
    turnover: pd.Series
    benchmark_returns: pd.Series | None


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def build_summary(
    strategy_returns: pd.Series,
    turnover: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    total_days = max(len(strategy_returns), 1)
    years = total_days / 252
    equity = (1.0 + strategy_returns).cumprod()

    annual_return = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else 0.0
    annual_volatility = strategy_returns.std(ddof=0) * np.sqrt(252)
    sharpe = (
        strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(252)
        if strategy_returns.std(ddof=0) > 0
        else 0.0
    )

    summary = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()),
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark_equity = (1.0 + benchmark_returns).cumprod()
        benchmark_years = max(len(benchmark_returns), 1) / 252
        benchmark_cagr = (
            benchmark_equity.iloc[-1] ** (1.0 / benchmark_years) - 1.0
            if benchmark_years > 0
            else 0.0
        )
        summary["benchmark_total_return"] = float(benchmark_equity.iloc[-1] - 1.0)
        summary["benchmark_cagr"] = float(benchmark_cagr)

    return pd.DataFrame([summary])


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float,
    benchmark_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    asset_returns = closes.pct_change().fillna(0.0)
    aligned_weights = target_weights.reindex(closes.index).ffill().fillna(0.0)
    live_weights = aligned_weights.shift(1).fillna(0.0)

    gross_returns = (live_weights * asset_returns).sum(axis=1)
    turnover = aligned_weights.diff().abs().sum(axis=1).fillna(0.0)
    transaction_cost = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = gross_returns - transaction_cost

    equity_curve = pd.DataFrame(
        {
            "strategy": (1.0 + net_returns).cumprod(),
        },
        index=closes.index,
    )

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
