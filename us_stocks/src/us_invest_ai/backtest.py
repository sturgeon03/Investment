from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_invest_ai.config import RiskConfig
from us_invest_ai.portfolio import apply_risk_limits


@dataclass(slots=True)
class BacktestResult:
    summary: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_returns: pd.Series
    turnover: pd.Series
    benchmark_returns: pd.Series | None


def _annualized_return(equity_multiple: float, total_days: int) -> float:
    if total_days <= 0 or equity_multiple <= 0:
        return 0.0
    years = total_days / 252
    if years <= 0:
        return 0.0
    return float(equity_multiple ** (1.0 / years) - 1.0)


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def _downside_volatility(strategy_returns: pd.Series) -> float:
    if strategy_returns.empty:
        return 0.0
    downside = strategy_returns.clip(upper=0.0)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))


def _apply_risk_limits_frame(target_weights: pd.DataFrame, risk_config: RiskConfig) -> pd.DataFrame:
    limited = target_weights.fillna(0.0).apply(
        lambda row: apply_risk_limits(row.astype(float), risk_config),
        axis=1,
    )
    limited.index = target_weights.index
    limited.columns = target_weights.columns
    return limited.fillna(0.0)


def build_summary(
    strategy_returns: pd.Series,
    turnover: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    strategy_returns = strategy_returns.astype(float)
    turnover = turnover.reindex(strategy_returns.index).fillna(0.0).astype(float)
    if strategy_returns.empty:
        summary = {
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_volatility": 0.0,
            "downside_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "avg_daily_turnover": 0.0,
        }
        if benchmark_returns is not None and not benchmark_returns.empty:
            summary["benchmark_total_return"] = 0.0
            summary["benchmark_cagr"] = 0.0
            summary["excess_total_return"] = 0.0
            summary["excess_cagr"] = 0.0
            summary["tracking_error"] = 0.0
            summary["information_ratio"] = 0.0
        return pd.DataFrame([summary])

    total_days = len(strategy_returns)
    equity = (1.0 + strategy_returns).cumprod()
    annual_return = _annualized_return(float(equity.iloc[-1]), total_days)
    annual_volatility = strategy_returns.std(ddof=0) * np.sqrt(252)
    downside_volatility = _downside_volatility(strategy_returns)
    sharpe = (
        strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(252)
        if strategy_returns.std(ddof=0) > 0
        else 0.0
    )
    sortino = (
        strategy_returns.mean() * 252 / downside_volatility
        if downside_volatility > 0
        else 0.0
    )
    max_drawdown = _max_drawdown(equity)
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    summary = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "downside_volatility": float(downside_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "avg_daily_turnover": float(turnover.mean()),
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0.0).astype(float)
        benchmark_equity = (1.0 + benchmark_returns).cumprod()
        benchmark_cagr = _annualized_return(float(benchmark_equity.iloc[-1]), len(benchmark_returns))
        active_returns = strategy_returns - benchmark_returns
        active_volatility = active_returns.std(ddof=0)
        tracking_error = active_volatility * np.sqrt(252)
        information_ratio = (
            active_returns.mean() / active_volatility * np.sqrt(252)
            if active_volatility > 0
            else 0.0
        )
        summary["benchmark_total_return"] = float(benchmark_equity.iloc[-1] - 1.0)
        summary["benchmark_cagr"] = float(benchmark_cagr)
        summary["excess_total_return"] = float(equity.iloc[-1] / benchmark_equity.iloc[-1] - 1.0)
        summary["excess_cagr"] = float(annual_return - benchmark_cagr)
        summary["tracking_error"] = float(tracking_error)
        summary["information_ratio"] = float(information_ratio)

    return pd.DataFrame([summary])


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float,
    benchmark_prices: pd.DataFrame | None = None,
    risk_config: RiskConfig | None = None,
) -> BacktestResult:
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    asset_returns = closes.pct_change().fillna(0.0)
    aligned_weights = target_weights.reindex(closes.index).ffill().fillna(0.0)
    if risk_config is not None:
        aligned_weights = _apply_risk_limits_frame(aligned_weights, risk_config)
    live_weights = aligned_weights.shift(1).fillna(0.0)

    gross_returns = (live_weights * asset_returns).sum(axis=1)
    turnover = aligned_weights.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = aligned_weights.iloc[0].abs().sum()
    turnover = turnover.fillna(0.0)
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
