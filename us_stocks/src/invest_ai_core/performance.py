from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(equity_multiple: float, total_days: int) -> float:
    if total_days <= 0 or equity_multiple <= 0:
        return 0.0
    years = total_days / 252
    if years <= 0:
        return 0.0
    return float(equity_multiple ** (1.0 / years) - 1.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def downside_volatility(strategy_returns: pd.Series) -> float:
    if strategy_returns.empty:
        return 0.0
    downside = strategy_returns.clip(upper=0.0)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))


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
    annual_return = annualized_return(float(equity.iloc[-1]), total_days)
    annual_volatility = strategy_returns.std(ddof=0) * np.sqrt(252)
    downside = downside_volatility(strategy_returns)
    sharpe = (
        strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(252)
        if strategy_returns.std(ddof=0) > 0
        else 0.0
    )
    sortino = strategy_returns.mean() * 252 / downside if downside > 0 else 0.0
    drawdown = max_drawdown(equity)
    calmar = annual_return / abs(drawdown) if drawdown < 0 else 0.0

    summary = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "downside_volatility": float(downside),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown),
        "calmar": float(calmar),
        "avg_daily_turnover": float(turnover.mean()),
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0.0).astype(float)
        benchmark_equity = (1.0 + benchmark_returns).cumprod()
        benchmark_cagr = annualized_return(float(benchmark_equity.iloc[-1]), len(benchmark_returns))
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
