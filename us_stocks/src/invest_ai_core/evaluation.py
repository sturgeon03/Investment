from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from invest_ai_core.performance import build_summary
from invest_ai_core.reporting import build_evaluation_row, build_value_curve


class SupportsBacktestWindow(Protocol):
    daily_returns: pd.Series
    turnover: pd.Series
    benchmark_returns: pd.Series | None


@dataclass(frozen=True, slots=True)
class WindowEvaluation:
    returns: pd.Series
    turnover: pd.Series
    benchmark_returns: pd.Series | None
    summary: pd.DataFrame
    curve: pd.DataFrame


def evaluate_backtest_window(
    result: SupportsBacktestWindow,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp | None = None,
    *,
    initial_capital: float,
) -> WindowEvaluation:
    start = pd.Timestamp(eval_start).normalize()
    if eval_end is None:
        returns = result.daily_returns.loc[result.daily_returns.index >= start]
    else:
        end = pd.Timestamp(eval_end).normalize()
        returns = result.daily_returns.loc[(result.daily_returns.index >= start) & (result.daily_returns.index <= end)]
    turnover = result.turnover.reindex(returns.index)
    benchmark_returns = (
        result.benchmark_returns.reindex(returns.index)
        if result.benchmark_returns is not None
        else None
    )
    summary = build_summary(returns, turnover, benchmark_returns)
    curve = build_value_curve(returns, benchmark_returns, initial_capital)
    return WindowEvaluation(
        returns=returns,
        turnover=turnover,
        benchmark_returns=benchmark_returns,
        summary=summary,
        curve=curve,
    )


def build_backtest_evaluation_row(
    model_name: str,
    result: SupportsBacktestWindow,
    history: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp | None = None,
    *,
    initial_capital: float,
    window_label: str | None = None,
    include_rebalance_count: bool = False,
    extra: dict[str, object] | None = None,
    metric_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    evaluation = evaluate_backtest_window(
        result,
        eval_start,
        eval_end,
        initial_capital=initial_capital,
    )
    resolved_eval_end = (
        pd.Timestamp(eval_end).normalize()
        if eval_end is not None
        else pd.Timestamp(evaluation.returns.index.max()).normalize()
    )
    return build_evaluation_row(
        model_name=model_name,
        summary=evaluation.summary,
        history=history,
        curve=evaluation.curve,
        eval_start=pd.Timestamp(evaluation.returns.index.min()).normalize(),
        eval_end=resolved_eval_end,
        initial_capital=initial_capital,
        window_label=window_label,
        include_rebalance_count=include_rebalance_count,
        extra=extra,
        metric_columns=metric_columns,
    )
