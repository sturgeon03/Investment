from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from invest_ai_core.performance import build_summary
from invest_ai_core.performance import annualized_return


@dataclass(slots=True)
class BacktestResult:
    summary: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_returns: pd.Series
    turnover: pd.Series
    benchmark_returns: pd.Series | None
    gross_daily_returns: pd.Series | None = None
    linear_costs: pd.Series | None = None
    spread_costs: pd.Series | None = None
    market_impact_costs: pd.Series | None = None
    max_participation_rate: pd.Series | None = None


WeightLimiter = Callable[[pd.Series], pd.Series]


def enrich_summary_with_execution_details(
    summary: pd.DataFrame,
    *,
    net_returns: pd.Series,
    gross_returns: pd.Series | None,
    linear_costs: pd.Series | None,
    spread_costs: pd.Series | None,
    market_impact_costs: pd.Series | None,
    max_participation_rate: pd.Series | None,
) -> pd.DataFrame:
    enriched = summary.copy()
    if net_returns.empty:
        enriched["gross_total_return"] = 0.0
        enriched["gross_cagr"] = 0.0
        enriched["cost_drag_total_return"] = 0.0
        enriched["avg_daily_linear_cost"] = 0.0
        enriched["avg_daily_spread_cost"] = 0.0
        enriched["avg_daily_market_impact_cost"] = 0.0
        enriched["avg_daily_total_cost"] = 0.0
        enriched["avg_daily_max_participation_rate"] = 0.0
        return enriched

    aligned_index = net_returns.index
    gross = gross_returns.reindex(aligned_index).fillna(net_returns) if gross_returns is not None else net_returns
    linear = linear_costs.reindex(aligned_index).fillna(0.0) if linear_costs is not None else pd.Series(0.0, index=aligned_index)
    spread = spread_costs.reindex(aligned_index).fillna(0.0) if spread_costs is not None else pd.Series(0.0, index=aligned_index)
    impact = (
        market_impact_costs.reindex(aligned_index).fillna(0.0)
        if market_impact_costs is not None
        else pd.Series(0.0, index=aligned_index)
    )
    participation = (
        max_participation_rate.reindex(aligned_index).fillna(0.0)
        if max_participation_rate is not None
        else pd.Series(0.0, index=aligned_index)
    )
    total_costs = linear + spread + impact

    gross_equity = (1.0 + gross).cumprod()
    net_equity = (1.0 + net_returns).cumprod()
    enriched["gross_total_return"] = float(gross_equity.iloc[-1] - 1.0)
    enriched["gross_cagr"] = float(annualized_return(float(gross_equity.iloc[-1]), len(gross)))
    enriched["cost_drag_total_return"] = float(gross_equity.iloc[-1] - net_equity.iloc[-1])
    enriched["avg_daily_linear_cost"] = float(linear.mean())
    enriched["avg_daily_spread_cost"] = float(spread.mean())
    enriched["avg_daily_market_impact_cost"] = float(impact.mean())
    enriched["avg_daily_total_cost"] = float(total_costs.mean())
    enriched["avg_daily_max_participation_rate"] = float(participation.mean())
    return enriched


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float,
    benchmark_prices: pd.DataFrame | None = None,
    weight_limiter: WeightLimiter | None = None,
    *,
    spread_cost_bps: float = 0.0,
    market_impact_bps: float = 0.0,
    market_impact_exponent: float = 0.5,
    liquidity_lookback_days: int = 20,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if "volume" in prices.columns:
        volumes = (
            prices.pivot(index="date", columns="ticker", values="volume")
            .reindex(index=closes.index, columns=closes.columns)
            .fillna(0.0)
        )
    else:
        # Some tests and adapter outputs only provide close prices; treat missing
        # volume as unknown liquidity and fall back to the conservative impact path.
        volumes = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    asset_returns = closes.pct_change().fillna(0.0)
    aligned_weights = (
        target_weights.reindex(closes.index)
        .reindex(columns=closes.columns, fill_value=0.0)
        .ffill()
        .fillna(0.0)
    )
    if weight_limiter is not None:
        aligned_weights = aligned_weights.fillna(0.0).apply(
            lambda row: weight_limiter(row.astype(float)),
            axis=1,
        )
        aligned_weights.index = closes.index
        aligned_weights.columns = closes.columns
        aligned_weights = aligned_weights.fillna(0.0)
    live_weights = aligned_weights.shift(1).fillna(0.0)
    weight_changes = aligned_weights.diff().fillna(aligned_weights).fillna(0.0)
    turnover = weight_changes.abs().sum(axis=1).fillna(0.0)

    dollar_volume = (closes.fillna(0.0) * volumes).fillna(0.0)
    liquidity_reference = (
        dollar_volume.rolling(max(liquidity_lookback_days, 1), min_periods=1).mean().shift(1)
    )
    liquidity_reference = liquidity_reference.where(liquidity_reference > 0.0, dollar_volume)

    gross_returns_values: list[float] = []
    net_returns_values: list[float] = []
    linear_cost_values: list[float] = []
    spread_cost_values: list[float] = []
    market_impact_cost_values: list[float] = []
    max_participation_values: list[float] = []
    portfolio_value = float(initial_capital)

    for current_date in closes.index:
        live_row = live_weights.loc[current_date].astype(float).fillna(0.0)
        returns_row = asset_returns.loc[current_date].astype(float).fillna(0.0)
        trade_weights = weight_changes.loc[current_date].abs().astype(float).fillna(0.0)

        gross_return = float((live_row * returns_row).sum())
        linear_cost = float(trade_weights.sum() * (transaction_cost_bps / 10_000.0))
        spread_cost = float(trade_weights.sum() * (spread_cost_bps / 10_000.0))
        market_impact_cost = 0.0
        max_participation = 0.0

        if market_impact_bps > 0.0 and trade_weights.gt(0.0).any():
            day_liquidity = liquidity_reference.loc[current_date].astype(float).fillna(0.0)
            trade_notional = trade_weights * portfolio_value
            safe_liquidity = day_liquidity.where(day_liquidity > 0.0, trade_notional)
            safe_liquidity = safe_liquidity.where(safe_liquidity > 0.0, 1.0)
            participation = (trade_notional / safe_liquidity).where(trade_weights > 0.0, 0.0).fillna(0.0)
            market_impact_cost = float(
                (
                    trade_weights
                    * (market_impact_bps / 10_000.0)
                    * participation.pow(float(market_impact_exponent))
                ).sum()
            )
            max_participation = float(participation.max())

        total_cost = linear_cost + spread_cost + market_impact_cost
        net_return = gross_return - total_cost

        gross_returns_values.append(gross_return)
        net_returns_values.append(net_return)
        linear_cost_values.append(linear_cost)
        spread_cost_values.append(spread_cost)
        market_impact_cost_values.append(market_impact_cost)
        max_participation_values.append(max_participation)
        portfolio_value *= 1.0 + net_return

    gross_returns = pd.Series(gross_returns_values, index=closes.index, dtype=float)
    net_returns = pd.Series(net_returns_values, index=closes.index, dtype=float)
    linear_costs = pd.Series(linear_cost_values, index=closes.index, dtype=float)
    spread_costs = pd.Series(spread_cost_values, index=closes.index, dtype=float)
    market_impact_costs = pd.Series(market_impact_cost_values, index=closes.index, dtype=float)
    max_participation_rate = pd.Series(max_participation_values, index=closes.index, dtype=float)

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

    summary = enrich_summary_with_execution_details(
        build_summary(net_returns, turnover, benchmark_returns),
        net_returns=net_returns,
        gross_returns=gross_returns,
        linear_costs=linear_costs,
        spread_costs=spread_costs,
        market_impact_costs=market_impact_costs,
        max_participation_rate=max_participation_rate,
    )
    return BacktestResult(
        summary=summary,
        equity_curve=equity_curve,
        daily_returns=net_returns,
        turnover=turnover,
        benchmark_returns=benchmark_returns,
        gross_daily_returns=gross_returns,
        linear_costs=linear_costs,
        spread_costs=spread_costs,
        market_impact_costs=market_impact_costs,
        max_participation_rate=max_participation_rate,
    )
