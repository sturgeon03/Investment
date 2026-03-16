from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_invest_ai.config import RiskConfig


def apply_risk_limits(target_weights: pd.Series, risk: RiskConfig) -> pd.Series:
    adjusted = target_weights.fillna(0.0).clip(lower=0.0).astype(float).copy()
    adjusted = adjusted.clip(upper=risk.max_position_weight)
    investable_weight = max(0.0, 1.0 - risk.cash_buffer)
    current_sum = float(adjusted.sum())
    if current_sum > investable_weight and current_sum > 0:
        adjusted = adjusted * (investable_weight / current_sum)
    return adjusted


def load_current_positions(path: str | Path) -> pd.DataFrame:
    position_path = Path(path)
    if not position_path.exists():
        return pd.DataFrame(columns=["ticker", "shares"])

    positions = pd.read_csv(position_path)
    if positions.empty:
        return pd.DataFrame(columns=["ticker", "shares"])
    required = {"ticker", "shares"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"Current positions file is missing columns: {sorted(missing)}")
    positions = positions.copy()
    positions["ticker"] = positions["ticker"].str.upper()
    positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce").fillna(0.0)
    return positions[["ticker", "shares"]]


def _target_shares(target_notional: float, close: float, allow_fractional_shares: bool) -> float:
    if close <= 0:
        return 0.0
    raw_shares = target_notional / close
    if allow_fractional_shares:
        return round(raw_shares, 6)
    return float(np.floor(raw_shares))


def latest_prices_by_ticker(
    prices: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "close"])

    latest = prices.copy()
    latest["date"] = pd.to_datetime(latest["date"]).dt.normalize()
    latest["ticker"] = latest["ticker"].str.upper()
    if as_of_date is not None:
        latest = latest.loc[latest["date"] <= pd.Timestamp(as_of_date).normalize()].copy()
    if latest.empty:
        return pd.DataFrame(columns=["ticker", "close"])

    latest = latest.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    return latest[["ticker", "close"]].reset_index(drop=True)


def build_target_portfolio(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    risk: RiskConfig,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    weights = target_weights.copy()
    weights.index = pd.to_datetime(weights.index).normalize()
    if as_of_date is None:
        portfolio_date = weights.index.max()
    else:
        portfolio_date = pd.Timestamp(as_of_date).normalize()
        available_dates = weights.index[weights.index <= portfolio_date]
        if len(available_dates) == 0:
            raise ValueError(f"No target weights available on or before {portfolio_date.date().isoformat()}.")
        portfolio_date = available_dates.max()

    latest_weights = weights.loc[portfolio_date]
    latest_weights = latest_weights[latest_weights > 0]
    limited_weights = apply_risk_limits(latest_weights, risk)

    close_frame = (
        prices.assign(date=pd.to_datetime(prices["date"]).dt.normalize())
        .loc[lambda frame: frame["date"] == portfolio_date, ["ticker", "close"]]
        .copy()
    )
    close_frame["ticker"] = close_frame["ticker"].str.upper()

    portfolio = pd.DataFrame(
        {
            "ticker": limited_weights.index,
            "raw_weight": latest_weights.reindex(limited_weights.index).fillna(0.0).to_numpy(),
            "target_weight": limited_weights.to_numpy(),
        }
    )
    portfolio = portfolio.merge(close_frame, on="ticker", how="left")
    if portfolio["close"].isna().any():
        missing = portfolio.loc[portfolio["close"].isna(), "ticker"].tolist()
        raise ValueError(f"Missing close price for target tickers on {portfolio_date.date().isoformat()}: {missing}")
    portfolio["target_notional"] = portfolio["target_weight"] * risk.capital_base
    portfolio["target_shares"] = portfolio.apply(
        lambda row: _target_shares(
            target_notional=float(row["target_notional"]),
            close=float(row["close"]),
            allow_fractional_shares=risk.allow_fractional_shares,
        ),
        axis=1,
    )
    portfolio["target_notional_after_rounding"] = portfolio["target_shares"] * portfolio["close"]
    portfolio.insert(0, "date", portfolio_date)
    return portfolio.sort_values(["target_weight", "ticker"], ascending=[False, True]).reset_index(drop=True)


def build_rebalance_orders(
    target_portfolio: pd.DataFrame,
    current_positions: pd.DataFrame,
    risk: RiskConfig,
    latest_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    current = current_positions.copy()
    target = target_portfolio.copy()
    target_date = pd.to_datetime(target.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce").max()

    merged = current.merge(
        target[["ticker", "date", "close", "target_weight", "target_shares", "target_notional_after_rounding"]],
        on="ticker",
        how="outer",
    )
    if latest_prices is not None and not latest_prices.empty:
        price_frame = latest_prices[["ticker", "close"]].copy()
        price_frame["ticker"] = price_frame["ticker"].str.upper()
        price_frame = price_frame.rename(columns={"close": "latest_close"})
        merged = merged.merge(price_frame, on="ticker", how="left")
        merged["close"] = merged["close"].fillna(merged["latest_close"])
        merged = merged.drop(columns=["latest_close"])
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    if pd.notna(target_date):
        merged["date"] = merged["date"].fillna(target_date)
    merged["shares"] = merged["shares"].fillna(0.0)
    merged["target_shares"] = merged["target_shares"].fillna(0.0)
    merged["close"] = merged["close"].fillna(0.0)
    merged["target_weight"] = merged["target_weight"].fillna(0.0)
    merged["target_notional_after_rounding"] = merged["target_notional_after_rounding"].fillna(0.0)
    merged["order_shares"] = merged["target_shares"] - merged["shares"]
    merged["trade_notional"] = (merged["order_shares"].abs() * merged["close"]).fillna(0.0)
    merged["side"] = np.where(merged["order_shares"] > 0, "BUY", np.where(merged["order_shares"] < 0, "SELL", "HOLD"))
    orders = merged.loc[
        (merged["side"] != "HOLD") & (merged["trade_notional"] >= risk.min_trade_notional)
    ].copy()
    return orders[
        [
            "date",
            "ticker",
            "side",
            "order_shares",
            "close",
            "trade_notional",
            "target_weight",
            "target_notional_after_rounding",
        ]
    ].reset_index(drop=True)


def build_next_positions(target_portfolio: pd.DataFrame) -> pd.DataFrame:
    if target_portfolio.empty:
        return pd.DataFrame(columns=["ticker", "shares"])
    next_positions = target_portfolio[["ticker", "target_shares"]].copy()
    next_positions = next_positions.rename(columns={"target_shares": "shares"})
    return next_positions.reset_index(drop=True)


def save_table(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
