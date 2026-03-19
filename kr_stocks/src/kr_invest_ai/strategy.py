from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class KRStrategyConfig:
    rebalance: str = "monthly"
    top_n: int = 3
    min_history_days: int = 20
    momentum_20_weight: float = 1.0
    momentum_5_weight: float = 0.35
    volatility_weight: float = -0.25
    earnings_weight: float = 0.30
    capital_event_weight: float = -0.20


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _rebalance_dates(features: pd.DataFrame, rule: str) -> list[pd.Timestamp]:
    if rule != "monthly":
        raise ValueError(f"Unsupported rebalance rule: {rule}")
    dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()
    periods = dates.dt.to_period("M")
    return dates.groupby(periods).max().tolist()


def generate_target_weights(
    features: pd.DataFrame,
    config: KRStrategyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_config = config or KRStrategyConfig()
    scored = features.copy()
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    scored["ticker"] = scored["ticker"].astype(str).str.upper()
    scored["enough_history"] = scored.groupby("ticker").cumcount() + 1 >= strategy_config.min_history_days

    tickers = sorted(scored["ticker"].unique().tolist())
    dates = sorted(pd.to_datetime(scored["date"].unique()).tolist())
    target_weights = pd.DataFrame(0.0, index=dates, columns=tickers, dtype=float)

    snapshots: list[pd.DataFrame] = []
    for rebalance_date in _rebalance_dates(scored, strategy_config.rebalance):
        snapshot = scored.loc[scored["date"] == rebalance_date].copy()
        eligible = snapshot.loc[
            snapshot["enough_history"]
            & snapshot["ret_20"].notna()
            & snapshot["ret_5"].notna()
            & snapshot["vol_20"].notna()
        ].copy()

        if eligible.empty:
            snapshots.append(snapshot.assign(score=pd.NA, selected=False, weight=0.0))
            continue

        eligible["momentum_20_z"] = _cross_sectional_zscore(eligible["ret_20"])
        eligible["momentum_5_z"] = _cross_sectional_zscore(eligible["ret_5"])
        eligible["volatility_z"] = _cross_sectional_zscore(eligible["vol_20"])
        eligible["earnings_z"] = _cross_sectional_zscore(eligible["earnings_filing_count_60"])
        eligible["capital_event_z"] = _cross_sectional_zscore(eligible["capital_event_count_60"])
        eligible["score"] = (
            eligible["momentum_20_z"] * strategy_config.momentum_20_weight
            + eligible["momentum_5_z"] * strategy_config.momentum_5_weight
            + eligible["volatility_z"] * strategy_config.volatility_weight
            + eligible["earnings_z"] * strategy_config.earnings_weight
            + eligible["capital_event_z"] * strategy_config.capital_event_weight
        )

        ranked = eligible.sort_values(["score", "avg_dollar_volume_20"], ascending=[False, False]).head(
            strategy_config.top_n
        )
        ranked = ranked.copy()
        ranked["weight"] = 1.0 / len(ranked)
        target_weights.loc[rebalance_date, ranked["ticker"]] = ranked["weight"].to_numpy()

        snapshot = snapshot.merge(ranked[["ticker", "score", "weight"]], on="ticker", how="left")
        snapshot["selected"] = snapshot["weight"].fillna(0.0) > 0
        snapshot["weight"] = snapshot["weight"].fillna(0.0)
        snapshots.append(snapshot)

    return target_weights.ffill().fillna(0.0), pd.concat(snapshots, ignore_index=True)
