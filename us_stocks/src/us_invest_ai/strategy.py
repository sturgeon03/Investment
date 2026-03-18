from __future__ import annotations

import numpy as np
import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.signals import attach_llm_scores


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


def _prepare_scored_frame(
    features: pd.DataFrame,
    config: StrategyConfig,
    llm_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    if llm_scores is not None:
        scored = attach_llm_scores(features, llm_scores)
    else:
        scored = features.copy()
        scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
        scored["ticker"] = scored["ticker"].str.upper()
        scored["llm_score"] = 0.0

    scored["llm_score"] = scored["llm_score"].fillna(0.0)
    scored["enough_history"] = scored.groupby("ticker").cumcount() + 1 >= config.min_history_days
    return scored


def generate_target_weights(
    features: pd.DataFrame,
    config: StrategyConfig,
    llm_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = _prepare_scored_frame(features, config, llm_scores)
    tickers = sorted(scored["ticker"].unique().tolist())
    dates = sorted(pd.to_datetime(scored["date"].unique()).tolist())
    target_weights = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    snapshots: list[pd.DataFrame] = []
    for rebalance_date in _rebalance_dates(scored, config.rebalance):
        snapshot = scored.loc[scored["date"] == rebalance_date].copy()
        eligible_universe = (
            snapshot["eligible_universe"].fillna(False)
            if "eligible_universe" in snapshot.columns
            else pd.Series(True, index=snapshot.index)
        )
        base_eligible = snapshot.loc[
            eligible_universe
            & snapshot["enough_history"]
            & snapshot["ret_20"].notna()
            & snapshot["ret_60"].notna()
            & snapshot["vol_20"].notna()
        ].copy()

        if config.trend_filter_mode == "hard":
            eligible = base_eligible.loc[base_eligible["trend_ok"]].copy()
        elif config.trend_filter_mode == "soft":
            eligible = base_eligible.copy()
        else:
            raise ValueError(f"Unsupported trend_filter_mode: {config.trend_filter_mode}")

        target_weights.loc[rebalance_date] = 0.0

        if eligible.empty:
            snapshots.append(snapshot.assign(score=np.nan, selected=False, weight=0.0))
            continue

        eligible["momentum_20_z"] = _cross_sectional_zscore(eligible["ret_20"])
        eligible["momentum_60_z"] = _cross_sectional_zscore(eligible["ret_60"])
        eligible["volatility_z"] = _cross_sectional_zscore(eligible["vol_20"])
        eligible["llm_z"] = _cross_sectional_zscore(eligible["llm_score"])
        eligible["score"] = (
            eligible["momentum_20_z"] * config.momentum_20_weight
            + eligible["momentum_60_z"] * config.momentum_60_weight
            + eligible["volatility_z"] * config.volatility_weight
            + eligible["llm_z"] * config.llm_weight
        )
        if config.trend_filter_mode == "soft":
            eligible["score"] = eligible["score"] - (~eligible["trend_ok"]).astype(float) * config.trend_penalty

        ranked = eligible.sort_values("score", ascending=False).head(config.top_n).copy()
        ranked["weight"] = 1.0 / len(ranked)
        target_weights.loc[rebalance_date, ranked["ticker"]] = ranked["weight"].to_numpy()

        snapshot = snapshot.merge(
            ranked[["ticker", "score", "weight"]],
            on="ticker",
            how="left",
        )
        snapshot["selected"] = snapshot["weight"].fillna(0.0) > 0
        snapshot["weight"] = snapshot["weight"].fillna(0.0)
        snapshots.append(snapshot)

    weights = target_weights.ffill().fillna(0.0)
    ranking_history = pd.concat(snapshots, ignore_index=True)
    return weights, ranking_history
