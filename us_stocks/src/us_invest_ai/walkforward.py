from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.signals import attach_llm_scores


DEFAULT_FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "rel_ret_20",
    "rel_ret_60",
    "vol_20",
    "vol_60",
    "vol_ratio_20_60",
    "trend_flag",
    "market_trend_flag",
    "market_high_vol_flag",
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",
    "sma20_vs_sma50",
    "sma50_vs_sma200",
    "benchmark_ret_20",
    "benchmark_ret_60",
    "benchmark_vol_20",
    "benchmark_drawdown_60",
    "benchmark_price_vs_sma200",
    "benchmark_sma50_vs_sma200",
    "drawdown_20",
    "drawdown_60",
    "dollar_volume_20",
    "universe_age_days",
    "volume_ratio_20",
    "log_dollar_volume_20",
    "range_pct_20",
    "momentum_20_60_gap",
    "benchmark_momentum_gap",
    "relative_momentum_gap",
    "cs_ret_20_z",
    "cs_ret_60_z",
    "cs_rel_ret_20_z",
    "cs_vol_20_z",
    "universe_momentum_rank_pct",
    "universe_vol_rank_pct",
    "market_breadth_trend_share",
    "sector_size",
    "sector_ret_20_gap",
    "sector_rel_ret_20_gap",
    "sector_vol_20_gap",
    "sector_momentum_rank_pct",
    "sector_trend_share",
]


@dataclass(slots=True)
class WalkForwardConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 60
    embargo_days: int | None = None
    min_training_samples: int = 252
    min_validation_samples: int = 120
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    use_llm_feature: bool = False

    def resolved_embargo_days(self) -> int:
        return self.embargo_days if self.embargo_days is not None else self.label_horizon_days

    def resolved_feature_columns(self) -> list[str]:
        feature_columns = list(self.feature_columns)
        if self.use_llm_feature and "llm_score" not in feature_columns:
            feature_columns.append("llm_score")
        return feature_columns


def _rebalance_dates(features: pd.DataFrame, rule: str) -> list[pd.Timestamp]:
    if rule != "monthly":
        raise ValueError(f"Unsupported rebalance rule: {rule}")

    dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()
    periods = dates.dt.to_period("M")
    return dates.groupby(periods).max().tolist()


def prepare_learning_frame(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    walkforward_config: WalkForwardConfig,
    llm_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if llm_scores is not None:
        frame = attach_llm_scores(features, llm_scores)
    else:
        frame = features.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["ticker"] = frame["ticker"].str.upper()
        frame["llm_score"] = 0.0

    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", group_keys=False)

    frame["future_return"] = grouped["close"].shift(-walkforward_config.label_horizon_days) / frame["close"] - 1.0
    frame["label_available_date"] = grouped["date"].shift(-walkforward_config.label_horizon_days)
    frame["trend_flag"] = frame["trend_ok"].fillna(False).astype(float)
    if "price_vs_sma20" not in frame.columns:
        frame["price_vs_sma20"] = frame["close"] / frame["sma_20"] - 1.0
    if "price_vs_sma50" not in frame.columns:
        frame["price_vs_sma50"] = frame["close"] / frame["sma_50"] - 1.0
    if "price_vs_sma200" not in frame.columns:
        frame["price_vs_sma200"] = frame["close"] / frame["sma_200"] - 1.0
    if "sma20_vs_sma50" not in frame.columns:
        frame["sma20_vs_sma50"] = frame["sma_20"] / frame["sma_50"] - 1.0
    if "sma50_vs_sma200" not in frame.columns:
        frame["sma50_vs_sma200"] = frame["sma_50"] / frame["sma_200"] - 1.0
    frame["market_trend_flag"] = frame.get("market_trend_ok", False)
    frame["market_trend_flag"] = pd.Series(frame["market_trend_flag"], index=frame.index).fillna(False).astype(float)
    frame["market_high_vol_flag"] = frame.get("market_high_vol_regime", False)
    frame["market_high_vol_flag"] = pd.Series(frame["market_high_vol_flag"], index=frame.index).fillna(False).astype(float)
    frame["enough_history"] = grouped.cumcount() + 1 >= strategy_config.min_history_days
    frame["eligible_universe"] = frame.get("eligible_universe", True)
    frame["eligible_universe"] = (
        pd.Series(frame["eligible_universe"], index=frame.index).fillna(True).astype(bool)
    )
    frame["llm_score"] = pd.to_numeric(frame["llm_score"], errors="coerce").fillna(0.0)

    feature_columns = walkforward_config.resolved_feature_columns()
    frame["has_model_features"] = frame[feature_columns].notna().all(axis=1)
    frame["has_training_label"] = frame[["future_return", "label_available_date"]].notna().all(axis=1)
    return frame


def select_walkforward_splits(
    frame: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    walkforward_config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    cutoff = rebalance_ts - pd.offsets.BDay(walkforward_config.resolved_embargo_days())

    historical = frame.loc[
        frame["eligible_universe"]
        & frame["enough_history"]
        & frame["has_model_features"]
        & frame["has_training_label"]
        & (pd.to_datetime(frame["label_available_date"]).dt.normalize() < cutoff)
    ].copy()
    if historical.empty:
        return pd.DataFrame(), pd.DataFrame()

    historical["date"] = pd.to_datetime(historical["date"]).dt.normalize()
    unique_dates = sorted(historical["date"].drop_duplicates().tolist())
    if len(unique_dates) <= walkforward_config.validation_window_days:
        return pd.DataFrame(), pd.DataFrame()

    validation_start = unique_dates[-walkforward_config.validation_window_days]
    validation_frame = historical.loc[historical["date"] >= validation_start].copy()
    training_frame = historical.loc[historical["date"] < validation_start].copy()

    if len(training_frame) < walkforward_config.min_training_samples:
        return pd.DataFrame(), pd.DataFrame()
    if len(validation_frame) < walkforward_config.min_validation_samples:
        return pd.DataFrame(), pd.DataFrame()

    return training_frame, validation_frame


def select_live_candidates(
    frame: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    snapshot = frame.loc[pd.to_datetime(frame["date"]).dt.normalize() == rebalance_ts].copy()
    if snapshot.empty:
        return snapshot

    eligible = snapshot.loc[
        snapshot["eligible_universe"]
        & snapshot["enough_history"]
        & snapshot["has_model_features"]
    ].copy()

    if strategy_config.trend_filter_mode == "hard":
        eligible = eligible.loc[eligible["trend_ok"]].copy()
    elif strategy_config.trend_filter_mode != "soft":
        raise ValueError(f"Unsupported trend_filter_mode: {strategy_config.trend_filter_mode}")

    return eligible


def rebalance_dates(frame: pd.DataFrame, rule: str) -> list[pd.Timestamp]:
    return _rebalance_dates(frame, rule)
