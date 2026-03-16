from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.signals import attach_llm_scores


DEFAULT_FEATURE_COLUMNS = [
    "ret_1",
    "ret_20",
    "ret_60",
    "vol_20",
    "trend_flag",
    "price_vs_sma50",
    "sma50_vs_sma200",
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
    frame["price_vs_sma50"] = frame["close"] / frame["sma_50"] - 1.0
    frame["sma50_vs_sma200"] = frame["sma_50"] / frame["sma_200"] - 1.0
    frame["enough_history"] = grouped.cumcount() + 1 >= strategy_config.min_history_days
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
        frame["enough_history"]
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
        snapshot["enough_history"]
        & snapshot["has_model_features"]
    ].copy()

    if strategy_config.trend_filter_mode == "hard":
        eligible = eligible.loc[eligible["trend_ok"]].copy()
    elif strategy_config.trend_filter_mode != "soft":
        raise ValueError(f"Unsupported trend_filter_mode: {strategy_config.trend_filter_mode}")

    return eligible


def rebalance_dates(frame: pd.DataFrame, rule: str) -> list[pd.Timestamp]:
    return _rebalance_dates(frame, rule)
