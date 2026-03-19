from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kr_invest_ai.strategy import KRStrategyConfig


DEFAULT_FEATURE_COLUMNS = (
    "ret_1",
    "ret_5",
    "ret_20",
    "rel_ret_5",
    "rel_ret_20",
    "benchmark_ret_20",
    "avg_dollar_volume_20",
    "vol_20",
    "benchmark_vol_20",
    "range_20",
    "market_trend_ok",
    "filing_count_20",
    "earnings_filing_count_60",
    "capital_event_count_60",
    "days_since_last_filing",
)


@dataclass(frozen=True, slots=True)
class KRWalkForwardConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 40
    embargo_days: int | None = None
    min_training_samples: int = 60
    min_validation_samples: int = 20
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS

    def resolved_embargo_days(self) -> int:
        return self.embargo_days if self.embargo_days is not None else self.label_horizon_days


def rebalance_dates(features: pd.DataFrame, rule: str) -> list[pd.Timestamp]:
    if rule != "monthly":
        raise ValueError(f"Unsupported rebalance rule: {rule}")
    dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()
    periods = dates.dt.to_period("M")
    return dates.groupby(periods).max().tolist()


def prepare_learning_frame(
    features: pd.DataFrame,
    strategy_config: KRStrategyConfig,
    walkforward_config: KRWalkForwardConfig,
) -> pd.DataFrame:
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = frame.groupby("ticker", group_keys=False)
    frame["future_return"] = grouped["close"].shift(-walkforward_config.label_horizon_days) / frame["close"] - 1.0
    frame["label_available_date"] = grouped["date"].shift(-walkforward_config.label_horizon_days)
    frame["enough_history"] = grouped.cumcount() + 1 >= strategy_config.min_history_days
    frame["has_model_features"] = frame[list(walkforward_config.feature_columns)].notna().all(axis=1)
    frame["has_training_label"] = frame[["future_return", "label_available_date"]].notna().all(axis=1)
    frame["eligible_universe"] = True
    return frame


def select_walkforward_splits(
    frame: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    walkforward_config: KRWalkForwardConfig,
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
) -> pd.DataFrame:
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    snapshot = frame.loc[pd.to_datetime(frame["date"]).dt.normalize() == rebalance_ts].copy()
    if snapshot.empty:
        return snapshot
    return snapshot.loc[snapshot["eligible_universe"] & snapshot["enough_history"] & snapshot["has_model_features"]].copy()
