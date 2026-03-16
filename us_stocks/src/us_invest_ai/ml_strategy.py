from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
class MLModelConfig:
    label_horizon_days: int = 20
    ridge_alpha: float = 8.0
    min_training_samples: int = 252
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    use_llm_feature: bool = False


@dataclass(slots=True)
class RidgeModel:
    intercept: float
    coefficients: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    feature_columns: tuple[str, ...]


def _rebalance_dates(features: pd.DataFrame, rule: str) -> list[pd.Timestamp]:
    if rule != "monthly":
        raise ValueError(f"Unsupported rebalance rule: {rule}")

    dates = pd.Series(pd.to_datetime(features["date"].unique())).sort_values()
    periods = dates.dt.to_period("M")
    return dates.groupby(periods).max().tolist()


def _prepare_ml_frame(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: MLModelConfig,
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

    frame["future_return"] = grouped["close"].shift(-model_config.label_horizon_days) / frame["close"] - 1.0
    frame["label_available_date"] = grouped["date"].shift(-model_config.label_horizon_days)
    frame["trend_flag"] = frame["trend_ok"].fillna(False).astype(float)
    frame["price_vs_sma50"] = frame["close"] / frame["sma_50"] - 1.0
    frame["sma50_vs_sma200"] = frame["sma_50"] / frame["sma_200"] - 1.0
    frame["enough_history"] = grouped.cumcount() + 1 >= strategy_config.min_history_days
    frame["llm_score"] = pd.to_numeric(frame["llm_score"], errors="coerce").fillna(0.0)

    feature_columns = list(model_config.feature_columns)
    if model_config.use_llm_feature and "llm_score" not in feature_columns:
        feature_columns.append("llm_score")

    frame["has_model_features"] = frame[feature_columns].notna().all(axis=1)
    frame["has_training_label"] = frame[["future_return", "label_available_date"]].notna().all(axis=1)
    return frame


def fit_ridge_model(train_frame: pd.DataFrame, feature_columns: list[str], alpha: float) -> RidgeModel:
    design = train_frame[feature_columns].astype(float).to_numpy()
    target = train_frame["future_return"].astype(float).to_numpy()
    means = design.mean(axis=0)
    stds = design.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    scaled = (design - means) / stds

    augmented = np.column_stack([np.ones(len(scaled)), scaled])
    penalty = alpha * np.eye(augmented.shape[1])
    penalty[0, 0] = 0.0
    lhs = augmented.T @ augmented + penalty
    rhs = augmented.T @ target
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs

    return RidgeModel(
        intercept=float(weights[0]),
        coefficients=weights[1:],
        feature_means=means,
        feature_stds=stds,
        feature_columns=tuple(feature_columns),
    )


def predict_ridge_model(model: RidgeModel, frame: pd.DataFrame) -> np.ndarray:
    design = frame[list(model.feature_columns)].astype(float).to_numpy()
    scaled = (design - model.feature_means) / model.feature_stds
    return model.intercept + scaled @ model.coefficients


def generate_ml_target_weights(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: MLModelConfig,
    eval_start: str | pd.Timestamp,
    llm_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _prepare_ml_frame(features, strategy_config, model_config, llm_scores)
    eval_start_ts = pd.Timestamp(eval_start).normalize()
    tickers = sorted(frame["ticker"].unique().tolist())
    dates = sorted(pd.to_datetime(frame["date"].unique()).tolist())
    target_weights = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    feature_columns = list(model_config.feature_columns)
    if model_config.use_llm_feature and "llm_score" not in feature_columns:
        feature_columns.append("llm_score")

    snapshots: list[pd.DataFrame] = []
    for rebalance_date in _rebalance_dates(frame, strategy_config.rebalance):
        snapshot = frame.loc[frame["date"] == rebalance_date].copy()
        snapshot["predicted_return"] = np.nan
        snapshot["selected"] = False
        snapshot["weight"] = 0.0
        snapshot["train_sample_count"] = 0

        target_weights.loc[rebalance_date] = 0.0
        if rebalance_date < eval_start_ts:
            snapshots.append(snapshot)
            continue

        train_frame = frame.loc[
            frame["enough_history"]
            & frame["has_model_features"]
            & frame["has_training_label"]
            & (pd.to_datetime(frame["label_available_date"]).dt.normalize() < rebalance_date)
        ].copy()
        if len(train_frame) < model_config.min_training_samples:
            snapshots.append(snapshot)
            continue

        eligible = snapshot.loc[
            snapshot["enough_history"]
            & snapshot["has_model_features"]
        ].copy()
        if strategy_config.trend_filter_mode == "hard":
            eligible = eligible.loc[eligible["trend_ok"]].copy()
        elif strategy_config.trend_filter_mode != "soft":
            raise ValueError(f"Unsupported trend_filter_mode: {strategy_config.trend_filter_mode}")

        if eligible.empty:
            snapshots.append(snapshot)
            continue

        model = fit_ridge_model(train_frame, feature_columns, model_config.ridge_alpha)
        eligible["predicted_return"] = predict_ridge_model(model, eligible)
        if strategy_config.trend_filter_mode == "soft":
            eligible["predicted_return"] = (
                eligible["predicted_return"]
                - (~eligible["trend_ok"]).astype(float) * strategy_config.trend_penalty
            )

        ranked = eligible.sort_values("predicted_return", ascending=False).head(strategy_config.top_n).copy()
        ranked["weight"] = 1.0 / len(ranked)
        ranked["selected"] = True
        ranked["train_sample_count"] = len(train_frame)

        target_weights.loc[rebalance_date, ranked["ticker"]] = ranked["weight"].to_numpy()

        snapshot = snapshot.merge(
            ranked[["ticker", "predicted_return", "selected", "weight", "train_sample_count"]],
            on="ticker",
            how="left",
            suffixes=("", "_ranked"),
        )
        for column in ["predicted_return", "weight"]:
            ranked_column = f"{column}_ranked"
            if ranked_column in snapshot.columns:
                snapshot[column] = snapshot[ranked_column].combine_first(snapshot[column])
                snapshot = snapshot.drop(columns=[ranked_column])
        if "selected_ranked" in snapshot.columns:
            snapshot["selected"] = snapshot["selected_ranked"].fillna(snapshot["selected"]).astype(bool)
            snapshot = snapshot.drop(columns=["selected_ranked"])
        if "train_sample_count_ranked" in snapshot.columns:
            snapshot["train_sample_count"] = (
                snapshot["train_sample_count_ranked"].fillna(snapshot["train_sample_count"]).astype(int)
            )
            snapshot = snapshot.drop(columns=["train_sample_count_ranked"])
        snapshots.append(snapshot)

    weights = target_weights.ffill().fillna(0.0)
    ranking_history = pd.concat(snapshots, ignore_index=True)
    return weights, ranking_history
