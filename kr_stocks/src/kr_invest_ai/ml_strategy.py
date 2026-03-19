from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kr_invest_ai.strategy import KRStrategyConfig
from kr_invest_ai.walkforward import (
    DEFAULT_FEATURE_COLUMNS,
    KRWalkForwardConfig,
    prepare_learning_frame,
    rebalance_dates,
    select_live_candidates,
    select_walkforward_splits,
)


@dataclass(frozen=True, slots=True)
class KRMLModelConfig:
    label_horizon_days: int = 20
    ridge_alpha: float = 8.0
    min_training_samples: int = 60
    validation_window_days: int = 40
    embargo_days: int | None = None
    min_validation_samples: int = 20
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS


@dataclass(frozen=True, slots=True)
class KRRidgeModel:
    intercept: float
    coefficients: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    feature_columns: tuple[str, ...]

def fit_ridge_model(train_frame: pd.DataFrame, feature_columns: list[str], alpha: float) -> KRRidgeModel:
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
    return KRRidgeModel(
        intercept=float(weights[0]),
        coefficients=weights[1:],
        feature_means=means,
        feature_stds=stds,
        feature_columns=tuple(feature_columns),
    )


def predict_ridge_model(model: KRRidgeModel, frame: pd.DataFrame) -> np.ndarray:
    design = frame[list(model.feature_columns)].astype(float).to_numpy()
    scaled = (design - model.feature_means) / model.feature_stds
    return model.intercept + scaled @ model.coefficients


def generate_ridge_target_weights(
    features: pd.DataFrame,
    strategy_config: KRStrategyConfig | None = None,
    model_config: KRMLModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_strategy = strategy_config or KRStrategyConfig()
    resolved_model = model_config or KRMLModelConfig()
    frame = prepare_learning_frame(
        features,
        resolved_strategy,
        KRWalkForwardConfig(
            label_horizon_days=resolved_model.label_horizon_days,
            validation_window_days=resolved_model.validation_window_days,
            embargo_days=resolved_model.embargo_days,
            min_training_samples=resolved_model.min_training_samples,
            min_validation_samples=resolved_model.min_validation_samples,
            feature_columns=resolved_model.feature_columns,
        ),
    )

    tickers = sorted(frame["ticker"].unique().tolist())
    dates = sorted(pd.to_datetime(frame["date"].unique()).tolist())
    target_weights = pd.DataFrame(0.0, index=dates, columns=tickers, dtype=float)
    snapshots: list[pd.DataFrame] = []
    feature_columns = list(resolved_model.feature_columns)

    for rebalance_date in rebalance_dates(frame, resolved_strategy.rebalance):
        snapshot = frame.loc[frame["date"] == rebalance_date].copy()
        snapshot["predicted_return"] = np.nan
        snapshot["selected"] = False
        snapshot["weight"] = 0.0
        snapshot["train_sample_count"] = 0

        train_frame = frame.loc[
            frame["enough_history"]
            & frame["has_model_features"]
            & frame["has_training_label"]
            & (pd.to_datetime(frame["label_available_date"]).dt.normalize() < rebalance_date)
        ].copy()
        if len(train_frame) < resolved_model.min_training_samples:
            snapshots.append(snapshot)
            continue

        eligible = snapshot.loc[snapshot["enough_history"] & snapshot["has_model_features"]].copy()
        if eligible.empty:
            snapshots.append(snapshot)
            continue

        model = fit_ridge_model(train_frame, feature_columns, resolved_model.ridge_alpha)
        eligible["predicted_return"] = predict_ridge_model(model, eligible)
        ranked = eligible.sort_values("predicted_return", ascending=False).head(resolved_strategy.top_n).copy()
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
        for column in ("predicted_return", "weight", "train_sample_count"):
            ranked_column = f"{column}_ranked"
            if ranked_column in snapshot.columns:
                snapshot[column] = snapshot[ranked_column].combine_first(snapshot[column])
                snapshot = snapshot.drop(columns=[ranked_column])
        if "selected_ranked" in snapshot.columns:
            snapshot["selected"] = snapshot["selected_ranked"].fillna(False).astype(bool)
            snapshot = snapshot.drop(columns=["selected_ranked"])
        snapshots.append(snapshot)

    return target_weights.ffill().fillna(0.0), pd.concat(snapshots, ignore_index=True)


def generate_ridge_walkforward_target_weights(
    features: pd.DataFrame,
    strategy_config: KRStrategyConfig | None = None,
    model_config: KRMLModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_strategy = strategy_config or KRStrategyConfig()
    resolved_model = model_config or KRMLModelConfig()
    walkforward_config = KRWalkForwardConfig(
        label_horizon_days=resolved_model.label_horizon_days,
        validation_window_days=resolved_model.validation_window_days,
        embargo_days=resolved_model.embargo_days,
        min_training_samples=resolved_model.min_training_samples,
        min_validation_samples=resolved_model.min_validation_samples,
        feature_columns=resolved_model.feature_columns,
    )
    frame = prepare_learning_frame(features, resolved_strategy, walkforward_config)

    tickers = sorted(frame["ticker"].unique().tolist())
    dates = sorted(pd.to_datetime(frame["date"].unique()).tolist())
    target_weights = pd.DataFrame(0.0, index=dates, columns=tickers, dtype=float)
    snapshots: list[pd.DataFrame] = []
    feature_columns = list(resolved_model.feature_columns)

    for rebalance_date in rebalance_dates(frame, resolved_strategy.rebalance):
        snapshot = frame.loc[frame["date"] == rebalance_date].copy()
        snapshot["predicted_return"] = np.nan
        snapshot["selected"] = False
        snapshot["weight"] = 0.0
        snapshot["train_sample_count"] = 0
        snapshot["validation_sample_count"] = 0
        snapshot["validation_mse"] = np.nan

        train_frame, validation_frame = select_walkforward_splits(frame, rebalance_date, walkforward_config)
        if train_frame.empty or validation_frame.empty:
            snapshots.append(snapshot)
            continue

        eligible = select_live_candidates(frame, rebalance_date)
        if eligible.empty:
            snapshots.append(snapshot)
            continue

        model = fit_ridge_model(train_frame, feature_columns, resolved_model.ridge_alpha)
        validation_predictions = predict_ridge_model(model, validation_frame)
        validation_mse = float(np.mean((validation_predictions - validation_frame["future_return"].to_numpy()) ** 2))

        eligible["predicted_return"] = predict_ridge_model(model, eligible)
        ranked = eligible.sort_values("predicted_return", ascending=False).head(resolved_strategy.top_n).copy()
        ranked["weight"] = 1.0 / len(ranked)
        ranked["selected"] = True
        ranked["train_sample_count"] = len(train_frame)
        ranked["validation_sample_count"] = len(validation_frame)
        ranked["validation_mse"] = validation_mse
        target_weights.loc[rebalance_date, ranked["ticker"]] = ranked["weight"].to_numpy()

        snapshot = snapshot.merge(
            ranked[
                [
                    "ticker",
                    "predicted_return",
                    "selected",
                    "weight",
                    "train_sample_count",
                    "validation_sample_count",
                    "validation_mse",
                ]
            ],
            on="ticker",
            how="left",
            suffixes=("", "_ranked"),
        )
        for column in (
            "predicted_return",
            "weight",
            "train_sample_count",
            "validation_sample_count",
            "validation_mse",
        ):
            ranked_column = f"{column}_ranked"
            if ranked_column in snapshot.columns:
                snapshot[column] = snapshot[ranked_column].combine_first(snapshot[column])
                snapshot = snapshot.drop(columns=[ranked_column])
        if "selected_ranked" in snapshot.columns:
            snapshot["selected"] = snapshot["selected_ranked"].fillna(False).astype(bool)
            snapshot = snapshot.drop(columns=["selected_ranked"])
        snapshots.append(snapshot)

    return target_weights.ffill().fillna(0.0), pd.concat(snapshots, ignore_index=True)
