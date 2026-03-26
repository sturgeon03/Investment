from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting import binning as hist_binning
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as hist_gradient_boosting

from us_invest_ai.config import StrategyConfig
from us_invest_ai.walkforward import (
    DEFAULT_FEATURE_COLUMNS,
    WalkForwardConfig,
    prepare_learning_frame,
    rebalance_dates,
    select_live_candidates,
    select_walkforward_splits,
)


@dataclass(slots=True)
class TreeModelConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 60
    embargo_days: int | None = None
    min_training_samples: int = 252
    min_validation_samples: int = 120
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    use_llm_feature: bool = False
    learning_rate: float = 0.05
    max_iter: int = 200
    max_depth: int | None = 3
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    random_seed: int = 7


@dataclass(slots=True)
class TreeModel:
    estimator: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    validation_mse: float


@contextmanager
def _limit_fit_threads() -> None:
    thread_env_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    original_values = {name: os.environ.get(name) for name in thread_env_vars}
    with ExitStack() as stack:
        for name in thread_env_vars:
            os.environ[name] = "1"

        def _one_thread(*args, **kwargs) -> int:
            return 1

        stack.enter_context(
            patch.object(hist_gradient_boosting, "_openmp_effective_n_threads", side_effect=_one_thread)
        )
        stack.enter_context(
            patch.object(hist_binning, "_openmp_effective_n_threads", side_effect=_one_thread)
        )
        try:
            yield
        finally:
            for name, value in original_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def fit_tree_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    config: TreeModelConfig,
) -> TreeModel:
    x_train = train_frame[feature_columns].astype(float).to_numpy()
    y_train = train_frame["future_return"].astype(float).to_numpy()
    x_validation = validation_frame[feature_columns].astype(float).to_numpy()
    y_validation = validation_frame["future_return"].astype(float).to_numpy()

    estimator = HistGradientBoostingRegressor(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=True,
        validation_fraction=None,
        n_iter_no_change=15,
        random_state=config.random_seed,
    )
    with _limit_fit_threads():
        estimator.fit(x_train, y_train, X_val=x_validation, y_val=y_validation)
    validation_predictions = estimator.predict(x_validation)
    validation_mse = float(np.mean((validation_predictions - y_validation) ** 2))
    return TreeModel(
        estimator=estimator,
        feature_columns=tuple(feature_columns),
        validation_mse=validation_mse,
    )


def predict_tree_model(model: TreeModel, frame: pd.DataFrame) -> np.ndarray:
    return model.estimator.predict(frame[list(model.feature_columns)].astype(float).to_numpy())


def generate_tree_target_weights(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: TreeModelConfig,
    eval_start: str | pd.Timestamp,
    llm_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    walkforward_config = WalkForwardConfig(
        label_horizon_days=model_config.label_horizon_days,
        validation_window_days=model_config.validation_window_days,
        embargo_days=model_config.embargo_days,
        min_training_samples=model_config.min_training_samples,
        min_validation_samples=model_config.min_validation_samples,
        feature_columns=model_config.feature_columns,
        use_llm_feature=model_config.use_llm_feature,
    )
    frame = prepare_learning_frame(features, strategy_config, walkforward_config, llm_scores)
    eval_start_ts = pd.Timestamp(eval_start).normalize()
    tickers = sorted(frame["ticker"].unique().tolist())
    dates = sorted(pd.to_datetime(frame["date"].unique()).tolist())
    target_weights = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    feature_columns = walkforward_config.resolved_feature_columns()

    snapshots: list[pd.DataFrame] = []
    for rebalance_date in rebalance_dates(frame, strategy_config.rebalance):
        snapshot = frame.loc[frame["date"] == rebalance_date].copy()
        snapshot["predicted_return"] = np.nan
        snapshot["selected"] = False
        snapshot["weight"] = 0.0
        snapshot["train_sample_count"] = 0
        snapshot["validation_sample_count"] = 0
        snapshot["validation_mse"] = np.nan

        target_weights.loc[rebalance_date] = 0.0
        if rebalance_date < eval_start_ts:
            snapshots.append(snapshot)
            continue

        train_frame, validation_frame = select_walkforward_splits(frame, rebalance_date, walkforward_config)
        if train_frame.empty or validation_frame.empty:
            snapshots.append(snapshot)
            continue

        eligible = select_live_candidates(frame, rebalance_date, strategy_config)
        if eligible.empty:
            snapshots.append(snapshot)
            continue

        model = fit_tree_model(train_frame, validation_frame, feature_columns, model_config)
        eligible["predicted_return"] = predict_tree_model(model, eligible)
        if strategy_config.trend_filter_mode == "soft":
            eligible["predicted_return"] = (
                eligible["predicted_return"]
                - (~eligible["trend_ok"]).astype(float) * strategy_config.trend_penalty
            )

        ranked = eligible.sort_values("predicted_return", ascending=False).head(strategy_config.top_n).copy()
        ranked["weight"] = 1.0 / len(ranked)
        ranked["selected"] = True
        ranked["train_sample_count"] = len(train_frame)
        ranked["validation_sample_count"] = len(validation_frame)
        ranked["validation_mse"] = model.validation_mse

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
        for column in [
            "predicted_return",
            "weight",
            "train_sample_count",
            "validation_sample_count",
            "validation_mse",
        ]:
            ranked_column = f"{column}_ranked"
            if ranked_column in snapshot.columns:
                snapshot[column] = snapshot[ranked_column].combine_first(snapshot[column])
                snapshot = snapshot.drop(columns=[ranked_column])
        if "selected_ranked" in snapshot.columns:
            snapshot["selected"] = snapshot["selected_ranked"].fillna(snapshot["selected"]).astype(bool)
            snapshot = snapshot.drop(columns=["selected_ranked"])
        snapshots.append(snapshot)

    weights = target_weights.ffill().fillna(0.0)
    ranking_history = pd.concat(snapshots, ignore_index=True)
    return weights, ranking_history
