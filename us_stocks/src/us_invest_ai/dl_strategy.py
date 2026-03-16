from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
class MLPModelConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 60
    embargo_days: int | None = None
    min_training_samples: int = 252
    min_validation_samples: int = 120
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    use_llm_feature: bool = False
    hidden_dim: int = 32
    learning_rate: float = 0.01
    max_epochs: int = 250
    batch_size: int = 128
    weight_decay: float = 1e-4
    patience: int = 25
    random_seed: int = 7


@dataclass(slots=True)
class MLPModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    feature_means: np.ndarray
    feature_stds: np.ndarray
    target_mean: float
    target_std: float
    feature_columns: tuple[str, ...]


def _standardize_inputs(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = frame[feature_columns].astype(float).to_numpy()
    means = design.mean(axis=0)
    stds = design.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    scaled = (design - means) / stds
    return scaled, means, stds


def _forward(x_batch: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: float) -> tuple[np.ndarray, np.ndarray]:
    hidden_linear = x_batch @ w1 + b1
    hidden = np.tanh(hidden_linear)
    output = hidden @ w2 + b2
    return hidden, output


def fit_mlp_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    config: MLPModelConfig,
) -> MLPModel:
    x_train, feature_means, feature_stds = _standardize_inputs(train_frame, feature_columns)
    x_validation = (validation_frame[feature_columns].astype(float).to_numpy() - feature_means) / feature_stds

    y_train = train_frame["future_return"].astype(float).to_numpy()
    y_validation = validation_frame["future_return"].astype(float).to_numpy()
    target_mean = float(y_train.mean())
    target_std = float(y_train.std(ddof=0))
    if target_std == 0:
        target_std = 1.0

    y_train_scaled = ((y_train - target_mean) / target_std).reshape(-1, 1)
    y_validation_scaled = ((y_validation - target_mean) / target_std).reshape(-1, 1)

    rng = np.random.default_rng(config.random_seed)
    input_dim = x_train.shape[1]
    hidden_dim = config.hidden_dim
    w1 = rng.normal(0.0, np.sqrt(2.0 / max(input_dim, 1)), size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=float)
    w2 = rng.normal(0.0, np.sqrt(1.0 / max(hidden_dim, 1)), size=(hidden_dim, 1))
    b2 = 0.0

    adam_state: dict[str, np.ndarray | float] = {
        "mw1": np.zeros_like(w1),
        "vw1": np.zeros_like(w1),
        "mb1": np.zeros_like(b1),
        "vb1": np.zeros_like(b1),
        "mw2": np.zeros_like(w2),
        "vw2": np.zeros_like(w2),
        "mb2": 0.0,
        "vb2": 0.0,
    }
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    best_validation_loss = float("inf")
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    epochs_without_improvement = 0
    global_step = 0

    for _ in range(config.max_epochs):
        permutation = rng.permutation(len(x_train))
        x_epoch = x_train[permutation]
        y_epoch = y_train_scaled[permutation]

        for start in range(0, len(x_epoch), config.batch_size):
            end = min(start + config.batch_size, len(x_epoch))
            x_batch = x_epoch[start:end]
            y_batch = y_epoch[start:end]
            global_step += 1

            hidden, predictions = _forward(x_batch, w1, b1, w2, b2)
            errors = predictions - y_batch
            batch_size = max(len(x_batch), 1)
            output_grad = (2.0 / batch_size) * errors

            grad_w2 = hidden.T @ output_grad + config.weight_decay * w2
            grad_b2 = float(output_grad.sum())
            hidden_grad = (output_grad @ w2.T) * (1.0 - hidden ** 2)
            grad_w1 = x_batch.T @ hidden_grad + config.weight_decay * w1
            grad_b1 = hidden_grad.sum(axis=0)

            for name, grad in (
                ("w1", grad_w1),
                ("b1", grad_b1),
                ("w2", grad_w2),
                ("b2", grad_b2),
            ):
                m_key = f"m{name}"
                v_key = f"v{name}"
                adam_state[m_key] = beta1 * adam_state[m_key] + (1.0 - beta1) * grad
                adam_state[v_key] = beta2 * adam_state[v_key] + (1.0 - beta2) * (grad ** 2)
                m_hat = adam_state[m_key] / (1.0 - beta1 ** global_step)
                v_hat = adam_state[v_key] / (1.0 - beta2 ** global_step)
                update = config.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                if name == "w1":
                    w1 = w1 - update
                elif name == "b1":
                    b1 = b1 - update
                elif name == "w2":
                    w2 = w2 - update
                else:
                    b2 = float(b2 - update)

        _, validation_predictions = _forward(x_validation, w1, b1, w2, b2)
        validation_loss = float(np.mean((validation_predictions - y_validation_scaled) ** 2))
        if validation_loss + 1e-9 < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = (w1.copy(), b1.copy(), w2.copy(), float(b2))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        best_state = (w1.copy(), b1.copy(), w2.copy(), float(b2))

    return MLPModel(
        w1=best_state[0],
        b1=best_state[1],
        w2=best_state[2],
        b2=best_state[3],
        feature_means=feature_means,
        feature_stds=feature_stds,
        target_mean=target_mean,
        target_std=target_std,
        feature_columns=tuple(feature_columns),
    )


def predict_mlp_model(model: MLPModel, frame: pd.DataFrame) -> np.ndarray:
    design = frame[list(model.feature_columns)].astype(float).to_numpy()
    scaled = (design - model.feature_means) / model.feature_stds
    _, predictions = _forward(scaled, model.w1, model.b1, model.w2, model.b2)
    return predictions.ravel() * model.target_std + model.target_mean


def generate_mlp_target_weights(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: MLPModelConfig,
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

        model = fit_mlp_model(train_frame, validation_frame, feature_columns, model_config)
        eligible["predicted_return"] = predict_mlp_model(model, eligible)
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
