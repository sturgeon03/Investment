from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_invest_ai.config import StrategyConfig
from us_invest_ai.tcn_strategy import DEFAULT_SEQUENCE_FEATURE_COLUMNS, prepare_sequence_dataset
from us_invest_ai.walkforward import (
    WalkForwardConfig,
    rebalance_dates,
    select_live_candidates,
    select_walkforward_splits,
)


@dataclass(slots=True)
class TransformerModelConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 60
    embargo_days: int | None = None
    min_training_samples: int = 252
    min_validation_samples: int = 120
    feature_columns: tuple[str, ...] = DEFAULT_SEQUENCE_FEATURE_COLUMNS
    use_llm_feature: bool = False
    lookback_window: int = 20
    training_lookback_days: int | None = 252
    model_dim: int = 4
    learning_rate: float = 0.005
    max_epochs: int = 12
    batch_size: int = 1024
    weight_decay: float = 1e-4
    patience: int = 3
    random_seed: int = 7


@dataclass(slots=True)
class TransformerModel:
    input_weights: np.ndarray
    input_bias: np.ndarray
    positional_embedding: np.ndarray
    query_weights: np.ndarray
    key_weights: np.ndarray
    value_weights: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    feature_means: np.ndarray
    feature_stds: np.ndarray
    target_mean: float
    target_std: float
    feature_columns: tuple[str, ...]
    lookback_window: int
    model_dim: int
    validation_mse: float


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _standardize_sequences(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = sequences.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = sequences.std(axis=(0, 1), ddof=0, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1.0
    scaled = (sequences - means) / stds
    return scaled.astype(np.float32), means, stds


def _forward(
    x_batch: np.ndarray,
    input_weights: np.ndarray,
    input_bias: np.ndarray,
    positional_embedding: np.ndarray,
    query_weights: np.ndarray,
    key_weights: np.ndarray,
    value_weights: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected = np.einsum("btf,fd->btd", x_batch, input_weights) + input_bias
    encoded = projected + positional_embedding[None, :, :]
    queries = np.einsum("btd,dh->bth", encoded, query_weights)
    keys = np.einsum("btd,dh->bth", encoded, key_weights)
    values = np.einsum("btd,dh->bth", encoded, value_weights)
    scale = np.sqrt(max(encoded.shape[2], 1))
    scores = np.matmul(queries, np.swapaxes(keys, 1, 2)) / scale
    attention = _softmax(scores)
    context = np.matmul(attention, values)
    hidden = np.tanh(context)
    pooled = hidden.mean(axis=1)
    predictions = pooled @ output_weights + output_bias
    return encoded, queries, keys, values, attention, hidden, predictions


def fit_transformer_model(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    validation_sequences: np.ndarray,
    validation_targets: np.ndarray,
    feature_columns: tuple[str, ...],
    config: TransformerModelConfig,
) -> TransformerModel:
    x_train, feature_means, feature_stds = _standardize_sequences(train_sequences.astype(np.float32))
    x_validation = ((validation_sequences.astype(np.float32) - feature_means) / feature_stds).astype(np.float32)

    y_train = train_targets.astype(np.float32)
    y_validation = validation_targets.astype(np.float32)
    target_mean = float(y_train.mean())
    target_std = float(y_train.std(ddof=0))
    if target_std == 0:
        target_std = 1.0

    y_train_scaled = ((y_train - target_mean) / target_std).reshape(-1, 1).astype(np.float32)
    y_validation_scaled = ((y_validation - target_mean) / target_std).reshape(-1, 1).astype(np.float32)

    rng = np.random.default_rng(config.random_seed)
    input_dim = x_train.shape[2]
    model_dim = config.model_dim
    sequence_length = x_train.shape[1]

    input_weights = rng.normal(
        0.0,
        np.sqrt(2.0 / max(input_dim, 1)),
        size=(input_dim, model_dim),
    ).astype(np.float32)
    input_bias = np.zeros(model_dim, dtype=np.float32)
    positional_embedding = rng.normal(
        0.0,
        0.02,
        size=(sequence_length, model_dim),
    ).astype(np.float32)
    query_weights = rng.normal(
        0.0,
        np.sqrt(1.0 / max(model_dim, 1)),
        size=(model_dim, model_dim),
    ).astype(np.float32)
    key_weights = rng.normal(
        0.0,
        np.sqrt(1.0 / max(model_dim, 1)),
        size=(model_dim, model_dim),
    ).astype(np.float32)
    value_weights = rng.normal(
        0.0,
        np.sqrt(1.0 / max(model_dim, 1)),
        size=(model_dim, model_dim),
    ).astype(np.float32)
    output_weights = rng.normal(
        0.0,
        np.sqrt(1.0 / max(model_dim, 1)),
        size=(model_dim, 1),
    ).astype(np.float32)
    output_bias = 0.0

    adam_state: dict[str, np.ndarray | float] = {
        "m_input_weights": np.zeros_like(input_weights),
        "v_input_weights": np.zeros_like(input_weights),
        "m_input_bias": np.zeros_like(input_bias),
        "v_input_bias": np.zeros_like(input_bias),
        "m_positional_embedding": np.zeros_like(positional_embedding),
        "v_positional_embedding": np.zeros_like(positional_embedding),
        "m_query_weights": np.zeros_like(query_weights),
        "v_query_weights": np.zeros_like(query_weights),
        "m_key_weights": np.zeros_like(key_weights),
        "v_key_weights": np.zeros_like(key_weights),
        "m_value_weights": np.zeros_like(value_weights),
        "v_value_weights": np.zeros_like(value_weights),
        "m_output_weights": np.zeros_like(output_weights),
        "v_output_weights": np.zeros_like(output_weights),
        "m_output_bias": 0.0,
        "v_output_bias": 0.0,
    }
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    best_validation_loss = float("inf")
    best_validation_mse = float("inf")
    best_state: tuple[np.ndarray, ...] | None = None
    epochs_without_improvement = 0
    global_step = 0
    attention_scale = np.sqrt(max(model_dim, 1))

    for _ in range(config.max_epochs):
        permutation = rng.permutation(len(x_train))
        x_epoch = x_train[permutation]
        y_epoch = y_train_scaled[permutation]

        for start in range(0, len(x_epoch), config.batch_size):
            end = min(start + config.batch_size, len(x_epoch))
            x_batch = x_epoch[start:end]
            y_batch = y_epoch[start:end]
            global_step += 1

            encoded, queries, keys, values, attention, hidden, predictions = _forward(
                x_batch,
                input_weights,
                input_bias,
                positional_embedding,
                query_weights,
                key_weights,
                value_weights,
                output_weights,
                output_bias,
            )
            errors = predictions - y_batch
            batch_size = max(len(x_batch), 1)
            output_grad = (2.0 / batch_size) * errors

            pooled = hidden.mean(axis=1)
            grad_output_weights = pooled.T @ output_grad + config.weight_decay * output_weights
            grad_output_bias = float(output_grad.sum())
            grad_pooled = output_grad @ output_weights.T
            grad_hidden = np.repeat((grad_pooled / hidden.shape[1])[:, None, :], hidden.shape[1], axis=1)
            grad_context = grad_hidden * (1.0 - hidden**2)

            grad_attention = np.matmul(grad_context, np.swapaxes(values, 1, 2))
            grad_values = np.matmul(np.swapaxes(attention, 1, 2), grad_context)

            attn_dot = (grad_attention * attention).sum(axis=-1, keepdims=True)
            grad_scores = attention * (grad_attention - attn_dot)
            grad_queries = np.matmul(grad_scores, keys) / attention_scale
            grad_keys = np.matmul(np.swapaxes(grad_scores, 1, 2), queries) / attention_scale

            grad_query_weights = np.einsum("btd,bth->dh", encoded, grad_queries) + config.weight_decay * query_weights
            grad_key_weights = np.einsum("btd,bth->dh", encoded, grad_keys) + config.weight_decay * key_weights
            grad_value_weights = np.einsum("btd,bth->dh", encoded, grad_values) + config.weight_decay * value_weights

            grad_encoded = (
                np.einsum("bth,hd->btd", grad_queries, query_weights.T)
                + np.einsum("bth,hd->btd", grad_keys, key_weights.T)
                + np.einsum("bth,hd->btd", grad_values, value_weights.T)
            )
            grad_input_weights = np.einsum("btf,btd->fd", x_batch, grad_encoded) + config.weight_decay * input_weights
            grad_input_bias = grad_encoded.sum(axis=(0, 1))
            grad_positional_embedding = grad_encoded.sum(axis=0)

            for name, grad in (
                ("input_weights", grad_input_weights),
                ("input_bias", grad_input_bias),
                ("positional_embedding", grad_positional_embedding),
                ("query_weights", grad_query_weights),
                ("key_weights", grad_key_weights),
                ("value_weights", grad_value_weights),
                ("output_weights", grad_output_weights),
                ("output_bias", grad_output_bias),
            ):
                m_key = f"m_{name}"
                v_key = f"v_{name}"
                adam_state[m_key] = beta1 * adam_state[m_key] + (1.0 - beta1) * grad
                adam_state[v_key] = beta2 * adam_state[v_key] + (1.0 - beta2) * (grad**2)
                m_hat = adam_state[m_key] / (1.0 - beta1**global_step)
                v_hat = adam_state[v_key] / (1.0 - beta2**global_step)
                update = config.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                if name == "input_weights":
                    input_weights = input_weights - update.astype(np.float32)
                elif name == "input_bias":
                    input_bias = input_bias - update.astype(np.float32)
                elif name == "positional_embedding":
                    positional_embedding = positional_embedding - update.astype(np.float32)
                elif name == "query_weights":
                    query_weights = query_weights - update.astype(np.float32)
                elif name == "key_weights":
                    key_weights = key_weights - update.astype(np.float32)
                elif name == "value_weights":
                    value_weights = value_weights - update.astype(np.float32)
                elif name == "output_weights":
                    output_weights = output_weights - update.astype(np.float32)
                else:
                    output_bias = float(output_bias - update)

        _, _, _, _, _, _, validation_predictions = _forward(
            x_validation,
            input_weights,
            input_bias,
            positional_embedding,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            output_bias,
        )
        validation_loss = float(np.mean((validation_predictions - y_validation_scaled) ** 2))
        validation_predictions_actual = validation_predictions.ravel() * target_std + target_mean
        validation_mse = float(np.mean((validation_predictions_actual - y_validation) ** 2))
        if validation_loss + 1e-9 < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_mse = validation_mse
            best_state = (
                input_weights.copy(),
                input_bias.copy(),
                positional_embedding.copy(),
                query_weights.copy(),
                key_weights.copy(),
                value_weights.copy(),
                output_weights.copy(),
                float(output_bias),
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        best_state = (
            input_weights.copy(),
            input_bias.copy(),
            positional_embedding.copy(),
            query_weights.copy(),
            key_weights.copy(),
            value_weights.copy(),
            output_weights.copy(),
            float(output_bias),
        )

    return TransformerModel(
        input_weights=best_state[0],
        input_bias=best_state[1],
        positional_embedding=best_state[2],
        query_weights=best_state[3],
        key_weights=best_state[4],
        value_weights=best_state[5],
        output_weights=best_state[6],
        output_bias=best_state[7],
        feature_means=feature_means,
        feature_stds=feature_stds,
        target_mean=target_mean,
        target_std=target_std,
        feature_columns=tuple(feature_columns),
        lookback_window=config.lookback_window,
        model_dim=config.model_dim,
        validation_mse=best_validation_mse,
    )


def predict_transformer_model(model: TransformerModel, sequences: np.ndarray) -> np.ndarray:
    scaled = ((sequences.astype(np.float32) - model.feature_means) / model.feature_stds).astype(np.float32)
    _, _, _, _, _, _, predictions = _forward(
        scaled,
        model.input_weights,
        model.input_bias,
        model.positional_embedding,
        model.query_weights,
        model.key_weights,
        model.value_weights,
        model.output_weights,
        model.output_bias,
    )
    return predictions.ravel() * model.target_std + model.target_mean


def generate_transformer_target_weights(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: TransformerModelConfig,
    eval_start: str | pd.Timestamp,
    llm_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = prepare_sequence_dataset(features, strategy_config, model_config, llm_scores)
    frame = dataset.frame
    sequences = dataset.sequences

    eval_start_ts = pd.Timestamp(eval_start).normalize()
    tickers = sorted(features["ticker"].astype(str).str.upper().unique().tolist())
    dates = sorted(pd.to_datetime(features["date"].unique()).tolist())
    target_weights = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    walkforward_config = WalkForwardConfig(
        label_horizon_days=model_config.label_horizon_days,
        validation_window_days=model_config.validation_window_days,
        embargo_days=model_config.embargo_days,
        min_training_samples=model_config.min_training_samples,
        min_validation_samples=model_config.min_validation_samples,
        feature_columns=model_config.feature_columns,
        use_llm_feature=model_config.use_llm_feature,
    )

    snapshots: list[pd.DataFrame] = []
    for rebalance_date in rebalance_dates(features, strategy_config.rebalance):
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

        if model_config.training_lookback_days is not None:
            training_dates = sorted(pd.to_datetime(train_frame["date"]).dt.normalize().unique().tolist())
            if len(training_dates) > model_config.training_lookback_days:
                training_start = training_dates[-model_config.training_lookback_days]
                train_frame = train_frame.loc[pd.to_datetime(train_frame["date"]).dt.normalize() >= training_start].copy()
                if len(train_frame) < model_config.min_training_samples:
                    snapshots.append(snapshot)
                    continue

        eligible = select_live_candidates(frame, rebalance_date, strategy_config)
        if eligible.empty:
            snapshots.append(snapshot)
            continue

        train_sequences = sequences[train_frame["sequence_index"].astype(int).to_numpy()]
        validation_sequences = sequences[validation_frame["sequence_index"].astype(int).to_numpy()]
        eligible_sequences = sequences[eligible["sequence_index"].astype(int).to_numpy()]
        model = fit_transformer_model(
            train_sequences=train_sequences,
            train_targets=train_frame["future_return"].astype(float).to_numpy(),
            validation_sequences=validation_sequences,
            validation_targets=validation_frame["future_return"].astype(float).to_numpy(),
            feature_columns=dataset.feature_columns,
            config=model_config,
        )
        eligible["predicted_return"] = predict_transformer_model(model, eligible_sequences)
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
