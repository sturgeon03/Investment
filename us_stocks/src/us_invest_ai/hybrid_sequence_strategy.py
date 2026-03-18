from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from us_invest_ai.config import StrategyConfig
from us_invest_ai.tcn_strategy import DEFAULT_SEQUENCE_FEATURE_COLUMNS, prepare_sequence_dataset
from us_invest_ai.walkforward import (
    DEFAULT_FEATURE_COLUMNS,
    WalkForwardConfig,
    rebalance_dates,
    select_live_candidates,
    select_walkforward_splits,
)


DEFAULT_HYBRID_STATIC_FEATURE_COLUMNS = tuple(DEFAULT_FEATURE_COLUMNS)


@dataclass(slots=True)
class HybridSequenceModelConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 60
    embargo_days: int | None = None
    min_training_samples: int = 252
    min_validation_samples: int = 120
    sequence_feature_columns: tuple[str, ...] = DEFAULT_SEQUENCE_FEATURE_COLUMNS
    static_feature_columns: tuple[str, ...] = DEFAULT_HYBRID_STATIC_FEATURE_COLUMNS
    use_llm_feature: bool = False
    lookback_window: int = 20
    kernel_size: int = 5
    sequence_hidden_channels: int = 8
    static_hidden_dim: int = 16
    learning_rate: float = 0.005
    max_epochs: int = 120
    batch_size: int = 256
    weight_decay: float = 1e-4
    patience: int = 15
    random_seed: int = 7

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.sequence_feature_columns


@dataclass(slots=True)
class HybridSequenceModel:
    conv_kernels: np.ndarray
    conv_bias: np.ndarray
    static_weights: np.ndarray
    static_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    sequence_feature_means: np.ndarray
    sequence_feature_stds: np.ndarray
    static_feature_means: np.ndarray
    static_feature_stds: np.ndarray
    target_mean: float
    target_std: float
    sequence_feature_columns: tuple[str, ...]
    static_feature_columns: tuple[str, ...]
    lookback_window: int
    kernel_size: int
    validation_mse: float


def _standardize_sequences(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = sequences.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = sequences.std(axis=(0, 1), ddof=0, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1.0
    scaled = (sequences - means) / stds
    return scaled.astype(np.float32), means, stds


def _standardize_static_features(static_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = static_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = static_features.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1.0
    scaled = (static_features - means) / stds
    return scaled.astype(np.float32), means, stds


def _forward(
    sequence_batch: np.ndarray,
    static_batch: np.ndarray,
    conv_kernels: np.ndarray,
    conv_bias: np.ndarray,
    static_weights: np.ndarray,
    static_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = sliding_window_view(sequence_batch, window_shape=conv_kernels.shape[0], axis=1)
    windows = np.moveaxis(windows, -1, 2)
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    sequence_hidden_linear = np.tensordot(windows, conv_kernels, axes=([2, 3], [0, 1])) + conv_bias
    sequence_hidden = np.tanh(sequence_hidden_linear)
    sequence_pooled = sequence_hidden.mean(axis=1)
    static_hidden_linear = static_batch @ static_weights + static_bias
    static_hidden = np.tanh(static_hidden_linear)
    merged = np.concatenate([sequence_pooled, static_hidden], axis=1)
    output = merged @ output_weights + output_bias
    return windows, sequence_hidden, sequence_pooled, static_hidden, output


def fit_hybrid_sequence_model(
    train_sequences: np.ndarray,
    train_static_features: np.ndarray,
    train_targets: np.ndarray,
    validation_sequences: np.ndarray,
    validation_static_features: np.ndarray,
    validation_targets: np.ndarray,
    sequence_feature_columns: tuple[str, ...],
    static_feature_columns: tuple[str, ...],
    config: HybridSequenceModelConfig,
) -> HybridSequenceModel:
    if config.kernel_size <= 0 or config.kernel_size > config.lookback_window:
        raise ValueError("kernel_size must be positive and no larger than lookback_window.")

    x_train_sequence, sequence_feature_means, sequence_feature_stds = _standardize_sequences(
        train_sequences.astype(np.float32)
    )
    x_validation_sequence = (
        (validation_sequences.astype(np.float32) - sequence_feature_means) / sequence_feature_stds
    ).astype(np.float32)
    x_train_static, static_feature_means, static_feature_stds = _standardize_static_features(
        train_static_features.astype(np.float32)
    )
    x_validation_static = (
        (validation_static_features.astype(np.float32) - static_feature_means) / static_feature_stds
    ).astype(np.float32)

    y_train = train_targets.astype(np.float32)
    y_validation = validation_targets.astype(np.float32)
    target_mean = float(y_train.mean())
    target_std = float(y_train.std(ddof=0))
    if target_std == 0:
        target_std = 1.0

    y_train_scaled = ((y_train - target_mean) / target_std).reshape(-1, 1).astype(np.float32)
    y_validation_scaled = ((y_validation - target_mean) / target_std).reshape(-1, 1).astype(np.float32)

    rng = np.random.default_rng(config.random_seed)
    sequence_input_dim = x_train_sequence.shape[2]
    conv_kernels = rng.normal(
        0.0,
        np.sqrt(2.0 / max(config.kernel_size * sequence_input_dim, 1)),
        size=(config.kernel_size, sequence_input_dim, config.sequence_hidden_channels),
    ).astype(np.float32)
    conv_bias = np.zeros(config.sequence_hidden_channels, dtype=np.float32)
    static_input_dim = x_train_static.shape[1]
    static_weights = rng.normal(
        0.0,
        np.sqrt(2.0 / max(static_input_dim, 1)),
        size=(static_input_dim, config.static_hidden_dim),
    ).astype(np.float32)
    static_bias = np.zeros(config.static_hidden_dim, dtype=np.float32)
    output_weights = rng.normal(
        0.0,
        np.sqrt(1.0 / max(config.sequence_hidden_channels + config.static_hidden_dim, 1)),
        size=(config.sequence_hidden_channels + config.static_hidden_dim, 1),
    ).astype(np.float32)
    output_bias = 0.0

    adam_state: dict[str, np.ndarray | float] = {
        "m_conv_kernels": np.zeros_like(conv_kernels),
        "v_conv_kernels": np.zeros_like(conv_kernels),
        "m_conv_bias": np.zeros_like(conv_bias),
        "v_conv_bias": np.zeros_like(conv_bias),
        "m_static_weights": np.zeros_like(static_weights),
        "v_static_weights": np.zeros_like(static_weights),
        "m_static_bias": np.zeros_like(static_bias),
        "v_static_bias": np.zeros_like(static_bias),
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
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    epochs_without_improvement = 0
    global_step = 0

    for _ in range(config.max_epochs):
        permutation = rng.permutation(len(x_train_sequence))
        x_epoch_sequence = x_train_sequence[permutation]
        x_epoch_static = x_train_static[permutation]
        y_epoch = y_train_scaled[permutation]

        for start in range(0, len(x_epoch_sequence), config.batch_size):
            end = min(start + config.batch_size, len(x_epoch_sequence))
            sequence_batch = x_epoch_sequence[start:end]
            static_batch = x_epoch_static[start:end]
            y_batch = y_epoch[start:end]
            global_step += 1

            windows, sequence_hidden, sequence_pooled, static_hidden, predictions = _forward(
                sequence_batch,
                static_batch,
                conv_kernels,
                conv_bias,
                static_weights,
                static_bias,
                output_weights,
                output_bias,
            )
            errors = predictions - y_batch
            batch_size = max(len(sequence_batch), 1)
            output_grad = (2.0 / batch_size) * errors

            merged = np.concatenate([sequence_pooled, static_hidden], axis=1)
            grad_output_weights = merged.T @ output_grad + config.weight_decay * output_weights
            grad_output_bias = float(output_grad.sum())
            grad_merged = output_grad @ output_weights.T

            grad_sequence_pooled = grad_merged[:, : config.sequence_hidden_channels]
            grad_static_hidden = grad_merged[:, config.sequence_hidden_channels :]
            grad_static_linear = grad_static_hidden * (1.0 - static_hidden**2)
            grad_static_weights = static_batch.T @ grad_static_linear + config.weight_decay * static_weights
            grad_static_bias = grad_static_linear.sum(axis=0)

            pooled_steps = sequence_hidden.shape[1]
            grad_sequence_hidden = np.repeat(
                (grad_sequence_pooled / max(pooled_steps, 1))[:, None, :],
                pooled_steps,
                axis=1,
            )
            grad_sequence_hidden_linear = grad_sequence_hidden * (1.0 - sequence_hidden**2)
            grad_conv_kernels = (
                np.tensordot(windows, grad_sequence_hidden_linear, axes=([0, 1], [0, 1]))
                + config.weight_decay * conv_kernels
            )
            grad_conv_bias = grad_sequence_hidden_linear.sum(axis=(0, 1))

            for name, grad in (
                ("conv_kernels", grad_conv_kernels),
                ("conv_bias", grad_conv_bias),
                ("static_weights", grad_static_weights),
                ("static_bias", grad_static_bias),
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
                if name == "conv_kernels":
                    conv_kernels = conv_kernels - update.astype(np.float32)
                elif name == "conv_bias":
                    conv_bias = conv_bias - update.astype(np.float32)
                elif name == "static_weights":
                    static_weights = static_weights - update.astype(np.float32)
                elif name == "static_bias":
                    static_bias = static_bias - update.astype(np.float32)
                elif name == "output_weights":
                    output_weights = output_weights - update.astype(np.float32)
                else:
                    output_bias = float(output_bias - update)

        _, _, _, _, validation_predictions = _forward(
            x_validation_sequence,
            x_validation_static,
            conv_kernels,
            conv_bias,
            static_weights,
            static_bias,
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
                conv_kernels.copy(),
                conv_bias.copy(),
                static_weights.copy(),
                static_bias.copy(),
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
            conv_kernels.copy(),
            conv_bias.copy(),
            static_weights.copy(),
            static_bias.copy(),
            output_weights.copy(),
            float(output_bias),
        )

    return HybridSequenceModel(
        conv_kernels=best_state[0],
        conv_bias=best_state[1],
        static_weights=best_state[2],
        static_bias=best_state[3],
        output_weights=best_state[4],
        output_bias=best_state[5],
        sequence_feature_means=sequence_feature_means,
        sequence_feature_stds=sequence_feature_stds,
        static_feature_means=static_feature_means,
        static_feature_stds=static_feature_stds,
        target_mean=target_mean,
        target_std=target_std,
        sequence_feature_columns=tuple(sequence_feature_columns),
        static_feature_columns=tuple(static_feature_columns),
        lookback_window=config.lookback_window,
        kernel_size=config.kernel_size,
        validation_mse=best_validation_mse,
    )


def predict_hybrid_sequence_model(
    model: HybridSequenceModel,
    sequences: np.ndarray,
    static_features: np.ndarray,
) -> np.ndarray:
    scaled_sequences = (
        (sequences.astype(np.float32) - model.sequence_feature_means) / model.sequence_feature_stds
    ).astype(np.float32)
    scaled_static = (
        (static_features.astype(np.float32) - model.static_feature_means) / model.static_feature_stds
    ).astype(np.float32)
    _, _, _, _, predictions = _forward(
        scaled_sequences,
        scaled_static,
        model.conv_kernels,
        model.conv_bias,
        model.static_weights,
        model.static_bias,
        model.output_weights,
        model.output_bias,
    )
    return predictions.ravel() * model.target_std + model.target_mean


def generate_hybrid_sequence_target_weights(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: HybridSequenceModelConfig,
    eval_start: str | pd.Timestamp,
    llm_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = prepare_sequence_dataset(
        features=features,
        strategy_config=strategy_config,
        model_config=model_config,
        llm_scores=llm_scores,
    )
    frame = dataset.frame.copy()
    sequences = dataset.sequences

    static_feature_columns = list(model_config.static_feature_columns)
    if model_config.use_llm_feature and "llm_score" not in static_feature_columns:
        static_feature_columns.append("llm_score")
    frame["has_static_features"] = frame[static_feature_columns].notna().all(axis=1)
    frame["has_model_features"] = frame["has_model_features"].to_numpy() & frame["has_static_features"].to_numpy()

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
        feature_columns=model_config.sequence_feature_columns,
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

        eligible = select_live_candidates(frame, rebalance_date, strategy_config)
        if eligible.empty:
            snapshots.append(snapshot)
            continue

        train_sequences = sequences[train_frame["sequence_index"].astype(int).to_numpy()]
        validation_sequences = sequences[validation_frame["sequence_index"].astype(int).to_numpy()]
        eligible_sequences = sequences[eligible["sequence_index"].astype(int).to_numpy()]

        train_static = train_frame[static_feature_columns].astype(float).to_numpy(dtype=np.float32)
        validation_static = validation_frame[static_feature_columns].astype(float).to_numpy(dtype=np.float32)
        eligible_static = eligible[static_feature_columns].astype(float).to_numpy(dtype=np.float32)

        model = fit_hybrid_sequence_model(
            train_sequences=train_sequences,
            train_static_features=train_static,
            train_targets=train_frame["future_return"].astype(float).to_numpy(),
            validation_sequences=validation_sequences,
            validation_static_features=validation_static,
            validation_targets=validation_frame["future_return"].astype(float).to_numpy(),
            sequence_feature_columns=dataset.feature_columns,
            static_feature_columns=tuple(static_feature_columns),
            config=model_config,
        )
        eligible["predicted_return"] = predict_hybrid_sequence_model(model, eligible_sequences, eligible_static)
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
