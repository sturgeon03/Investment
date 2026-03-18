from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from us_invest_ai.config import StrategyConfig
from us_invest_ai.walkforward import (
    WalkForwardConfig,
    prepare_learning_frame,
    rebalance_dates,
    select_live_candidates,
    select_walkforward_splits,
)


DEFAULT_SEQUENCE_FEATURE_COLUMNS = (
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "rel_ret_20",
    "rel_ret_60",
    "vol_20",
    "vol_60",
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",
    "sma20_vs_sma50",
    "sma50_vs_sma200",
    "drawdown_20",
    "drawdown_60",
    "volume_ratio_20",
    "range_pct_20",
    "benchmark_ret_20",
    "benchmark_ret_60",
    "benchmark_vol_20",
    "cs_rel_ret_20_z",
    "universe_momentum_rank_pct",
    "sector_rel_ret_20_gap",
    "sector_momentum_rank_pct",
    "market_trend_flag",
    "market_high_vol_flag",
)


@dataclass(slots=True)
class TCNModelConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 60
    embargo_days: int | None = None
    min_training_samples: int = 252
    min_validation_samples: int = 120
    feature_columns: tuple[str, ...] = DEFAULT_SEQUENCE_FEATURE_COLUMNS
    use_llm_feature: bool = False
    lookback_window: int = 20
    kernel_size: int = 5
    hidden_channels: int = 8
    learning_rate: float = 0.005
    max_epochs: int = 120
    batch_size: int = 256
    weight_decay: float = 1e-4
    patience: int = 15
    random_seed: int = 7


@dataclass(slots=True)
class SequenceDataset:
    frame: pd.DataFrame
    sequences: np.ndarray
    feature_columns: tuple[str, ...]


@dataclass(slots=True)
class TCNModel:
    conv_kernels: np.ndarray
    conv_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    feature_means: np.ndarray
    feature_stds: np.ndarray
    target_mean: float
    target_std: float
    feature_columns: tuple[str, ...]
    lookback_window: int
    kernel_size: int
    validation_mse: float


def _rolling_sequence_windows(design: np.ndarray, lookback_window: int) -> np.ndarray:
    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive.")
    if len(design) < lookback_window:
        return np.empty((0, lookback_window, design.shape[1]), dtype=np.float32)

    windows = sliding_window_view(design, window_shape=lookback_window, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    return np.ascontiguousarray(windows, dtype=np.float32)


def prepare_sequence_dataset(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: TCNModelConfig,
    llm_scores: pd.DataFrame | None = None,
) -> SequenceDataset:
    walkforward_config = WalkForwardConfig(
        label_horizon_days=model_config.label_horizon_days,
        validation_window_days=model_config.validation_window_days,
        embargo_days=model_config.embargo_days,
        min_training_samples=model_config.min_training_samples,
        min_validation_samples=model_config.min_validation_samples,
        feature_columns=model_config.feature_columns,
        use_llm_feature=model_config.use_llm_feature,
    )
    base_frame = prepare_learning_frame(features, strategy_config, walkforward_config, llm_scores)
    feature_columns = tuple(walkforward_config.resolved_feature_columns())

    sequence_parts: list[pd.DataFrame] = []
    sequence_arrays: list[np.ndarray] = []
    sequence_offset = 0

    for _, group in base_frame.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        design = group[list(feature_columns)].astype(np.float32).to_numpy()
        windows = _rolling_sequence_windows(design, model_config.lookback_window)
        if len(windows) == 0:
            continue

        aligned_group = group.iloc[model_config.lookback_window - 1 :].copy().reset_index(drop=True)
        aligned_group["sequence_index"] = np.arange(sequence_offset, sequence_offset + len(aligned_group), dtype=int)
        sequence_offset += len(aligned_group)
        sequence_parts.append(aligned_group)
        sequence_arrays.append(windows)

    if not sequence_parts:
        empty_frame = base_frame.iloc[0:0].copy()
        empty_frame["sequence_index"] = pd.Series(dtype=int)
        empty_frame["has_sequence_features"] = pd.Series(dtype=bool)
        empty_frame["has_model_features"] = pd.Series(dtype=bool)
        return SequenceDataset(
            frame=empty_frame,
            sequences=np.empty((0, model_config.lookback_window, len(feature_columns)), dtype=np.float32),
            feature_columns=feature_columns,
        )

    sequence_frame = pd.concat(sequence_parts, ignore_index=True)
    sequences = np.concatenate(sequence_arrays, axis=0).astype(np.float32, copy=False)
    finite_mask = np.isfinite(sequences).all(axis=(1, 2))
    sequence_frame["has_sequence_features"] = finite_mask
    sequence_frame["has_model_features"] = sequence_frame["has_model_features"].to_numpy() & finite_mask
    return SequenceDataset(frame=sequence_frame, sequences=sequences, feature_columns=feature_columns)


def _standardize_sequences(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = sequences.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = sequences.std(axis=(0, 1), ddof=0, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1.0
    scaled = (sequences - means) / stds
    return scaled.astype(np.float32), means, stds


def _forward(
    x_batch: np.ndarray,
    conv_kernels: np.ndarray,
    conv_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = sliding_window_view(x_batch, window_shape=conv_kernels.shape[0], axis=1)
    windows = np.moveaxis(windows, -1, 2)
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    hidden_linear = np.tensordot(windows, conv_kernels, axes=([2, 3], [0, 1])) + conv_bias
    hidden = np.tanh(hidden_linear)
    pooled = hidden.mean(axis=1)
    output = pooled @ output_weights + output_bias
    return windows, hidden, pooled, output


def fit_tcn_model(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    validation_sequences: np.ndarray,
    validation_targets: np.ndarray,
    feature_columns: tuple[str, ...],
    config: TCNModelConfig,
) -> TCNModel:
    if config.kernel_size <= 0 or config.kernel_size > config.lookback_window:
        raise ValueError("kernel_size must be positive and no larger than lookback_window.")

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
    hidden_channels = config.hidden_channels
    conv_kernels = rng.normal(
        0.0,
        np.sqrt(2.0 / max(config.kernel_size * input_dim, 1)),
        size=(config.kernel_size, input_dim, hidden_channels),
    ).astype(np.float32)
    conv_bias = np.zeros(hidden_channels, dtype=np.float32)
    output_weights = rng.normal(
        0.0,
        np.sqrt(1.0 / max(hidden_channels, 1)),
        size=(hidden_channels, 1),
    ).astype(np.float32)
    output_bias = 0.0

    adam_state: dict[str, np.ndarray | float] = {
        "m_conv_kernels": np.zeros_like(conv_kernels),
        "v_conv_kernels": np.zeros_like(conv_kernels),
        "m_conv_bias": np.zeros_like(conv_bias),
        "v_conv_bias": np.zeros_like(conv_bias),
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

            windows, hidden, pooled, predictions = _forward(
                x_batch,
                conv_kernels,
                conv_bias,
                output_weights,
                output_bias,
            )
            errors = predictions - y_batch
            batch_size = max(len(x_batch), 1)
            output_grad = (2.0 / batch_size) * errors

            grad_output_weights = pooled.T @ output_grad + config.weight_decay * output_weights
            grad_output_bias = float(output_grad.sum())
            grad_pooled = output_grad @ output_weights.T
            pooled_steps = hidden.shape[1]
            grad_hidden = np.repeat(
                (grad_pooled / max(pooled_steps, 1))[:, None, :],
                pooled_steps,
                axis=1,
            )
            grad_hidden_linear = grad_hidden * (1.0 - hidden ** 2)
            grad_conv_kernels = (
                np.tensordot(windows, grad_hidden_linear, axes=([0, 1], [0, 1]))
                + config.weight_decay * conv_kernels
            )
            grad_conv_bias = grad_hidden_linear.sum(axis=(0, 1))

            for name, grad in (
                ("conv_kernels", grad_conv_kernels),
                ("conv_bias", grad_conv_bias),
                ("output_weights", grad_output_weights),
                ("output_bias", grad_output_bias),
            ):
                m_key = f"m_{name}"
                v_key = f"v_{name}"
                adam_state[m_key] = beta1 * adam_state[m_key] + (1.0 - beta1) * grad
                adam_state[v_key] = beta2 * adam_state[v_key] + (1.0 - beta2) * (grad ** 2)
                m_hat = adam_state[m_key] / (1.0 - beta1 ** global_step)
                v_hat = adam_state[v_key] / (1.0 - beta2 ** global_step)
                update = config.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                if name == "conv_kernels":
                    conv_kernels = conv_kernels - update.astype(np.float32)
                elif name == "conv_bias":
                    conv_bias = conv_bias - update.astype(np.float32)
                elif name == "output_weights":
                    output_weights = output_weights - update.astype(np.float32)
                else:
                    output_bias = float(output_bias - update)

        _, _, _, validation_predictions = _forward(
            x_validation,
            conv_kernels,
            conv_bias,
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
            output_weights.copy(),
            float(output_bias),
        )

    return TCNModel(
        conv_kernels=best_state[0],
        conv_bias=best_state[1],
        output_weights=best_state[2],
        output_bias=best_state[3],
        feature_means=feature_means,
        feature_stds=feature_stds,
        target_mean=target_mean,
        target_std=target_std,
        feature_columns=tuple(feature_columns),
        lookback_window=config.lookback_window,
        kernel_size=config.kernel_size,
        validation_mse=best_validation_mse,
    )


def predict_tcn_model(model: TCNModel, sequences: np.ndarray) -> np.ndarray:
    scaled = ((sequences.astype(np.float32) - model.feature_means) / model.feature_stds).astype(np.float32)
    _, _, _, predictions = _forward(
        scaled,
        model.conv_kernels,
        model.conv_bias,
        model.output_weights,
        model.output_bias,
    )
    return predictions.ravel() * model.target_std + model.target_mean


def generate_tcn_target_weights(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: TCNModelConfig,
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

        eligible = select_live_candidates(frame, rebalance_date, strategy_config)
        if eligible.empty:
            snapshots.append(snapshot)
            continue

        train_sequences = sequences[train_frame["sequence_index"].astype(int).to_numpy()]
        validation_sequences = sequences[validation_frame["sequence_index"].astype(int).to_numpy()]
        eligible_sequences = sequences[eligible["sequence_index"].astype(int).to_numpy()]
        model = fit_tcn_model(
            train_sequences=train_sequences,
            train_targets=train_frame["future_return"].astype(float).to_numpy(),
            validation_sequences=validation_sequences,
            validation_targets=validation_frame["future_return"].astype(float).to_numpy(),
            feature_columns=dataset.feature_columns,
            config=model_config,
        )
        eligible["predicted_return"] = predict_tcn_model(model, eligible_sequences)
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
