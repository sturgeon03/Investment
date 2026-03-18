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
class LSTMModelConfig:
    label_horizon_days: int = 20
    validation_window_days: int = 60
    embargo_days: int | None = None
    min_training_samples: int = 252
    min_validation_samples: int = 120
    feature_columns: tuple[str, ...] = DEFAULT_SEQUENCE_FEATURE_COLUMNS
    use_llm_feature: bool = False
    lookback_window: int = 20
    hidden_dim: int = 12
    learning_rate: float = 0.003
    max_epochs: int = 100
    batch_size: int = 256
    weight_decay: float = 1e-4
    patience: int = 12
    random_seed: int = 7


@dataclass(slots=True)
class LSTMModel:
    w_f: np.ndarray
    b_f: np.ndarray
    w_i: np.ndarray
    b_i: np.ndarray
    w_g: np.ndarray
    b_g: np.ndarray
    w_o: np.ndarray
    b_o: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    feature_means: np.ndarray
    feature_stds: np.ndarray
    target_mean: float
    target_std: float
    feature_columns: tuple[str, ...]
    lookback_window: int
    hidden_dim: int
    validation_mse: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _standardize_sequences(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = sequences.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = sequences.std(axis=(0, 1), ddof=0, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1.0
    scaled = (sequences - means) / stds
    return scaled.astype(np.float32), means, stds


def _forward(
    x_batch: np.ndarray,
    w_f: np.ndarray,
    b_f: np.ndarray,
    w_i: np.ndarray,
    b_i: np.ndarray,
    w_g: np.ndarray,
    b_g: np.ndarray,
    w_o: np.ndarray,
    b_o: np.ndarray,
    output_weights: np.ndarray,
    output_bias: float,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    batch_size, _, input_dim = x_batch.shape
    hidden_dim = output_weights.shape[0]
    h_t = np.zeros((batch_size, hidden_dim), dtype=np.float32)
    c_t = np.zeros((batch_size, hidden_dim), dtype=np.float32)
    caches: list[dict[str, np.ndarray]] = []

    for step in range(x_batch.shape[1]):
        x_t = x_batch[:, step, :]
        concat = np.concatenate([x_t, h_t], axis=1)
        f_t = _sigmoid(concat @ w_f + b_f)
        i_t = _sigmoid(concat @ w_i + b_i)
        g_t = np.tanh(concat @ w_g + b_g)
        o_t = _sigmoid(concat @ w_o + b_o)
        c_prev = c_t
        h_prev = h_t
        c_t = f_t * c_t + i_t * g_t
        tanh_c = np.tanh(c_t)
        h_t = o_t * tanh_c
        caches.append(
            {
                "concat": concat,
                "f_t": f_t,
                "i_t": i_t,
                "g_t": g_t,
                "o_t": o_t,
                "c_prev": c_prev,
                "c_t": c_t,
                "tanh_c": tanh_c,
                "h_prev": h_prev,
                "input_dim": np.array([input_dim], dtype=np.int32),
            }
        )

    predictions = h_t @ output_weights + output_bias
    return caches, h_t, predictions


def fit_lstm_model(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    validation_sequences: np.ndarray,
    validation_targets: np.ndarray,
    feature_columns: tuple[str, ...],
    config: LSTMModelConfig,
) -> LSTMModel:
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
    hidden_dim = config.hidden_dim
    gate_input_dim = input_dim + hidden_dim

    def init_gate() -> tuple[np.ndarray, np.ndarray]:
        weights = rng.normal(
            0.0,
            np.sqrt(1.0 / max(gate_input_dim, 1)),
            size=(gate_input_dim, hidden_dim),
        ).astype(np.float32)
        bias = np.zeros(hidden_dim, dtype=np.float32)
        return weights, bias

    w_f, b_f = init_gate()
    w_i, b_i = init_gate()
    w_g, b_g = init_gate()
    w_o, b_o = init_gate()
    output_weights = rng.normal(
        0.0,
        np.sqrt(1.0 / max(hidden_dim, 1)),
        size=(hidden_dim, 1),
    ).astype(np.float32)
    output_bias = 0.0

    adam_state: dict[str, np.ndarray | float] = {
        "m_w_f": np.zeros_like(w_f),
        "v_w_f": np.zeros_like(w_f),
        "m_b_f": np.zeros_like(b_f),
        "v_b_f": np.zeros_like(b_f),
        "m_w_i": np.zeros_like(w_i),
        "v_w_i": np.zeros_like(w_i),
        "m_b_i": np.zeros_like(b_i),
        "v_b_i": np.zeros_like(b_i),
        "m_w_g": np.zeros_like(w_g),
        "v_w_g": np.zeros_like(w_g),
        "m_b_g": np.zeros_like(b_g),
        "v_b_g": np.zeros_like(b_g),
        "m_w_o": np.zeros_like(w_o),
        "v_w_o": np.zeros_like(w_o),
        "m_b_o": np.zeros_like(b_o),
        "v_b_o": np.zeros_like(b_o),
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

    for _ in range(config.max_epochs):
        permutation = rng.permutation(len(x_train))
        x_epoch = x_train[permutation]
        y_epoch = y_train_scaled[permutation]

        for start in range(0, len(x_epoch), config.batch_size):
            end = min(start + config.batch_size, len(x_epoch))
            x_batch = x_epoch[start:end]
            y_batch = y_epoch[start:end]
            global_step += 1

            caches, last_hidden, predictions = _forward(
                x_batch,
                w_f,
                b_f,
                w_i,
                b_i,
                w_g,
                b_g,
                w_o,
                b_o,
                output_weights,
                output_bias,
            )
            errors = predictions - y_batch
            batch_size = max(len(x_batch), 1)
            output_grad = (2.0 / batch_size) * errors

            grad_output_weights = last_hidden.T @ output_grad + config.weight_decay * output_weights
            grad_output_bias = float(output_grad.sum())
            grad_h = output_grad @ output_weights.T
            grad_c = np.zeros_like(last_hidden)

            grad_w_f = np.zeros_like(w_f)
            grad_b_f = np.zeros_like(b_f)
            grad_w_i = np.zeros_like(w_i)
            grad_b_i = np.zeros_like(b_i)
            grad_w_g = np.zeros_like(w_g)
            grad_b_g = np.zeros_like(b_g)
            grad_w_o = np.zeros_like(w_o)
            grad_b_o = np.zeros_like(b_o)

            for cache in reversed(caches):
                concat = cache["concat"]
                f_t = cache["f_t"]
                i_t = cache["i_t"]
                g_t = cache["g_t"]
                o_t = cache["o_t"]
                c_prev = cache["c_prev"]
                c_t = cache["c_t"]
                tanh_c = cache["tanh_c"]

                do = grad_h * tanh_c
                do_pre = do * o_t * (1.0 - o_t)
                dc = grad_h * o_t * (1.0 - tanh_c**2) + grad_c
                df = dc * c_prev
                df_pre = df * f_t * (1.0 - f_t)
                di = dc * g_t
                di_pre = di * i_t * (1.0 - i_t)
                dg = dc * i_t
                dg_pre = dg * (1.0 - g_t**2)

                grad_w_f += concat.T @ df_pre + config.weight_decay * w_f
                grad_b_f += df_pre.sum(axis=0)
                grad_w_i += concat.T @ di_pre + config.weight_decay * w_i
                grad_b_i += di_pre.sum(axis=0)
                grad_w_g += concat.T @ dg_pre + config.weight_decay * w_g
                grad_b_g += dg_pre.sum(axis=0)
                grad_w_o += concat.T @ do_pre + config.weight_decay * w_o
                grad_b_o += do_pre.sum(axis=0)

                dconcat = df_pre @ w_f.T + di_pre @ w_i.T + dg_pre @ w_g.T + do_pre @ w_o.T
                grad_h = dconcat[:, input_dim:]
                grad_c = dc * f_t

            for name, grad in (
                ("w_f", grad_w_f),
                ("b_f", grad_b_f),
                ("w_i", grad_w_i),
                ("b_i", grad_b_i),
                ("w_g", grad_w_g),
                ("b_g", grad_b_g),
                ("w_o", grad_w_o),
                ("b_o", grad_b_o),
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
                if name == "w_f":
                    w_f = w_f - update.astype(np.float32)
                elif name == "b_f":
                    b_f = b_f - update.astype(np.float32)
                elif name == "w_i":
                    w_i = w_i - update.astype(np.float32)
                elif name == "b_i":
                    b_i = b_i - update.astype(np.float32)
                elif name == "w_g":
                    w_g = w_g - update.astype(np.float32)
                elif name == "b_g":
                    b_g = b_g - update.astype(np.float32)
                elif name == "w_o":
                    w_o = w_o - update.astype(np.float32)
                elif name == "b_o":
                    b_o = b_o - update.astype(np.float32)
                elif name == "output_weights":
                    output_weights = output_weights - update.astype(np.float32)
                else:
                    output_bias = float(output_bias - update)

        _, _, validation_predictions = _forward(
            x_validation,
            w_f,
            b_f,
            w_i,
            b_i,
            w_g,
            b_g,
            w_o,
            b_o,
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
                w_f.copy(),
                b_f.copy(),
                w_i.copy(),
                b_i.copy(),
                w_g.copy(),
                b_g.copy(),
                w_o.copy(),
                b_o.copy(),
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
            w_f.copy(),
            b_f.copy(),
            w_i.copy(),
            b_i.copy(),
            w_g.copy(),
            b_g.copy(),
            w_o.copy(),
            b_o.copy(),
            output_weights.copy(),
            float(output_bias),
        )

    return LSTMModel(
        w_f=best_state[0],
        b_f=best_state[1],
        w_i=best_state[2],
        b_i=best_state[3],
        w_g=best_state[4],
        b_g=best_state[5],
        w_o=best_state[6],
        b_o=best_state[7],
        output_weights=best_state[8],
        output_bias=best_state[9],
        feature_means=feature_means,
        feature_stds=feature_stds,
        target_mean=target_mean,
        target_std=target_std,
        feature_columns=tuple(feature_columns),
        lookback_window=config.lookback_window,
        hidden_dim=config.hidden_dim,
        validation_mse=best_validation_mse,
    )


def predict_lstm_model(model: LSTMModel, sequences: np.ndarray) -> np.ndarray:
    scaled = ((sequences.astype(np.float32) - model.feature_means) / model.feature_stds).astype(np.float32)
    _, _, predictions = _forward(
        scaled,
        model.w_f,
        model.b_f,
        model.w_i,
        model.b_i,
        model.w_g,
        model.b_g,
        model.w_o,
        model.b_o,
        model.output_weights,
        model.output_bias,
    )
    return predictions.ravel() * model.target_std + model.target_mean


def generate_lstm_target_weights(
    features: pd.DataFrame,
    strategy_config: StrategyConfig,
    model_config: LSTMModelConfig,
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
        model = fit_lstm_model(
            train_sequences=train_sequences,
            train_targets=train_frame["future_return"].astype(float).to_numpy(),
            validation_sequences=validation_sequences,
            validation_targets=validation_frame["future_return"].astype(float).to_numpy(),
            feature_columns=dataset.feature_columns,
            config=model_config,
        )
        eligible["predicted_return"] = predict_lstm_model(model, eligible_sequences)
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
