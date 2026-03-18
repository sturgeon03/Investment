from __future__ import annotations

from us_invest_ai.transformer_strategy import TransformerModelConfig


def build_transformer_report_config(
    *,
    label_horizon_days: int,
    validation_window_days: int,
    embargo_days: int | None,
    min_training_samples: int,
    min_validation_samples: int,
    use_llm_feature: bool,
    sequence_lookback_window: int,
    training_lookback_days: int,
    model_dim: int,
    target_clip_quantile: float | None,
) -> TransformerModelConfig:
    return TransformerModelConfig(
        label_horizon_days=label_horizon_days,
        validation_window_days=validation_window_days,
        embargo_days=embargo_days,
        min_training_samples=min_training_samples,
        min_validation_samples=min_validation_samples,
        use_llm_feature=use_llm_feature,
        lookback_window=sequence_lookback_window,
        training_lookback_days=training_lookback_days,
        model_dim=model_dim,
        target_clip_quantile=target_clip_quantile,
    )
