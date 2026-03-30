from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_invest_ai.signal_hardening_report import (
    build_performance_degradation_analysis,
    build_signal_hardening_guidance,
    build_signal_hardening_leaderboard,
)


def _make_last_year_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model_name": "configured_baseline", "ending_capital": 116000.0, "eval_end": "2026-03-19"},
            {"model_name": "transformer_walkforward", "ending_capital": 127500.0, "eval_end": "2026-03-19"},
            {"model_name": "tree_walkforward", "ending_capital": 113000.0, "eval_end": "2026-03-19"},
        ]
    )


def _make_stability_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "configured_baseline",
                "window_label": "2024-03-15_to_2025-03-18",
                "eval_start": "2024-03-15",
                "eval_end": "2025-03-18",
                "ending_capital": 100660.0,
                "ending_capital_gap_vs_baseline": 0.0,
                "beat_baseline": False,
                "information_ratio": -0.9,
                "max_drawdown": -0.16,
                "avg_validation_mse": None,
            },
            {
                "model_name": "configured_baseline",
                "window_label": "2025-03-19_to_2026-03-19",
                "eval_start": "2025-03-19",
                "eval_end": "2026-03-19",
                "ending_capital": 116167.0,
                "ending_capital_gap_vs_baseline": 0.0,
                "beat_baseline": False,
                "information_ratio": -0.08,
                "max_drawdown": -0.13,
                "avg_validation_mse": None,
            },
            {
                "model_name": "transformer_walkforward",
                "window_label": "2024-03-15_to_2025-03-18",
                "eval_start": "2024-03-15",
                "eval_end": "2025-03-18",
                "ending_capital": 96939.0,
                "ending_capital_gap_vs_baseline": -3721.0,
                "beat_baseline": False,
                "information_ratio": -1.29,
                "max_drawdown": -0.17,
                "avg_validation_mse": 0.0077,
            },
            {
                "model_name": "transformer_walkforward",
                "window_label": "2025-03-19_to_2026-03-19",
                "eval_start": "2025-03-19",
                "eval_end": "2026-03-19",
                "ending_capital": 127124.0,
                "ending_capital_gap_vs_baseline": 10957.0,
                "beat_baseline": True,
                "information_ratio": 0.66,
                "max_drawdown": -0.15,
                "avg_validation_mse": 0.0108,
            },
            {
                "model_name": "tree_walkforward",
                "window_label": "2024-03-15_to_2025-03-18",
                "eval_start": "2024-03-15",
                "eval_end": "2025-03-18",
                "ending_capital": 102204.0,
                "ending_capital_gap_vs_baseline": 1544.0,
                "beat_baseline": True,
                "information_ratio": -0.73,
                "max_drawdown": -0.16,
                "avg_validation_mse": 0.0071,
            },
            {
                "model_name": "tree_walkforward",
                "window_label": "2025-03-19_to_2026-03-19",
                "eval_start": "2025-03-19",
                "eval_end": "2026-03-19",
                "ending_capital": 113883.0,
                "ending_capital_gap_vs_baseline": -2285.0,
                "beat_baseline": False,
                "information_ratio": -0.27,
                "max_drawdown": -0.16,
                "avg_validation_mse": 0.0103,
            },
        ]
    )


def test_build_signal_hardening_leaderboard_sorts_by_repeated_window_strength() -> None:
    leaderboard = build_signal_hardening_leaderboard(_make_stability_summary(), _make_last_year_summary())

    assert leaderboard.iloc[0]["model_name"] == "transformer_walkforward"
    assert leaderboard.loc[leaderboard["model_name"] == "transformer_walkforward", "priority_tier"].iloc[0] == "promote"
    assert leaderboard.loc[leaderboard["model_name"] == "configured_baseline", "priority_tier"].iloc[0] == "deprioritize"


def test_build_performance_degradation_analysis_preserves_transformer_split_metrics() -> None:
    leaderboard = build_signal_hardening_leaderboard(_make_stability_summary(), _make_last_year_summary())
    analysis = build_performance_degradation_analysis(leaderboard)

    assert analysis["transformer_split_check"]["available"] is True
    assert analysis["transformer_split_check"]["details"]["latest_report_gap_vs_baseline"] == 11500.0
    assert "transformer_walkforward" in analysis["model_diagnostics"]


def test_build_signal_hardening_guidance_mentions_transformer_split() -> None:
    leaderboard = build_signal_hardening_leaderboard(_make_stability_summary(), _make_last_year_summary())
    analysis = build_performance_degradation_analysis(leaderboard)

    guidance = build_signal_hardening_guidance(
        leaderboard,
        analysis,
        last_year_summary_path=Path("us_stocks/artifacts/last_year.csv"),
        stability_summary_path=Path("us_stocks/artifacts/stability.csv"),
    )

    assert "seq40 + clip_q95" in guidance
    assert "Top repeated-window model" in guidance
