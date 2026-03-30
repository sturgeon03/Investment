from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze last-year and repeated-window report outputs to quantify model degradation "
            "and produce signal-hardening guidance."
        )
    )
    parser.add_argument(
        "--last-year-summary",
        default="us_stocks/artifacts/deep_learning_large_cap_60_dynamic_seq40_clip_q95_last_year/deep_learning_summary_last_year.csv",
        help="Last-year deep-learning summary CSV.",
    )
    parser.add_argument(
        "--stability-summary",
        default="us_stocks/artifacts/stability_large_cap_60_dynamic_seq20_clip_q95/stability_window_summary.csv",
        help="Repeated-window stability summary CSV.",
    )
    parser.add_argument(
        "--leaderboard-output",
        default="outputs/experiments/signal_hardening/leaderboard.csv",
        help="Output path for the robustness leaderboard.",
    )
    parser.add_argument(
        "--analysis-output",
        default="outputs/reports/performance_degradation/analysis.json",
        help="Output path for the performance degradation analysis JSON.",
    )
    parser.add_argument(
        "--guidance-output",
        default="outputs/reports/signal_hardening/signal_hardening_guidance.md",
        help="Output path for the Markdown guidance report.",
    )
    return parser.parse_args()


def load_report_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("eval_start", "eval_end"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    return frame


def _extract_last_year_gaps(last_year_summary: pd.DataFrame) -> pd.DataFrame:
    summary = last_year_summary.copy()
    if summary.empty:
        return pd.DataFrame(columns=["model_name", "latest_report_gap_vs_baseline", "latest_report_ending_capital"])

    baseline_row = summary.loc[summary["model_name"] == "configured_baseline"]
    if baseline_row.empty:
        raise ValueError("Last-year summary is missing configured_baseline.")
    baseline_ending_capital = float(baseline_row["ending_capital"].iloc[0])
    summary["latest_report_gap_vs_baseline"] = summary["ending_capital"].astype(float) - baseline_ending_capital
    summary = summary.rename(columns={"ending_capital": "latest_report_ending_capital"})
    return summary[["model_name", "latest_report_gap_vs_baseline", "latest_report_ending_capital"]]


def build_signal_hardening_leaderboard(
    stability_summary: pd.DataFrame,
    last_year_summary: pd.DataFrame,
) -> pd.DataFrame:
    window_summary = stability_summary.copy()
    if window_summary.empty:
        raise ValueError("Stability summary is empty.")

    window_summary["ending_capital_gap_vs_baseline"] = window_summary["ending_capital_gap_vs_baseline"].astype(float)
    latest_eval_end = pd.Timestamp(window_summary["eval_end"].max()).normalize()
    latest_window = window_summary.loc[window_summary["eval_end"] == latest_eval_end].copy()
    latest_window = latest_window.rename(
        columns={
            "ending_capital_gap_vs_baseline": "stability_latest_gap_vs_baseline",
            "ending_capital": "stability_latest_ending_capital",
            "window_label": "stability_latest_window_label",
            "avg_validation_mse": "stability_latest_validation_mse",
        }
    )

    aggregate = (
        window_summary.groupby("model_name", as_index=False)
        .agg(
            stability_window_count=("window_label", "nunique"),
            stability_avg_gap_vs_baseline=("ending_capital_gap_vs_baseline", "mean"),
            stability_median_gap_vs_baseline=("ending_capital_gap_vs_baseline", "median"),
            stability_worst_gap_vs_baseline=("ending_capital_gap_vs_baseline", "min"),
            stability_best_gap_vs_baseline=("ending_capital_gap_vs_baseline", "max"),
            stability_gap_std=("ending_capital_gap_vs_baseline", lambda values: float(pd.Series(values).std(ddof=0))),
            stability_beat_rate=("beat_baseline", "mean"),
            stability_avg_information_ratio=("information_ratio", "mean"),
            stability_avg_max_drawdown=("max_drawdown", "mean"),
            stability_avg_validation_mse=("avg_validation_mse", "mean"),
        )
        .reset_index(drop=True)
    )

    leaderboard = aggregate.merge(
        latest_window[
            [
                "model_name",
                "stability_latest_window_label",
                "stability_latest_gap_vs_baseline",
                "stability_latest_ending_capital",
                "stability_latest_validation_mse",
            ]
        ],
        on="model_name",
        how="left",
    ).merge(
        _extract_last_year_gaps(last_year_summary),
        on="model_name",
        how="left",
    )
    leaderboard["latest_report_gap_delta_vs_stability_latest"] = (
        leaderboard["latest_report_gap_vs_baseline"] - leaderboard["stability_latest_gap_vs_baseline"]
    )
    leaderboard["degradation_gap_latest_vs_avg"] = (
        leaderboard["stability_latest_gap_vs_baseline"] - leaderboard["stability_avg_gap_vs_baseline"]
    )
    leaderboard["robust_positive_windows"] = (
        leaderboard["stability_beat_rate"] >= 0.5
    ) & (leaderboard["stability_worst_gap_vs_baseline"] > -5_000.0)
    leaderboard["priority_tier"] = "deprioritize"
    leaderboard.loc[
        (leaderboard["stability_avg_gap_vs_baseline"] > 0.0) & (leaderboard["stability_beat_rate"] >= 0.5),
        "priority_tier",
    ] = "promote"
    leaderboard.loc[
        (leaderboard["priority_tier"] == "deprioritize")
        & (
            (leaderboard["stability_latest_gap_vs_baseline"] > 0.0)
            | (leaderboard["latest_report_gap_vs_baseline"] > 0.0)
        ),
        "priority_tier",
    ] = "challenger"
    leaderboard = leaderboard.sort_values(
        [
            "stability_avg_gap_vs_baseline",
            "stability_worst_gap_vs_baseline",
            "stability_beat_rate",
            "latest_report_gap_vs_baseline",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", leaderboard.index + 1)
    return leaderboard


def build_performance_degradation_analysis(leaderboard: pd.DataFrame) -> dict[str, Any]:
    if leaderboard.empty:
        raise ValueError("Leaderboard is empty.")

    baseline_row = leaderboard.loc[leaderboard["model_name"] == "configured_baseline"]
    if baseline_row.empty:
        raise ValueError("Leaderboard is missing configured_baseline.")

    top_ranked = leaderboard.iloc[0]
    transformer_row = leaderboard.loc[leaderboard["model_name"] == "transformer_walkforward"]
    transformer_metrics = transformer_row.iloc[0].to_dict() if not transformer_row.empty else None

    model_diagnostics: dict[str, Any] = {}
    for _, row in leaderboard.iterrows():
        model_diagnostics[str(row["model_name"])] = {
            "rank": int(row["rank"]),
            "priority_tier": str(row["priority_tier"]),
            "stability_window_count": int(row["stability_window_count"]),
            "stability_latest_window_label": str(row["stability_latest_window_label"]),
            "stability_latest_gap_vs_baseline": float(row["stability_latest_gap_vs_baseline"]),
            "stability_avg_gap_vs_baseline": float(row["stability_avg_gap_vs_baseline"]),
            "stability_worst_gap_vs_baseline": float(row["stability_worst_gap_vs_baseline"]),
            "stability_gap_std": float(row["stability_gap_std"]),
            "stability_beat_rate": float(row["stability_beat_rate"]),
            "latest_report_gap_vs_baseline": (
                float(row["latest_report_gap_vs_baseline"])
                if pd.notna(row["latest_report_gap_vs_baseline"])
                else None
            ),
            "latest_report_gap_delta_vs_stability_latest": (
                float(row["latest_report_gap_delta_vs_stability_latest"])
                if pd.notna(row["latest_report_gap_delta_vs_stability_latest"])
                else None
            ),
            "degradation_gap_latest_vs_avg": float(row["degradation_gap_latest_vs_avg"]),
        }

    return {
        "summary": {
            "top_ranked_model": str(top_ranked["model_name"]),
            "top_ranked_priority_tier": str(top_ranked["priority_tier"]),
            "top_ranked_avg_gap_vs_baseline": float(top_ranked["stability_avg_gap_vs_baseline"]),
            "baseline_stability_latest_window_label": str(baseline_row["stability_latest_window_label"].iloc[0]),
        },
        "transformer_split_check": {
            "available": transformer_metrics is not None,
            "details": (
                {
                    "stability_latest_gap_vs_baseline": float(transformer_metrics["stability_latest_gap_vs_baseline"]),
                    "stability_avg_gap_vs_baseline": float(transformer_metrics["stability_avg_gap_vs_baseline"]),
                    "stability_worst_gap_vs_baseline": float(transformer_metrics["stability_worst_gap_vs_baseline"]),
                    "latest_report_gap_vs_baseline": (
                        float(transformer_metrics["latest_report_gap_vs_baseline"])
                        if pd.notna(transformer_metrics["latest_report_gap_vs_baseline"])
                        else None
                    ),
                    "latest_report_gap_delta_vs_stability_latest": (
                        float(transformer_metrics["latest_report_gap_delta_vs_stability_latest"])
                        if pd.notna(transformer_metrics["latest_report_gap_delta_vs_stability_latest"])
                        else None
                    ),
                }
                if transformer_metrics is not None
                else None
            ),
        },
        "recommendations": {
            "promote_tier_models": leaderboard.loc[
                leaderboard["priority_tier"] == "promote", "model_name"
            ].astype(str).tolist(),
            "challenger_tier_models": leaderboard.loc[
                leaderboard["priority_tier"] == "challenger", "model_name"
            ].astype(str).tolist(),
            "deprioritize_tier_models": leaderboard.loc[
                leaderboard["priority_tier"] == "deprioritize", "model_name"
            ].astype(str).tolist(),
            "next_focus": (
                "Keep the promoted transformer split under canonical monitoring and use repeated-window degradation, "
                "not latest-window upside alone, as the gate for further model promotion."
            ),
        },
        "model_diagnostics": model_diagnostics,
    }


def build_signal_hardening_guidance(
    leaderboard: pd.DataFrame,
    analysis: dict[str, Any],
    *,
    last_year_summary_path: Path,
    stability_summary_path: Path,
) -> str:
    top_models = leaderboard.head(3)
    transformer_row = leaderboard.loc[leaderboard["model_name"] == "transformer_walkforward"]
    lines = [
        "# Signal Hardening Guidance",
        "",
        "## Inputs",
        f"- last-year summary: `{last_year_summary_path.as_posix()}`",
        f"- repeated-window summary: `{stability_summary_path.as_posix()}`",
        "",
        "## Robustness leaderboard",
    ]
    for _, row in top_models.iterrows():
        lines.append(
            (
                f"- `{row['model_name']}`: tier `{row['priority_tier']}`, "
                f"average gap vs baseline `${row['stability_avg_gap_vs_baseline']:,.0f}`, "
                f"worst gap `${row['stability_worst_gap_vs_baseline']:,.0f}`, "
                f"beat rate `{row['stability_beat_rate']:.0%}`."
            )
        )

    lines.extend(
        [
            "",
            "## Degradation readout",
            (
                f"- Top repeated-window model: `{analysis['summary']['top_ranked_model']}` with average gap vs baseline "
                f"`${analysis['summary']['top_ranked_avg_gap_vs_baseline']:,.0f}`."
            ),
            "- Use repeated-window gap persistence and worst-window damage as the primary promotion gate.",
            "- Treat positive latest-window spread without repeated-window support as a challenger, not a promotion.",
        ]
    )

    if not transformer_row.empty:
        row = transformer_row.iloc[0]
        lines.extend(
            [
                "",
                "## Transformer lane",
                (
                    f"- Repeated-window transformer gap vs baseline averages `${row['stability_avg_gap_vs_baseline']:,.0f}` "
                    f"with a worst window of `${row['stability_worst_gap_vs_baseline']:,.0f}`."
                ),
                (
                    f"- Latest promoted report gap vs baseline is "
                    f"`{row['latest_report_gap_vs_baseline']:,.0f}` dollars; the delta versus the stability latest window is "
                    f"`{row['latest_report_gap_delta_vs_stability_latest']:,.0f}`."
                    if pd.notna(row["latest_report_gap_vs_baseline"])
                    and pd.notna(row["latest_report_gap_delta_vs_stability_latest"])
                    else "- Latest promoted report gap is unavailable."
                ),
                "- Keep `seq40 + clip_q95` for the headline latest-year view and `seq20 + clip_q95` for repeated-window control until a single setup wins both tests.",
            ]
        )

    lines.extend(
        [
            "",
            "## Recommended follow-up",
            "- Add the same degradation tracking to future promoted runs and block new model promotions when worst-window gap falls below the selected tolerance.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    last_year_summary_path = Path(args.last_year_summary)
    stability_summary_path = Path(args.stability_summary)
    leaderboard_output_path = Path(args.leaderboard_output)
    analysis_output_path = Path(args.analysis_output)
    guidance_output_path = Path(args.guidance_output)

    last_year_summary = load_report_frame(last_year_summary_path)
    stability_summary = load_report_frame(stability_summary_path)
    leaderboard = build_signal_hardening_leaderboard(stability_summary, last_year_summary)
    analysis = build_performance_degradation_analysis(leaderboard)
    guidance = build_signal_hardening_guidance(
        leaderboard,
        analysis,
        last_year_summary_path=last_year_summary_path,
        stability_summary_path=stability_summary_path,
    )

    leaderboard_output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(leaderboard_output_path, index=False)
    _write_json(analysis_output_path, analysis)
    _write_text(guidance_output_path, guidance)

    print(f"Saved leaderboard to: {leaderboard_output_path}")
    print(f"Saved performance degradation analysis to: {analysis_output_path}")
    print(f"Saved signal hardening guidance to: {guidance_output_path}")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
