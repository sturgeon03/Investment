from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


DEFAULT_SERIES_COLORS = ["#1d4ed8", "#0f766e", "#7c3aed", "#db2777", "#b45309", "#475569"]
DEFAULT_SELECTED_METRIC_COLUMNS: dict[str, str] = {
    "avg_train_sample_count": "train_sample_count",
    "avg_validation_sample_count": "validation_sample_count",
    "avg_validation_mse": "validation_mse",
}


def build_value_curve(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    initial_capital: float,
) -> pd.DataFrame:
    strategy_growth = (1.0 + daily_returns).cumprod()
    strategy_value = initial_capital * (strategy_growth / strategy_growth.iloc[0])
    frame = pd.DataFrame({"date": strategy_value.index, "strategy_value": strategy_value.to_numpy()})
    if benchmark_returns is not None:
        benchmark_growth = (1.0 + benchmark_returns).cumprod()
        benchmark_value = initial_capital * (benchmark_growth / benchmark_growth.iloc[0])
        frame["benchmark_value"] = benchmark_value.reindex(strategy_value.index).to_numpy()
    return frame


def compute_signal_metrics(ranking_history: pd.DataFrame) -> tuple[float, float]:
    if ranking_history.empty or "llm_score" not in ranking_history.columns:
        return 0.0, 0.0

    llm_values = ranking_history["llm_score"].fillna(0.0).astype(float)
    signal_coverage = float((llm_values.abs() > 0).mean()) if len(llm_values) else 0.0
    avg_llm_abs_score = float(llm_values.abs().mean()) if len(llm_values) else 0.0
    return signal_coverage, avg_llm_abs_score


def slice_history_window(
    history: pd.DataFrame,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if history.empty or "date" not in history.columns:
        return history.copy()

    dates = pd.to_datetime(history["date"]).dt.normalize()
    mask = pd.Series(True, index=history.index)
    if eval_start is not None:
        mask &= dates >= pd.Timestamp(eval_start).normalize()
    if eval_end is not None:
        mask &= dates <= pd.Timestamp(eval_end).normalize()
    return history.loc[mask].copy()


def summarize_selected_history_metrics(
    history: pd.DataFrame,
    metric_columns: dict[str, str] | None = None,
) -> dict[str, float]:
    columns = metric_columns or DEFAULT_SELECTED_METRIC_COLUMNS
    if history.empty or "selected" not in history.columns:
        selected = pd.DataFrame(columns=history.columns)
    else:
        selected = history.loc[history["selected"].fillna(False).astype(bool)]

    return {
        output_column: (
            float(selected[source_column].mean())
            if not selected.empty and source_column in selected.columns
            else float("nan")
        )
        for output_column, source_column in columns.items()
    }


def build_evaluation_row(
    model_name: str,
    summary: pd.DataFrame,
    history: pd.DataFrame,
    curve: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    initial_capital: float,
    *,
    window_label: str | None = None,
    include_rebalance_count: bool = False,
    extra: dict[str, object] | None = None,
    metric_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    row = summary.copy()
    row.insert(0, "model_name", model_name)
    if window_label is not None:
        row.insert(1, "window_label", str(window_label))
    row["eval_start"] = pd.Timestamp(eval_start).date().isoformat()
    row["eval_end"] = pd.Timestamp(eval_end).date().isoformat()
    row["starting_capital"] = initial_capital
    ending_capital = float(curve["strategy_value"].iloc[-1])
    row["ending_capital"] = ending_capital
    row["profit_dollars"] = float(ending_capital - initial_capital)

    history_slice = slice_history_window(history, eval_start, eval_end)
    if include_rebalance_count:
        row["rebalance_count"] = int(history_slice["date"].nunique()) if not history_slice.empty else 0
    for key, value in summarize_selected_history_metrics(history_slice, metric_columns=metric_columns).items():
        row[key] = value
    if extra:
        for key, value in extra.items():
            row[key] = value
    return row


def build_svg_chart(
    curves: pd.DataFrame,
    benchmark_name: str,
    output_path: Path,
    *,
    chart_title: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    height = 720
    left = 90
    right = 220
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    date_values = pd.to_datetime(curves["date"])
    x_values = (date_values - date_values.min()).dt.days.astype(float)
    x_range = max(float(x_values.max()), 1.0)

    numeric_columns = [column for column in curves.columns if column != "date"]
    y_min = float(curves[numeric_columns].min().min())
    y_max = float(curves[numeric_columns].max().max())
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_padding = max((y_max - y_min) * 0.08, 1.0)
    y_min -= y_padding
    y_max += y_padding

    def project_x(value: float) -> float:
        return left + (value / x_range) * plot_width

    def project_y(value: float) -> float:
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_height

    series_markup: list[str] = []
    legend_markup: list[str] = []
    for index, column in enumerate(numeric_columns):
        points = " ".join(
            f"{project_x(float(x)):.2f},{project_y(float(y)):.2f}"
            for x, y in zip(x_values, curves[column], strict=True)
        )
        color = DEFAULT_SERIES_COLORS[index % len(DEFAULT_SERIES_COLORS)]
        stroke_width = 3 if column == benchmark_name else 2.5
        series_markup.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" points="{points}" />'
        )
        legend_y = top + 24 + index * 28
        label = column.replace("_value", "")
        legend_markup.append(
            f'<line x1="{width - right + 20}" y1="{legend_y}" x2="{width - right + 52}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="{stroke_width}" />'
            f'<text x="{width - right + 62}" y="{legend_y + 5}" font-size="16" fill="#0f172a">{label}</text>'
        )

    y_ticks = np.linspace(y_min, y_max, 5)
    y_axis_markup = []
    for value in y_ticks:
        y = project_y(float(value))
        y_axis_markup.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1" />'
            f'<text x="{left - 12}" y="{y + 5:.2f}" font-size="14" text-anchor="end" fill="#475569">{_format_currency(float(value))}</text>'
        )

    x_labels = [
        (date_values.min(), date_values.min().strftime("%Y-%m-%d")),
        (date_values.iloc[len(date_values) // 2], date_values.iloc[len(date_values) // 2].strftime("%Y-%m-%d")),
        (date_values.max(), date_values.max().strftime("%Y-%m-%d")),
    ]
    x_axis_markup = []
    for date_value, label in x_labels:
        offset = float((date_value - date_values.min()).days)
        x = project_x(offset)
        x_axis_markup.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f1f5f9" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{top + plot_height + 28}" font-size="14" text-anchor="middle" fill="#475569">{label}</text>'
        )

    title = chart_title or f"Strategy Value - Last {len(curves)} Trading Days"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f8fafc" />
<text x="{left}" y="32" font-size="28" font-family="Segoe UI, Arial, sans-serif" fill="#0f172a">{title}</text>
<text x="{left}" y="54" font-size="15" font-family="Segoe UI, Arial, sans-serif" fill="#475569">Initial capital assumed: {_format_currency(curves[numeric_columns[0]].iloc[0])}</text>
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" />
{''.join(y_axis_markup)}
{''.join(x_axis_markup)}
{''.join(series_markup)}
{''.join(legend_markup)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"
