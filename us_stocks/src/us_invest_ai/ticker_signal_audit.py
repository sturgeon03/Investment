from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from us_invest_ai.signals import load_llm_scores


DEFAULT_FORWARD_HORIZONS = (5, 20, 60)


def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _empty_sections_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "title", "text"])


def load_price_history(path: str | Path, ticker: str) -> pd.DataFrame:
    prices = pd.read_csv(path)
    required = {"date", "ticker", "close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Price file is missing columns: {sorted(missing)}")

    normalized = prices.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    filtered = (
        normalized.loc[
            (normalized["ticker"] == ticker.upper()) & normalized["close"].notna(),
            ["date", "close"],
        ]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if filtered.empty:
        raise ValueError(f"No price history found for ticker={ticker!r} in {path}.")
    return filtered


def load_filtered_sections(
    path: str | Path,
    ticker: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        return _empty_sections_frame()

    sections = pd.read_csv(source_path)
    required = {"date", "ticker", "title", "text"}
    missing = required.difference(sections.columns)
    if missing:
        raise ValueError(f"Sections file is missing columns: {sorted(missing)}")

    start_ts = _normalize_date(start_date)
    end_ts = _normalize_date(end_date)
    normalized = sections.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    return (
        normalized.loc[
            (normalized["ticker"] == ticker.upper())
            & (normalized["date"] >= start_ts)
            & (normalized["date"] <= end_ts)
        ]
        .sort_values(["date", "title"])
        .reset_index(drop=True)
    )


def load_filtered_detailed_scores(
    path: str | Path,
    ticker: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    horizon_bucket: str,
) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        return pd.DataFrame()

    scores = pd.read_csv(source_path)
    required = {"date", "ticker", "adjusted_score", "effective_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Detailed score file is missing columns: {sorted(missing)}")

    start_ts = _normalize_date(start_date)
    end_ts = _normalize_date(end_date)
    normalized = scores.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    filtered = normalized.loc[
        (normalized["ticker"] == ticker.upper())
        & (normalized["date"] >= start_ts)
        & (normalized["date"] <= end_ts)
    ].copy()
    if "horizon_bucket" in normalized.columns:
        filtered = filtered.loc[filtered["horizon_bucket"].astype(str) == horizon_bucket].copy()
    return filtered.sort_values(["date", "title"]).reset_index(drop=True)


def _next_trading_date(trading_dates: pd.DatetimeIndex, signal_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    search_value = pd.Timestamp(signal_date).normalize()
    index = int(trading_dates.searchsorted(search_value, side="right"))
    if index >= len(trading_dates):
        return pd.NaT
    return pd.Timestamp(trading_dates[index]).normalize()


def build_signal_event_frame(
    scores: pd.DataFrame,
    price_history: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    forward_horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
) -> pd.DataFrame:
    output_columns = [
        "signal_date",
        "trade_date",
        "trade_close",
        "days_until_trade",
        "llm_score",
        "document_count",
        "section_count",
        "avg_confidence",
        "avg_risk_flag",
        *[
            column
            for horizon in forward_horizons
            for column in (f"forward_{horizon}d_date", f"forward_{horizon}d_return")
        ],
    ]
    start_ts = _normalize_date(start_date)
    end_ts = _normalize_date(end_date)
    expected_columns = [
        "date",
        "ticker",
        "llm_score",
        "document_count",
        "section_count",
        "avg_confidence",
        "avg_risk_flag",
    ]
    if scores.empty:
        return pd.DataFrame(columns=output_columns)

    missing = set(expected_columns).difference(scores.columns)
    if missing:
        raise ValueError(f"Aggregated score frame is missing columns: {sorted(missing)}")

    normalized = scores.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper()
    filtered = (
        normalized.loc[
            (normalized["date"] >= start_ts)
            & (normalized["date"] <= end_ts)
        ]
        .drop_duplicates(subset=["date", "ticker"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if filtered.empty:
        return pd.DataFrame(columns=output_columns)

    price_frame = price_history.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"]).dt.normalize()
    price_frame = price_frame.sort_values("date").reset_index(drop=True)
    trading_dates = pd.DatetimeIndex(price_frame["date"])
    close_by_date = price_frame.set_index("date")["close"]

    records: list[dict[str, object]] = []
    for row in filtered.itertuples(index=False):
        signal_date = pd.Timestamp(row.date).normalize()
        trade_date = _next_trading_date(trading_dates, signal_date)
        trade_close = float(close_by_date.loc[trade_date]) if pd.notna(trade_date) else np.nan
        record: dict[str, object] = {
            "signal_date": signal_date,
            "trade_date": trade_date,
            "trade_close": trade_close,
            "days_until_trade": (
                float((trade_date - signal_date).days)
                if pd.notna(trade_date)
                else np.nan
            ),
            "llm_score": float(row.llm_score),
            "document_count": int(row.document_count),
            "section_count": int(row.section_count),
            "avg_confidence": float(row.avg_confidence),
            "avg_risk_flag": float(row.avg_risk_flag),
        }

        if pd.isna(trade_date):
            for horizon in forward_horizons:
                record[f"forward_{horizon}d_date"] = pd.NaT
                record[f"forward_{horizon}d_return"] = np.nan
            records.append(record)
            continue

        entry_index = int(trading_dates.get_loc(trade_date))
        for horizon in forward_horizons:
            target_index = entry_index + int(horizon)
            if target_index >= len(trading_dates):
                record[f"forward_{horizon}d_date"] = pd.NaT
                record[f"forward_{horizon}d_return"] = np.nan
                continue
            future_date = pd.Timestamp(trading_dates[target_index]).normalize()
            future_close = float(close_by_date.loc[future_date])
            record[f"forward_{horizon}d_date"] = future_date
            record[f"forward_{horizon}d_return"] = future_close / trade_close - 1.0

        records.append(record)

    return pd.DataFrame(records, columns=output_columns)


def build_daily_signal_strategy(
    price_history: pd.DataFrame,
    signal_events: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    initial_capital: float = 100_000.0,
    long_threshold: float = 0.0,
) -> pd.DataFrame:
    start_ts = _normalize_date(start_date)
    end_ts = _normalize_date(end_date)
    daily = (
        price_history.loc[
            (pd.to_datetime(price_history["date"]).dt.normalize() >= start_ts)
            & (pd.to_datetime(price_history["date"]).dt.normalize() <= end_ts)
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    if daily.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "close",
                "daily_return",
                "target_position",
                "live_position",
                "strategy_return",
                "signal_strategy_value",
                "buy_hold_value",
            ]
        )

    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["daily_return"] = daily["close"].pct_change().fillna(0.0)
    daily["target_position"] = 0.0

    valid_events = (
        signal_events.dropna(subset=["trade_date"])
        .copy()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    for index, row in enumerate(valid_events.itertuples(index=False)):
        position = float(max(row.llm_score, 0.0)) if float(row.llm_score) > long_threshold else 0.0
        next_trade_date = (
            pd.Timestamp(valid_events.loc[index + 1, "trade_date"]).normalize()
            if index + 1 < len(valid_events)
            else None
        )
        mask = daily["date"] >= pd.Timestamp(row.trade_date).normalize()
        if next_trade_date is not None:
            mask &= daily["date"] < next_trade_date
        daily.loc[mask, "target_position"] = position

    daily["live_position"] = daily["target_position"].shift(1).fillna(0.0)
    daily["strategy_return"] = daily["live_position"] * daily["daily_return"]
    daily["signal_strategy_value"] = initial_capital * (1.0 + daily["strategy_return"]).cumprod()
    daily["buy_hold_value"] = initial_capital * (1.0 + daily["daily_return"]).cumprod()
    return daily


def _directional_hit_rate(signal_events: pd.DataFrame, horizon: int) -> float:
    column = f"forward_{horizon}d_return"
    valid = signal_events.loc[
        signal_events[column].notna()
        & signal_events["llm_score"].notna()
        & (signal_events["llm_score"] != 0.0)
    ].copy()
    if valid.empty:
        return np.nan
    return float((np.sign(valid["llm_score"]) == np.sign(valid[column])).mean())


def _correlation(signal_events: pd.DataFrame, horizon: int) -> float:
    column = f"forward_{horizon}d_return"
    valid = signal_events.loc[
        signal_events[column].notna()
        & signal_events["llm_score"].notna()
    , ["llm_score", column]]
    if len(valid) < 2:
        return np.nan
    return float(valid["llm_score"].corr(valid[column]))


def build_summary_frame(
    *,
    ticker: str,
    horizon_bucket: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    signal_events: pd.DataFrame,
    daily_strategy: pd.DataFrame,
    sections: pd.DataFrame,
    section_scores: pd.DataFrame,
    initial_capital: float,
    long_threshold: float,
    forward_horizons: Sequence[int] = DEFAULT_FORWARD_HORIZONS,
) -> pd.DataFrame:
    requested_start = _normalize_date(start_date).date().isoformat()
    requested_end = _normalize_date(end_date).date().isoformat()
    strategy_end_value = (
        float(daily_strategy["signal_strategy_value"].iloc[-1])
        if not daily_strategy.empty
        else float(initial_capital)
    )
    buy_hold_end_value = (
        float(daily_strategy["buy_hold_value"].iloc[-1])
        if not daily_strategy.empty
        else float(initial_capital)
    )
    summary: dict[str, object] = {
        "ticker": ticker.upper(),
        "horizon_bucket": horizon_bucket,
        "requested_start_date": requested_start,
        "requested_end_date": requested_end,
        "price_start_date": (
            pd.Timestamp(daily_strategy["date"].iloc[0]).date().isoformat()
            if not daily_strategy.empty
            else None
        ),
        "price_end_date": (
            pd.Timestamp(daily_strategy["date"].iloc[-1]).date().isoformat()
            if not daily_strategy.empty
            else None
        ),
        "section_count": int(len(sections)),
        "scored_section_count": int(len(section_scores)),
        "signal_event_count": int(len(signal_events)),
        "avg_llm_score": float(signal_events["llm_score"].mean()) if not signal_events.empty else np.nan,
        "avg_confidence": (
            float(signal_events["avg_confidence"].mean()) if not signal_events.empty else np.nan
        ),
        "avg_risk_flag": (
            float(signal_events["avg_risk_flag"].mean()) if not signal_events.empty else np.nan
        ),
        "initial_capital": float(initial_capital),
        "buy_hold_end_capital": buy_hold_end_value,
        "buy_hold_total_return": buy_hold_end_value / float(initial_capital) - 1.0,
        "signal_strategy_end_capital": strategy_end_value,
        "signal_strategy_total_return": strategy_end_value / float(initial_capital) - 1.0,
        "strategy_rule": "next_trading_day_after_signal_then_long_only_weight=max(llm_score,0)",
        "long_threshold": float(long_threshold),
    }
    for horizon in forward_horizons:
        summary[f"llm_forward_corr_{horizon}d"] = _correlation(signal_events, horizon)
        summary[f"directional_hit_rate_{horizon}d"] = _directional_hit_rate(signal_events, horizon)

    return pd.DataFrame([summary])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one ticker's document signals against later price returns. "
            "This is a single-name signal audit, not the main cross-sectional portfolio backtest."
        )
    )
    parser.add_argument("--ticker", required=True, help="Ticker to audit.")
    parser.add_argument("--start-date", default="2025-01-01", help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-12-31", help="End date in YYYY-MM-DD.")
    parser.add_argument("--horizon-bucket", default="swing", help="Signal horizon bucket to audit.")
    parser.add_argument(
        "--prices-csv",
        default="us_stocks/data/raw/prices.csv",
        help="Price CSV with columns date,ticker,close.",
    )
    parser.add_argument(
        "--sections-csv",
        default="us_stocks/documents/sec_sections.csv",
        help="Section-level documents CSV.",
    )
    parser.add_argument(
        "--scores-csv",
        default="us_stocks/signals/llm_scores.generated.csv",
        help="Aggregated signal CSV.",
    )
    parser.add_argument(
        "--detailed-scores-csv",
        default="us_stocks/signals/sec_llm_document_scores.csv",
        help="Optional per-section score CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="us_stocks/artifacts/ticker_signal_audit",
        help="Directory where filtered sections, events, daily strategy, and summary are saved.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital used for the naive audit strategy.",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.0,
        help="Only scores above this threshold open a long position in the naive audit strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ticker = args.ticker.upper()
    output_dir = Path(args.output_dir) / f"{ticker.lower()}_{args.start_date}_{args.end_date}_{args.horizon_bucket}"
    output_dir.mkdir(parents=True, exist_ok=True)

    price_history = load_price_history(args.prices_csv, ticker)
    aggregated_scores = load_llm_scores(Path(args.scores_csv), args.horizon_bucket)
    ticker_scores = aggregated_scores.loc[aggregated_scores["ticker"] == ticker].copy()
    sections = load_filtered_sections(args.sections_csv, ticker, args.start_date, args.end_date)
    section_scores = load_filtered_detailed_scores(
        args.detailed_scores_csv,
        ticker,
        args.start_date,
        args.end_date,
        args.horizon_bucket,
    )
    signal_events = build_signal_event_frame(
        ticker_scores,
        price_history,
        args.start_date,
        args.end_date,
    )
    daily_strategy = build_daily_signal_strategy(
        price_history,
        signal_events,
        args.start_date,
        args.end_date,
        initial_capital=args.initial_capital,
        long_threshold=args.long_threshold,
    )
    summary = build_summary_frame(
        ticker=ticker,
        horizon_bucket=args.horizon_bucket,
        start_date=args.start_date,
        end_date=args.end_date,
        signal_events=signal_events,
        daily_strategy=daily_strategy,
        sections=sections,
        section_scores=section_scores,
        initial_capital=args.initial_capital,
        long_threshold=args.long_threshold,
    )

    sections.to_csv(output_dir / "sections.csv", index=False)
    if not section_scores.empty:
        section_scores.to_csv(output_dir / "section_scores.csv", index=False)
    signal_events.to_csv(output_dir / "signal_events.csv", index=False)
    daily_strategy.to_csv(output_dir / "daily_signal_strategy.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved audit outputs to: {output_dir}")
    if signal_events.empty:
        print("No signal events were found in the requested window.")


if __name__ == "__main__":
    main()
