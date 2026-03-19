from __future__ import annotations

import numpy as np
import pandas as pd


def build_kr_feature_frame(
    prices: pd.DataFrame,
    filings: pd.DataFrame | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if prices.empty:
        return _empty_feature_frame()

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    frame["ret_1"] = frame.groupby("ticker")["close"].pct_change(1)
    frame["ret_5"] = frame.groupby("ticker")["close"].pct_change(5)
    frame["ret_20"] = frame.groupby("ticker")["close"].pct_change(20)
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    frame["avg_dollar_volume_20"] = (
        frame.groupby("ticker")["dollar_volume"].transform(lambda series: series.rolling(20, min_periods=1).mean())
    )
    frame["vol_20"] = frame.groupby("ticker")["ret_1"].transform(
        lambda series: series.rolling(20, min_periods=5).std()
    )
    frame["intraday_range"] = np.where(
        frame["close"].replace(0.0, np.nan).notna(),
        (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan),
        np.nan,
    )
    frame["range_20"] = frame.groupby("ticker")["intraday_range"].transform(
        lambda series: series.rolling(20, min_periods=1).mean()
    )

    benchmark_signals = _build_daily_benchmark_signals(benchmark_prices)
    frame = frame.merge(benchmark_signals, on="date", how="left")
    for column, default_value in (
        ("benchmark_close", np.nan),
        ("benchmark_ret_1", 0.0),
        ("benchmark_ret_5", 0.0),
        ("benchmark_ret_20", 0.0),
        ("benchmark_vol_20", 0.0),
        ("market_trend_ok", 1.0),
    ):
        frame[column] = frame[column].fillna(default_value)
    frame["rel_ret_1"] = frame["ret_1"].fillna(0.0) - frame["benchmark_ret_1"]
    frame["rel_ret_5"] = frame["ret_5"].fillna(0.0) - frame["benchmark_ret_5"]
    frame["rel_ret_20"] = frame["ret_20"].fillna(0.0) - frame["benchmark_ret_20"]

    filing_signals = _build_daily_filing_signals(filings)
    frame = frame.merge(filing_signals, on=["date", "ticker"], how="left")
    for column in ("filing_count", "earnings_filing_count", "capital_event_count", "governance_filing_count"):
        frame[column] = frame[column].fillna(0.0)

    frame["filing_count_20"] = frame.groupby("ticker")["filing_count"].transform(
        lambda series: series.rolling(20, min_periods=1).sum()
    )
    frame["earnings_filing_count_60"] = frame.groupby("ticker")["earnings_filing_count"].transform(
        lambda series: series.rolling(60, min_periods=1).sum()
    )
    frame["capital_event_count_60"] = frame.groupby("ticker")["capital_event_count"].transform(
        lambda series: series.rolling(60, min_periods=1).sum()
    )
    frame["days_since_last_filing"] = _days_since_last_event(frame, event_column="filing_count")

    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def _build_daily_benchmark_signals(benchmark_prices: pd.DataFrame | None) -> pd.DataFrame:
    if benchmark_prices is None or benchmark_prices.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "benchmark_close",
                "benchmark_ret_1",
                "benchmark_ret_5",
                "benchmark_ret_20",
                "benchmark_vol_20",
                "market_trend_ok",
            ]
        )

    benchmark = benchmark_prices.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    benchmark["benchmark_ret_1"] = benchmark["close"].pct_change(1)
    benchmark["benchmark_ret_5"] = benchmark["close"].pct_change(5)
    benchmark["benchmark_ret_20"] = benchmark["close"].pct_change(20)
    benchmark["benchmark_vol_20"] = benchmark["benchmark_ret_1"].rolling(20, min_periods=5).std()
    benchmark["benchmark_sma_20"] = benchmark["close"].rolling(20, min_periods=1).mean()
    benchmark["market_trend_ok"] = (
        benchmark["close"].ge(benchmark["benchmark_sma_20"]).fillna(True).astype(float)
    )
    return benchmark.rename(columns={"close": "benchmark_close"})[
        [
            "date",
            "benchmark_close",
            "benchmark_ret_1",
            "benchmark_ret_5",
            "benchmark_ret_20",
            "benchmark_vol_20",
            "market_trend_ok",
        ]
    ]


def _build_daily_filing_signals(filings: pd.DataFrame | None) -> pd.DataFrame:
    if filings is None or filings.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "filing_count",
                "earnings_filing_count",
                "capital_event_count",
                "governance_filing_count",
            ]
        )

    filing_frame = filings.copy()
    filing_frame["date"] = pd.to_datetime(filing_frame["session_date"]).dt.normalize()
    filing_frame["ticker"] = filing_frame["ticker"].astype(str).str.upper()
    filing_frame["category"] = filing_frame["category"].astype(str).str.lower()

    grouped = (
        filing_frame.groupby(["date", "ticker"], as_index=False)
        .agg(
            filing_count=("category", "size"),
            earnings_filing_count=("category", lambda series: float((series == "earnings").sum())),
            capital_event_count=("category", lambda series: float((series == "capital_event").sum())),
            governance_filing_count=("category", lambda series: float((series == "governance").sum())),
        )
        .reset_index(drop=True)
    )
    return grouped


def _days_since_last_event(frame: pd.DataFrame, *, event_column: str) -> pd.Series:
    last_event_dates = frame["date"].where(frame[event_column] > 0)
    last_event_dates = last_event_dates.groupby(frame["ticker"]).ffill()
    deltas = frame["date"].sub(last_event_dates).dt.days.astype(float)
    return deltas.fillna(9_999.0)


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "listing_code",
            "vendor_suffix",
            "provider_symbol",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "provider",
            "currency",
            "exchange_timezone",
            "ret_1",
            "ret_5",
            "ret_20",
            "dollar_volume",
            "avg_dollar_volume_20",
            "vol_20",
            "intraday_range",
            "range_20",
            "benchmark_close",
            "benchmark_ret_1",
            "benchmark_ret_5",
            "benchmark_ret_20",
            "benchmark_vol_20",
            "market_trend_ok",
            "rel_ret_1",
            "rel_ret_5",
            "rel_ret_20",
            "filing_count",
            "earnings_filing_count",
            "capital_event_count",
            "governance_filing_count",
            "filing_count_20",
            "earnings_filing_count_60",
            "capital_event_count_60",
            "days_since_last_filing",
        ]
    )
