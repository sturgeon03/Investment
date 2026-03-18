from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_eligibility_rules(eligibility_rules: dict[str, float | int] | None) -> tuple[float, float, int]:
    payload = eligibility_rules or {}
    min_close_price = float(payload.get("min_close_price", 0.0))
    min_dollar_volume_20 = float(payload.get("min_dollar_volume_20", 0.0))
    min_universe_age_days = int(payload.get("min_universe_age_days", 0))
    return min_close_price, min_dollar_volume_20, min_universe_age_days


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _rank_percentile(series: pd.Series) -> pd.Series:
    if series.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, method="average")


def _trend_share(series: pd.Series) -> float:
    valid = series.fillna(False).astype(float)
    return float(valid.mean()) if len(valid) else 0.0


def _apply_universe_snapshots(
    features: pd.DataFrame,
    universe_snapshots: pd.DataFrame | None,
) -> pd.DataFrame:
    if universe_snapshots is None or universe_snapshots.empty:
        return features

    filtered_parts: list[pd.DataFrame] = []
    snapshot_dates = sorted(pd.to_datetime(universe_snapshots["effective_date"]).dt.normalize().unique())
    normalized = features.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()

    for index, start_date in enumerate(snapshot_dates):
        end_date = snapshot_dates[index + 1] if index + 1 < len(snapshot_dates) else None
        active_tickers = set(
            universe_snapshots.loc[
                pd.to_datetime(universe_snapshots["effective_date"]).dt.normalize() == start_date,
                "ticker",
            ].astype(str)
        )
        if not active_tickers:
            continue

        date_mask = normalized["date"] >= start_date
        if end_date is not None:
            date_mask &= normalized["date"] < end_date
        filtered_parts.append(normalized.loc[date_mask & normalized["ticker"].isin(active_tickers)].copy())

    if not filtered_parts:
        return normalized.iloc[0:0].copy()

    return pd.concat(filtered_parts, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def _build_benchmark_feature_frame(benchmark_prices: pd.DataFrame) -> pd.DataFrame:
    benchmark = benchmark_prices.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    benchmark["benchmark_ret_1"] = benchmark["close"].pct_change()
    benchmark["benchmark_ret_20"] = benchmark["close"].pct_change(20)
    benchmark["benchmark_ret_60"] = benchmark["close"].pct_change(60)
    benchmark["benchmark_vol_20"] = benchmark["benchmark_ret_1"].rolling(20, min_periods=20).std()
    benchmark["benchmark_sma_50"] = benchmark["close"].rolling(50, min_periods=50).mean()
    benchmark["benchmark_sma_200"] = benchmark["close"].rolling(200, min_periods=200).mean()
    benchmark["benchmark_drawdown_60"] = (
        benchmark["close"] / benchmark["close"].rolling(60, min_periods=60).max() - 1.0
    )
    benchmark["benchmark_price_vs_sma200"] = benchmark["close"] / benchmark["benchmark_sma_200"] - 1.0
    benchmark["benchmark_sma50_vs_sma200"] = (
        benchmark["benchmark_sma_50"] / benchmark["benchmark_sma_200"] - 1.0
    )
    benchmark["market_trend_ok"] = (
        (benchmark["close"] > benchmark["benchmark_sma_50"])
        & (benchmark["benchmark_sma_50"] > benchmark["benchmark_sma_200"])
    )
    benchmark["benchmark_vol_20_mean_60"] = benchmark["benchmark_vol_20"].rolling(60, min_periods=60).mean()
    benchmark["market_high_vol_regime"] = benchmark["benchmark_vol_20"] > benchmark["benchmark_vol_20_mean_60"]

    return benchmark[
        [
            "date",
            "benchmark_ret_1",
            "benchmark_ret_20",
            "benchmark_ret_60",
            "benchmark_vol_20",
            "benchmark_drawdown_60",
            "benchmark_price_vs_sma200",
            "benchmark_sma50_vs_sma200",
            "market_trend_ok",
            "market_high_vol_regime",
        ]
    ]


def add_context_features(
    features: pd.DataFrame,
    eligibility_rules: dict[str, float | int] | None = None,
) -> pd.DataFrame:
    features = features.copy()
    benchmark_defaults = {
        "benchmark_ret_1": 0.0,
        "benchmark_ret_20": 0.0,
        "benchmark_ret_60": 0.0,
        "benchmark_vol_20": 0.0,
        "benchmark_drawdown_60": 0.0,
        "benchmark_price_vs_sma200": 0.0,
        "benchmark_sma50_vs_sma200": 0.0,
        "market_trend_ok": False,
        "market_high_vol_regime": False,
    }

    for column, default_value in benchmark_defaults.items():
        if column not in features.columns:
            features[column] = default_value
    features["rel_ret_20"] = features["ret_20"] - features["benchmark_ret_20"]
    features["rel_ret_60"] = features["ret_60"] - features["benchmark_ret_60"]
    features["benchmark_momentum_gap"] = features["benchmark_ret_20"] - features["benchmark_ret_60"]
    features["relative_momentum_gap"] = features["rel_ret_20"] - features["rel_ret_60"]
    features["first_universe_date"] = features.groupby("ticker")["date"].transform("min")
    features["universe_age_days"] = (
        pd.to_datetime(features["date"]).dt.normalize()
        - pd.to_datetime(features["first_universe_date"]).dt.normalize()
    ).dt.days.astype(float)

    min_close_price, min_dollar_volume_20, min_universe_age_days = _resolve_eligibility_rules(eligibility_rules)
    close_ok = features["close"].fillna(0.0) >= min_close_price
    dollar_volume_ok = features["dollar_volume_20"].fillna(0.0) >= min_dollar_volume_20
    age_ok = features["universe_age_days"].fillna(-1.0) >= float(min_universe_age_days)
    features["eligible_universe"] = close_ok & dollar_volume_ok & age_ok

    cross_sectional = features.groupby("date", group_keys=False)
    features["cs_ret_20_z"] = cross_sectional["ret_20"].transform(_cross_sectional_zscore)
    features["cs_ret_60_z"] = cross_sectional["ret_60"].transform(_cross_sectional_zscore)
    features["cs_rel_ret_20_z"] = cross_sectional["rel_ret_20"].transform(_cross_sectional_zscore)
    features["cs_vol_20_z"] = cross_sectional["vol_20"].transform(_cross_sectional_zscore)
    features["universe_momentum_rank_pct"] = cross_sectional["rel_ret_20"].transform(_rank_percentile)
    features["universe_vol_rank_pct"] = cross_sectional["vol_20"].transform(_rank_percentile)
    features["market_breadth_trend_share"] = cross_sectional["trend_ok"].transform(_trend_share)

    sector_grouped = features.groupby(["date", "sector"], group_keys=False)
    features["sector_size"] = sector_grouped["ticker"].transform("count")
    features["sector_ret_20_mean"] = sector_grouped["ret_20"].transform("mean")
    features["sector_rel_ret_20_mean"] = sector_grouped["rel_ret_20"].transform("mean")
    features["sector_vol_20_mean"] = sector_grouped["vol_20"].transform("mean")
    features["sector_ret_20_gap"] = features["ret_20"] - features["sector_ret_20_mean"]
    features["sector_rel_ret_20_gap"] = features["rel_ret_20"] - features["sector_rel_ret_20_mean"]
    features["sector_vol_20_gap"] = features["vol_20"] - features["sector_vol_20_mean"]
    features["sector_momentum_rank_pct"] = sector_grouped["rel_ret_20"].transform(_rank_percentile)
    features["sector_trend_share"] = sector_grouped["trend_ok"].transform(_trend_share)

    numeric_columns = features.select_dtypes(include=["number"]).columns
    features[numeric_columns] = features[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return features


def build_features(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
    ticker_metadata: pd.DataFrame | None = None,
    universe_snapshots: pd.DataFrame | None = None,
    eligibility_rules: dict[str, float | int] | None = None,
) -> pd.DataFrame:
    features = prices.sort_values(["ticker", "date"]).copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()

    if ticker_metadata is not None and not ticker_metadata.empty:
        metadata = ticker_metadata.copy()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
        features = features.merge(metadata, on="ticker", how="left")
    if "sector" not in features.columns:
        features["sector"] = "UNKNOWN"
    else:
        features["sector"] = features["sector"].fillna("UNKNOWN").astype(str)

    grouped = features.groupby("ticker", group_keys=False)
    close_grouped = grouped["close"]
    volume_grouped = grouped["volume"]

    features["ret_1"] = close_grouped.pct_change()
    features["ret_5"] = close_grouped.pct_change(5)
    features["ret_20"] = close_grouped.pct_change(20)
    features["ret_60"] = close_grouped.pct_change(60)
    features["ret_120"] = close_grouped.pct_change(120)
    features["vol_20"] = grouped["ret_1"].transform(lambda series: series.rolling(20, min_periods=20).std())
    features["vol_60"] = grouped["ret_1"].transform(lambda series: series.rolling(60, min_periods=60).std())
    features["sma_20"] = close_grouped.transform(lambda series: series.rolling(20, min_periods=20).mean())
    features["sma_50"] = close_grouped.transform(lambda series: series.rolling(50, min_periods=50).mean())
    features["sma_200"] = close_grouped.transform(lambda series: series.rolling(200, min_periods=200).mean())
    features["price_vs_sma20"] = features["close"] / features["sma_20"] - 1.0
    features["price_vs_sma50"] = features["close"] / features["sma_50"] - 1.0
    features["price_vs_sma200"] = features["close"] / features["sma_200"] - 1.0
    features["sma20_vs_sma50"] = features["sma_20"] / features["sma_50"] - 1.0
    features["sma50_vs_sma200"] = features["sma_50"] / features["sma_200"] - 1.0
    features["drawdown_20"] = close_grouped.transform(
        lambda series: series / series.rolling(20, min_periods=20).max() - 1.0
    )
    features["drawdown_60"] = close_grouped.transform(
        lambda series: series / series.rolling(60, min_periods=60).max() - 1.0
    )

    volume_mean_20 = volume_grouped.transform(lambda series: series.rolling(20, min_periods=20).mean())
    dollar_volume = features["close"] * features["volume"]
    dollar_volume_mean_20 = dollar_volume.groupby(features["ticker"]).transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    features["dollar_volume_20"] = dollar_volume_mean_20
    features["volume_ratio_20"] = features["volume"] / volume_mean_20 - 1.0
    features["log_dollar_volume_20"] = np.log1p(dollar_volume_mean_20)

    intraday_range = (features["high"] - features["low"]) / features["close"].replace(0.0, np.nan)
    features["range_pct_20"] = intraday_range.groupby(features["ticker"]).transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    features["vol_ratio_20_60"] = features["vol_20"] / features["vol_60"] - 1.0
    features["momentum_20_60_gap"] = features["ret_20"] - features["ret_60"]
    features["trend_ok"] = (features["close"] > features["sma_50"]) & (features["sma_50"] > features["sma_200"])

    benchmark_defaults = {
        "benchmark_ret_1": 0.0,
        "benchmark_ret_20": 0.0,
        "benchmark_ret_60": 0.0,
        "benchmark_vol_20": 0.0,
        "benchmark_drawdown_60": 0.0,
        "benchmark_price_vs_sma200": 0.0,
        "benchmark_sma50_vs_sma200": 0.0,
        "market_trend_ok": False,
        "market_high_vol_regime": False,
    }
    if benchmark_prices is not None and not benchmark_prices.empty:
        features = features.merge(_build_benchmark_feature_frame(benchmark_prices), on="date", how="left")
    else:
        for column, default_value in benchmark_defaults.items():
            features[column] = default_value

    features = _apply_universe_snapshots(features, universe_snapshots)
    return add_context_features(features, eligibility_rules)
