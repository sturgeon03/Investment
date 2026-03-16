from __future__ import annotations

import pandas as pd


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    features = prices.sort_values(["ticker", "date"]).copy()
    grouped = features.groupby("ticker", group_keys=False)

    features["ret_1"] = grouped["close"].pct_change()
    features["ret_20"] = grouped["close"].pct_change(20)
    features["ret_60"] = grouped["close"].pct_change(60)
    features["vol_20"] = grouped["ret_1"].transform(
        lambda series: series.rolling(20, min_periods=20).std()
    )
    features["sma_50"] = grouped["close"].transform(
        lambda series: series.rolling(50, min_periods=50).mean()
    )
    features["sma_200"] = grouped["close"].transform(
        lambda series: series.rolling(200, min_periods=200).mean()
    )
    features["trend_ok"] = (
        (features["close"] > features["sma_50"])
        & (features["sma_50"] > features["sma_200"])
    )

    return features
