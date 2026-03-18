from __future__ import annotations

import json

import pandas as pd


HORIZON_TO_FORWARD_RETURN = {
    "short_term": "next_5d_return",
    "swing": "next_20d_return",
    "long_term": "next_60d_return",
}


def _best_realized_horizon_label(row: pd.Series) -> str:
    mapping = {
        "short_term": row.get("next_5d_return"),
        "swing": row.get("next_20d_return"),
        "long_term": row.get("next_60d_return"),
    }
    valid = {label: value for label, value in mapping.items() if pd.notna(value)}
    if not valid:
        return ""
    return max(valid, key=valid.get)


def _serialize_price_features(row: pd.Series) -> str:
    payload = {
        "ret_1": float(row["ret_1"]) if pd.notna(row["ret_1"]) else None,
        "ret_5": float(row["ret_5"]) if pd.notna(row.get("ret_5")) else None,
        "ret_20": float(row["ret_20"]) if pd.notna(row["ret_20"]) else None,
        "ret_60": float(row["ret_60"]) if pd.notna(row["ret_60"]) else None,
        "rel_ret_20": float(row["rel_ret_20"]) if pd.notna(row.get("rel_ret_20")) else None,
        "rel_ret_60": float(row["rel_ret_60"]) if pd.notna(row.get("rel_ret_60")) else None,
        "vol_20": float(row["vol_20"]) if pd.notna(row["vol_20"]) else None,
        "vol_60": float(row["vol_60"]) if pd.notna(row.get("vol_60")) else None,
        "price_vs_sma50": float(row["price_vs_sma50"]) if pd.notna(row.get("price_vs_sma50")) else None,
        "benchmark_ret_20": float(row["benchmark_ret_20"]) if pd.notna(row.get("benchmark_ret_20")) else None,
        "benchmark_vol_20": float(row["benchmark_vol_20"]) if pd.notna(row.get("benchmark_vol_20")) else None,
        "trend_ok": bool(row["trend_ok"]) if pd.notna(row["trend_ok"]) else None,
        "market_trend_ok": bool(row["market_trend_ok"]) if pd.notna(row.get("market_trend_ok")) else None,
    }
    return json.dumps(payload, sort_keys=True)


def build_router_training_frame(features: pd.DataFrame, llm_scores: pd.DataFrame) -> pd.DataFrame:
    if llm_scores.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "horizon_bucket",
                "llm_score",
                "price_features",
                "next_5d_return",
                "next_20d_return",
                "next_60d_return",
                "best_realized_horizon_label",
            ]
        )

    feature_frame = features.sort_values(["ticker", "date"]).copy()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"]).dt.normalize().astype("datetime64[ns]")
    grouped = feature_frame.groupby("ticker", group_keys=False)
    feature_frame["next_5d_return"] = grouped["close"].shift(-5) / feature_frame["close"] - 1.0
    feature_frame["next_20d_return"] = grouped["close"].shift(-20) / feature_frame["close"] - 1.0
    feature_frame["next_60d_return"] = grouped["close"].shift(-60) / feature_frame["close"] - 1.0

    scores = llm_scores.copy()
    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize().astype("datetime64[ns]")
    scores["ticker"] = scores["ticker"].str.upper()
    if "horizon_bucket" not in scores.columns:
        scores["horizon_bucket"] = "swing"

    rows: list[pd.DataFrame] = []
    for (ticker, horizon_bucket), score_group in scores.groupby(["ticker", "horizon_bucket"], group_keys=False):
        feature_group = feature_frame.loc[feature_frame["ticker"] == ticker].copy()
        if feature_group.empty:
            continue
        desired_columns = [
            "date",
            "ret_1",
            "ret_5",
            "ret_20",
            "ret_60",
            "rel_ret_20",
            "rel_ret_60",
            "vol_20",
            "vol_60",
            "price_vs_sma50",
            "benchmark_ret_20",
            "benchmark_vol_20",
            "trend_ok",
            "market_trend_ok",
            "next_5d_return",
            "next_20d_return",
            "next_60d_return",
        ]
        for column in desired_columns:
            if column not in feature_group.columns:
                feature_group[column] = pd.NA
        aligned = pd.merge_asof(
            score_group.sort_values("date"),
            feature_group[desired_columns].sort_values("date"),
            on="date",
            direction="backward",
        )
        aligned["ticker"] = ticker
        aligned["horizon_bucket"] = horizon_bucket
        aligned["price_features"] = aligned.apply(_serialize_price_features, axis=1)
        aligned["best_realized_horizon_label"] = aligned.apply(_best_realized_horizon_label, axis=1)
        rows.append(aligned)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "horizon_bucket",
                "llm_score",
                "price_features",
                "next_5d_return",
                "next_20d_return",
                "next_60d_return",
                "best_realized_horizon_label",
            ]
        )

    dataset = pd.concat(rows, ignore_index=True)
    return dataset[
        [
            "date",
            "ticker",
            "horizon_bucket",
            "llm_score",
            "price_features",
            "next_5d_return",
            "next_20d_return",
            "next_60d_return",
            "best_realized_horizon_label",
        ]
    ].sort_values(["date", "ticker", "horizon_bucket"]).reset_index(drop=True)
