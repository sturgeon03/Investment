from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_llm_scores(path: Path, horizon_bucket: str | None = "swing") -> pd.DataFrame:
    scores = pd.read_csv(path)
    required = {"date", "ticker", "llm_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"LLM score file is missing columns: {sorted(missing)}")

    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize().astype("datetime64[ns]")
    scores["ticker"] = scores["ticker"].str.upper()
    scores["llm_score"] = pd.to_numeric(scores["llm_score"], errors="coerce")

    if "horizon_bucket" in scores.columns and horizon_bucket is not None:
        scores["horizon_bucket"] = scores["horizon_bucket"].fillna("").astype(str)
        filtered = scores.loc[scores["horizon_bucket"] == horizon_bucket].copy()
        if filtered.empty:
            raise ValueError(
                f"No LLM scores found for horizon_bucket={horizon_bucket!r} in {path}."
            )
        scores = filtered

    return scores.dropna(subset=["date", "ticker", "llm_score"]).reset_index(drop=True)


def attach_llm_scores(features: pd.DataFrame, llm_scores: pd.DataFrame) -> pd.DataFrame:
    scored = features.sort_values(["ticker", "date"]).copy()
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize().astype("datetime64[ns]")
    llm_scores = llm_scores.sort_values(["ticker", "date"]).copy()
    llm_scores["date"] = pd.to_datetime(llm_scores["date"]).dt.normalize().astype("datetime64[ns]")

    metadata_columns = [
        column
        for column in llm_scores.columns
        if column not in {"date", "ticker"}
    ]
    numeric_metadata = [
        column
        for column in metadata_columns
        if pd.api.types.is_numeric_dtype(llm_scores[column])
    ]
    non_numeric_metadata = [column for column in metadata_columns if column not in numeric_metadata]

    merged_frames: list[pd.DataFrame] = []
    for ticker, feature_group in scored.groupby("ticker", group_keys=False):
        signal_group = llm_scores.loc[llm_scores["ticker"] == ticker, ["date", *metadata_columns]]
        if signal_group.empty:
            merged = feature_group.copy()
            merged["llm_score"] = 0.0
            for column in numeric_metadata:
                if column != "llm_score":
                    merged[column] = 0.0
            for column in non_numeric_metadata:
                merged[column] = ""
        else:
            merged = pd.merge_asof(
                feature_group.sort_values("date"),
                signal_group.sort_values("date"),
                on="date",
                direction="backward",
            )
            merged["ticker"] = ticker
            for column in numeric_metadata:
                merged[column] = merged[column].fillna(0.0)
            for column in non_numeric_metadata:
                merged[column] = merged[column].fillna("")
        merged_frames.append(merged)

    return pd.concat(merged_frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(
        drop=True
    )
