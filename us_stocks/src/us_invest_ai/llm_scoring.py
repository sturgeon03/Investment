from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


REQUIRED_DOCUMENT_COLUMNS = {"date", "ticker", "title", "text"}
HORIZON_BUCKETS = ("short_term", "swing", "long_term")
SECTION_TYPE_WEIGHTS = {
    "item_2_02": 1.6,
    "item_7_01": 1.1,
    "item_8_01": 0.7,
    "mda": 1.0,
    "forward_guidance": 1.5,
    "liquidity": 0.9,
    "risk_factors": 0.5,
}
BOILERPLATE_MARKERS = (
    "forward-looking statements",
    "private securities litigation reform act",
    "assumes no obligation to revise or update",
    "see accompanying notes",
    "this item and other sections of this quarterly report",
    "this item and other sections of this annual report",
)
SYSTEM_PROMPT = """You are an equity-research signal extractor.
Read the document and estimate its likely stock impact across three horizons.
Return JSON only. Do not include markdown fences or any extra text.

The JSON schema must be:
{
  "short_term": {
    "score": float from -1.0 to 1.0,
    "confidence": float from 0.0 to 1.0,
    "risk_flag": float from 0.0 to 1.0,
    "summary": string up to 25 words
  },
  "swing": {
    "score": float from -1.0 to 1.0,
    "confidence": float from 0.0 to 1.0,
    "risk_flag": float from 0.0 to 1.0,
    "summary": string up to 25 words
  },
  "long_term": {
    "score": float from -1.0 to 1.0,
    "confidence": float from 0.0 to 1.0,
    "risk_flag": float from 0.0 to 1.0,
    "summary": string up to 25 words
  }
}
"""


@dataclass(slots=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: int = 60
    temperature: float = 0.0


def load_documents(path: str | Path) -> pd.DataFrame:
    documents = pd.read_csv(path)
    missing = REQUIRED_DOCUMENT_COLUMNS.difference(documents.columns)
    if missing:
        raise ValueError(f"Input document file is missing columns: {sorted(missing)}")

    documents = documents.copy()
    documents["date"] = pd.to_datetime(documents["date"]).dt.normalize()
    documents["ticker"] = documents["ticker"].str.upper()
    if "doc_type" not in documents.columns:
        documents["doc_type"] = "document"
    else:
        documents["doc_type"] = documents["doc_type"].fillna("document")

    if "source" not in documents.columns:
        documents["source"] = "unknown"
    else:
        documents["source"] = documents["source"].fillna("unknown")

    documents["title"] = documents["title"].fillna("").astype(str)
    documents["text"] = documents["text"].fillna("").astype(str)
    return documents


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model response did not contain JSON: {text}")
    return json.loads(match.group(0))


def normalize_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    score = _clamp(float(payload["score"]), -1.0, 1.0)
    confidence = _clamp(float(payload["confidence"]), 0.0, 1.0)
    risk_flag = _clamp(float(payload["risk_flag"]), 0.0, 1.0)
    adjusted_score = _clamp(score * (1.0 - 0.50 * risk_flag), -1.0, 1.0)
    effective_score = adjusted_score * max(confidence, 0.05)
    summary = str(payload.get("summary", "")).strip()[:240]

    return {
        "score": score,
        "confidence": confidence,
        "risk_flag": risk_flag,
        "adjusted_score": adjusted_score,
        "effective_score": effective_score,
        "summary": summary,
    }


def normalize_multi_horizon_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "score" in payload and "confidence" in payload and "risk_flag" in payload:
        normalized_single = normalize_score_payload(payload)
        return {bucket: normalized_single for bucket in HORIZON_BUCKETS}

    normalized: dict[str, dict[str, Any]] = {}
    for bucket in HORIZON_BUCKETS:
        bucket_payload = payload.get(bucket)
        if not isinstance(bucket_payload, dict):
            raise ValueError(f"Model response is missing horizon bucket: {bucket}")
        normalized[bucket] = normalize_score_payload(bucket_payload)
    return normalized


def _document_date_iso(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.normalize().date().isoformat()
    normalized = pd.to_datetime(value).normalize()
    return normalized.date().isoformat()


def build_messages(document: pd.Series) -> list[dict[str, str]]:
    metadata_lines = [
        f"Ticker: {document['ticker']}",
        f"Date: {_document_date_iso(document['date'])}",
        f"Document type: {document.get('doc_type', 'document')}",
        f"Source: {document.get('source', 'unknown')}",
    ]

    for key, label in (
        ("form", "Form"),
        ("section_type", "Section type"),
        ("items", "Items"),
        ("company_name", "Company"),
        ("source_url", "Source URL"),
    ):
        if key in document and pd.notna(document[key]) and str(document[key]).strip():
            metadata_lines.append(f"{label}: {document[key]}")

    user_prompt = (
        "\n".join(metadata_lines)
        + f"\nTitle: {str(document['title']).strip()}\n\nDocument text:\n{str(document['text']).strip()}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _sanitize_text_for_scoring(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    filtered = [
        sentence
        for sentence in sentences
        if sentence.strip()
        and not any(marker in sentence.lower() for marker in BOILERPLATE_MARKERS)
    ]
    sanitized = " ".join(filtered).strip()
    return sanitized or text


class HeuristicScorer:
    positive_keywords = {
        "strong",
        "improved",
        "healthy",
        "accelerated",
        "reaccelerated",
        "durable",
        "stable",
        "growth",
        "backlog",
        "renewal",
        "firm",
        "constructive",
        "supports",
        "resilient",
        "guidance",
        "outlook",
        "higher",
        "expansion",
    }
    negative_keywords = {
        "weakened",
        "slower",
        "soft",
        "softness",
        "decline",
        "cut",
        "cautious",
        "mixed",
        "pressure",
        "scrutiny",
        "antitrust",
        "constraints",
        "compliance",
        "lower",
        "miss",
        "headwind",
    }
    risk_keywords = {
        "regulator",
        "regulators",
        "lawsuit",
        "investigation",
        "antitrust",
        "scrutiny",
        "compliance",
        "uncertain",
        "risk",
        "volatility",
        "litigation",
    }

    def _base_signal(self, document: pd.Series) -> tuple[float, float, float]:
        text = _sanitize_text_for_scoring(f"{document.get('title', '')} {document.get('text', '')}").lower()
        positive_hits = sum(keyword in text for keyword in self.positive_keywords)
        negative_hits = sum(keyword in text for keyword in self.negative_keywords)
        risk_hits = sum(keyword in text for keyword in self.risk_keywords)

        raw_balance = positive_hits - negative_hits
        score = _clamp(raw_balance / 4.0, -1.0, 1.0)
        confidence = _clamp(0.35 + 0.08 * (positive_hits + negative_hits + risk_hits), 0.35, 0.95)
        risk_flag = _clamp(0.10 + 0.10 * risk_hits, 0.0, 1.0)
        return score, confidence, risk_flag

    def score_document(self, document: pd.Series) -> dict[str, dict[str, Any]]:
        base_score, base_confidence, base_risk = self._base_signal(document)
        section_type = str(document.get("section_type", "")).lower()
        text = _sanitize_text_for_scoring(f"{document.get('title', '')} {document.get('text', '')}").lower()

        if section_type == "risk_factors":
            base_score = min(base_score, -0.2)

        horizon_multipliers = {
            "short_term": 1.0,
            "swing": 1.0,
            "long_term": 1.0,
        }
        if section_type.startswith("item_2_02") or "earnings" in text:
            horizon_multipliers["short_term"] = 1.2
            horizon_multipliers["swing"] = 1.0
            horizon_multipliers["long_term"] = 0.8
        elif section_type == "forward_guidance":
            horizon_multipliers["short_term"] = 0.9
            horizon_multipliers["swing"] = 1.2
            horizon_multipliers["long_term"] = 1.0
        elif section_type == "risk_factors":
            horizon_multipliers["short_term"] = 0.7
            horizon_multipliers["swing"] = 0.9
            horizon_multipliers["long_term"] = 1.1
        elif section_type == "liquidity":
            horizon_multipliers["short_term"] = 0.8
            horizon_multipliers["swing"] = 1.0
            horizon_multipliers["long_term"] = 1.1

        summary = " ".join(str(document["text"]).split()[:25])
        results: dict[str, dict[str, Any]] = {}
        for bucket in HORIZON_BUCKETS:
            results[bucket] = normalize_score_payload(
                {
                    "score": _clamp(base_score * horizon_multipliers[bucket], -1.0, 1.0),
                    "confidence": base_confidence,
                    "risk_flag": base_risk,
                    "summary": summary,
                }
            )
        return results


class OpenAICompatibleScorer:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {config.api_key_env} is not set for openai-compatible scoring."
            )
        self.api_key = api_key

    def score_document(self, document: pd.Series) -> dict[str, dict[str, Any]]:
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "temperature": self.config.temperature,
                "messages": build_messages(document),
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        parsed = _extract_json_object(str(content))
        return normalize_multi_horizon_payload(parsed)


def score_documents(documents: pd.DataFrame, scorer: Any) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    passthrough_columns = [
        "doc_type",
        "source",
        "title",
        "form",
        "section_type",
        "items",
        "source_url",
        "document_id",
        "section_id",
        "company_name",
        "accession_number",
    ]

    for row_index, document in documents.reset_index(drop=True).iterrows():
        result = scorer.score_document(document)
        document_id = str(document.get("document_id") or document.get("accession_number") or row_index)
        section_id = str(document.get("section_id") or f"{document_id}::section")

        for horizon_bucket, normalized in result.items():
            record = {
                "date": document["date"],
                "ticker": document["ticker"],
                "horizon_bucket": horizon_bucket,
                "score": normalized["score"],
                "confidence": normalized["confidence"],
                "risk_flag": normalized["risk_flag"],
                "adjusted_score": normalized["adjusted_score"],
                "effective_score": normalized["effective_score"],
                "summary": normalized["summary"],
                "document_id": document_id,
                "section_id": section_id,
            }
            for column in passthrough_columns:
                if column in document:
                    record[column] = document[column]
            records.append(record)

    return pd.DataFrame(records)


def aggregate_document_scores(scored_documents: pd.DataFrame) -> pd.DataFrame:
    if scored_documents.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "horizon_bucket",
                "llm_score",
                "document_count",
                "section_count",
                "avg_confidence",
                "avg_risk_flag",
            ]
        )

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        section_weights = (
            group.get("section_type", pd.Series("", index=group.index))
            .map(SECTION_TYPE_WEIGHTS)
            .fillna(1.0)
            .astype(float)
        )
        combined_weight = group["confidence"].astype(float) * section_weights
        weight_sum = float(combined_weight.sum())
        if weight_sum > 0:
            llm_score = float((group["adjusted_score"] * combined_weight).sum() / weight_sum)
        else:
            llm_score = float(group["adjusted_score"].mean())

        return pd.Series(
            {
                "llm_score": _clamp(llm_score, -1.0, 1.0),
                "document_count": int(group["document_id"].nunique()),
                "section_count": int(group["section_id"].nunique()),
                "avg_confidence": float(group["confidence"].mean()),
                "avg_risk_flag": float(group["risk_flag"].mean()),
            }
        )

    aggregated = (
        scored_documents.groupby(["date", "ticker", "horizon_bucket"], as_index=False)
        .apply(_aggregate, include_groups=False)
        .reset_index()
    )
    aggregated = aggregated[
        [
            "date",
            "ticker",
            "horizon_bucket",
            "llm_score",
            "document_count",
            "section_count",
            "avg_confidence",
            "avg_risk_flag",
        ]
    ].copy()
    aggregated["document_count"] = aggregated["document_count"].astype(int)
    aggregated["section_count"] = aggregated["section_count"].astype(int)
    return aggregated


def save_score_outputs(
    scored_documents: pd.DataFrame,
    aggregated_scores: pd.DataFrame,
    detailed_output_path: str | Path,
    aggregated_output_path: str | Path,
) -> None:
    detailed_output_path = Path(detailed_output_path)
    aggregated_output_path = Path(aggregated_output_path)
    detailed_output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated_output_path.parent.mkdir(parents=True, exist_ok=True)

    scored_documents.to_csv(detailed_output_path, index=False)
    aggregated_scores.to_csv(aggregated_output_path, index=False)
