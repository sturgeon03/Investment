from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from us_invest_ai.llm_scoring import (
    HeuristicScorer,
    aggregate_document_scores,
    normalize_multi_horizon_payload,
    normalize_score_payload,
    score_documents,
)
from us_invest_ai.signals import attach_llm_scores, load_llm_scores


class LLMScoringTests(unittest.TestCase):
    def test_normalize_score_payload_clamps_ranges(self) -> None:
        normalized = normalize_score_payload(
            {
                "score": 5,
                "confidence": 2,
                "risk_flag": -1,
                "summary": "summary",
            }
        )

        self.assertEqual(normalized["score"], 1.0)
        self.assertEqual(normalized["confidence"], 1.0)
        self.assertEqual(normalized["risk_flag"], 0.0)

    def test_normalize_multi_horizon_payload_requires_all_horizons(self) -> None:
        normalized = normalize_multi_horizon_payload(
            {
                "short_term": {"score": 0.2, "confidence": 0.6, "risk_flag": 0.1, "summary": "a"},
                "swing": {"score": 0.4, "confidence": 0.7, "risk_flag": 0.2, "summary": "b"},
                "long_term": {"score": -0.1, "confidence": 0.5, "risk_flag": 0.3, "summary": "c"},
            }
        )

        self.assertEqual(sorted(normalized.keys()), ["long_term", "short_term", "swing"])
        self.assertAlmostEqual(normalized["swing"]["adjusted_score"], 0.36, places=2)

    def test_heuristic_scoring_produces_all_horizon_rows(self) -> None:
        documents = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-03-01"]),
                "ticker": ["AAA"],
                "title": ["Guidance improves"],
                "text": ["Strong demand improved backlog and durable growth supported margins."],
                "doc_type": ["sec_section"],
                "source": ["test"],
                "form": ["10-Q"],
                "section_type": ["forward_guidance"],
                "items": ["2"],
                "source_url": ["https://example.com"],
                "document_id": ["doc-1"],
                "section_id": ["doc-1::forward_guidance"],
            }
        )

        scored = score_documents(documents, HeuristicScorer())

        self.assertEqual(set(scored["horizon_bucket"]), {"short_term", "swing", "long_term"})
        self.assertTrue((scored["score"] > 0.0).all())
        self.assertTrue((scored["document_id"] == "doc-1").all())

    def test_aggregate_document_scores_keeps_horizon_dimension(self) -> None:
        scored_documents = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-03-01", "2025-03-01", "2025-03-01"]),
                "ticker": ["AAA", "AAA", "AAA"],
                "horizon_bucket": ["swing", "swing", "long_term"],
                "score": [0.5, -0.2, 0.1],
                "confidence": [0.9, 0.1, 0.8],
                "risk_flag": [0.0, 0.0, 0.2],
                "adjusted_score": [0.5, -0.2, 0.03],
                "effective_score": [0.45, -0.02, 0.024],
                "summary": ["one", "two", "three"],
                "document_id": ["doc-a", "doc-a", "doc-b"],
                "section_id": ["doc-a::1", "doc-a::2", "doc-b::1"],
                "section_type": ["forward_guidance", "risk_factors", "mda"],
            }
        )

        aggregated = aggregate_document_scores(scored_documents)
        swing_row = aggregated.loc[aggregated["horizon_bucket"] == "swing"].iloc[0]

        self.assertAlmostEqual(swing_row["llm_score"], 0.475, places=2)
        self.assertEqual(int(swing_row["document_count"]), 1)
        self.assertEqual(int(swing_row["section_count"]), 2)

    def test_load_llm_scores_filters_requested_horizon(self) -> None:
        sample = """date,ticker,horizon_bucket,llm_score,document_count,section_count,avg_confidence,avg_risk_flag
2025-01-31,AAA,short_term,0.1,1,1,0.8,0.1
2025-01-31,AAA,swing,0.3,1,1,0.8,0.1
"""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as handle:
            handle.write(sample)
            temp_path = Path(handle.name)
        try:
            scores = load_llm_scores(temp_path, horizon_bucket="swing")
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores.loc[0, "llm_score"], 0.3)

    def test_attach_llm_scores_uses_latest_available_signal(self) -> None:
        features = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-28"]),
                "ticker": ["AAA", "AAA", "AAA"],
                "close": [100, 101, 102],
            }
        )
        llm_scores = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-15", "2025-02-10"]),
                "ticker": ["AAA", "AAA"],
                "llm_score": [0.2, -0.4],
                "document_count": [1, 2],
            }
        )

        merged = attach_llm_scores(features, llm_scores)

        self.assertEqual(merged.loc[0, "llm_score"], 0.2)
        self.assertEqual(merged.loc[1, "llm_score"], 0.2)
        self.assertEqual(merged.loc[2, "llm_score"], -0.4)
        self.assertEqual(merged.loc[2, "document_count"], 2)


if __name__ == "__main__":
    unittest.main()
