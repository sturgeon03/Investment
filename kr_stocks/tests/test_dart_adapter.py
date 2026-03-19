from __future__ import annotations

import unittest
from datetime import date

from kr_invest_ai.dart_adapter import (
    classify_dart_filing_category,
    normalize_dart_filing,
    normalize_dart_filings,
)
from kr_invest_ai.market_config import load_kr_market_adapter_config


class DARTAdapterTests(unittest.TestCase):
    def test_classify_dart_filing_category_covers_small_reserved_set(self) -> None:
        self.assertEqual(
            classify_dart_filing_category("사업보고서", None, "제25기 사업보고서"),
            "annual_report",
        )
        self.assertEqual(
            classify_dart_filing_category(None, None, "단일판매ㆍ공급계약체결"),
            "major_contract",
        )
        self.assertEqual(
            classify_dart_filing_category(None, None, "유상증자결정"),
            "capital_event",
        )
        self.assertEqual(
            classify_dart_filing_category(None, None, "대표이사변경"),
            "governance",
        )

    def test_normalize_dart_filing_aligns_after_close_to_next_trading_day(self) -> None:
        adapter = load_kr_market_adapter_config()
        filing = normalize_dart_filing(
            {
                "corp_name": "Samsung Electronics",
                "corp_code": "00126380",
                "stock_code": "005930.KS",
                "report_name": "영업(잠정)실적",
                "title": "영업(잠정)실적(공정공시)",
                "receipt_datetime": "20260320160500",
                "rcept_no": "20260320000123",
                "document_text": "Earnings beat expectations.",
                "url": "https://dart.fss.or.kr/example",
            },
            adapter_config=adapter,
        )

        self.assertEqual(filing.issuer_name, "Samsung Electronics")
        self.assertEqual(filing.canonical_ticker.listing_code, "005930")
        self.assertEqual(filing.canonical_ticker.vendor_suffix, "KS")
        self.assertEqual(filing.category, "earnings")
        self.assertEqual(filing.session_date, date(2026, 3, 23))

    def test_normalize_dart_filing_supports_alternate_field_names(self) -> None:
        adapter = load_kr_market_adapter_config()
        filing = normalize_dart_filing(
            {
                "issuer_name": "NAVER",
                "issuer_code": "00401234",
                "listing_code": "035420",
                "filing_type": "분기보고서",
                "filing_subtype": "정정",
                "headline": "분기보고서(정정)",
                "filed_at": "2026-03-19T14:00:00+09:00",
                "receipt_no": "20260319000456",
                "body_text": "Quarterly report body.",
                "source_url": "https://dart.fss.or.kr/alt",
            },
            adapter_config=adapter,
        )

        self.assertEqual(filing.issuer_code, "00401234")
        self.assertEqual(filing.canonical_ticker.listing_code, "035420")
        self.assertEqual(filing.category, "quarterly_report")
        self.assertEqual(filing.session_date, date(2026, 3, 19))

    def test_normalize_dart_filings_keeps_shared_adapter_config(self) -> None:
        adapter = load_kr_market_adapter_config()
        filings = normalize_dart_filings(
            [
                {
                    "corp_name": "Company A",
                    "report_name": "사업보고서",
                    "receipt_datetime": "20260319090000",
                    "stock_code": "000001.KS",
                },
                {
                    "corp_name": "Company B",
                    "title": "주주총회소집결의",
                    "receipt_datetime": "20260319120000",
                    "stock_code": "000002.KQ",
                },
            ],
            adapter_config=adapter,
        )

        self.assertEqual(len(filings), 2)
        self.assertEqual(filings[0].category, "annual_report")
        self.assertEqual(filings[1].category, "governance")


if __name__ == "__main__":
    unittest.main()
