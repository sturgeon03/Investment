from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from us_invest_ai.sec_filings import (
    _sec_get_with_retry,
    clean_filing_text,
    extract_periodic_sections,
    extract_recent_filings,
    extract_scoring_documents,
    _extract_mda_block,
)


class SECFilingsTests(unittest.TestCase):
    def test_sec_get_with_retry_recovers_from_retryable_status(self) -> None:
        session = Mock()
        retryable_response = Mock()
        retryable_response.status_code = 503
        retryable_response.headers = {}
        ok_response = Mock()
        ok_response.status_code = 200
        ok_response.headers = {}
        ok_response.raise_for_status.return_value = None
        session.get.side_effect = [retryable_response, ok_response]

        with patch("us_invest_ai.sec_filings.time.sleep") as mocked_sleep:
            response = _sec_get_with_retry(session, "https://example.com", timeout=30)

        self.assertIs(response, ok_response)
        self.assertEqual(session.get.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_extract_recent_filings_filters_forms_dates_and_limit(self) -> None:
        submissions = {
            "cik": "0000123456",
            "name": "Example Corp",
            "filings": {
                "recent": {
                    "filingDate": ["2025-03-10", "2025-02-01", "2024-12-15"],
                    "form": ["8-K", "10-Q", "3"],
                    "items": ["2.02,9.01", "", ""],
                    "accessionNumber": [
                        "0000123456-25-000003",
                        "0000123456-25-000002",
                        "0000123456-24-000001",
                    ],
                    "primaryDocument": ["a8k.htm", "a10q.htm", "form3.xml"],
                    "primaryDocDescription": ["current report", "quarterly report", "ownership"],
                }
            },
        }

        filings = extract_recent_filings(
            submissions=submissions,
            ticker="AAA",
            forms=("10-Q", "8-K"),
            start_date="2025-01-01",
            limit_per_ticker=1,
        )

        self.assertEqual(len(filings), 1)
        self.assertEqual(filings[0].form, "8-K")
        self.assertEqual(filings[0].items, "2.02,9.01")
        self.assertIn("/123456/000012345625000003/a8k.htm", filings[0].url)

    def test_clean_filing_text_removes_hidden_inline_xbrl_noise(self) -> None:
        html = """
        <html>
          <head><title>Ignore me</title><style>.x{display:none;}</style></head>
          <body>
            <ix:header>hidden ixbrl content</ix:header>
            <div style="display:none">hidden style content</div>
            <p>Visible filing text.</p>
            <div aria-hidden="true">not visible</div>
          </body>
        </html>
        """

        cleaned = clean_filing_text(html, is_html=True)

        self.assertIn("Visible filing text.", cleaned)
        self.assertNotIn("hidden ixbrl content", cleaned)
        self.assertNotIn("hidden style content", cleaned)
        self.assertNotIn("not visible", cleaned)

    def test_extract_scoring_documents_keeps_allowed_8k_items_only(self) -> None:
        filings = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-03-01", "2025-03-02"]),
                "ticker": ["AAA", "BBB"],
                "form": ["8-K", "8-K"],
                "items": ["2.02,9.01", "5.02,9.01"],
                "accession_number": ["acc-1", "acc-2"],
                "url": ["https://example.com/1", "https://example.com/2"],
                "company_name": ["Example A", "Example B"],
                "raw_title": ["8-K", "8-K"],
                "raw_text": [
                    "Item 2.02 Results of Operations and Financial Condition Revenue improved materially. Item 9.01 Financial Statements and Exhibits",
                    "Item 5.02 Departure of Directors or Certain Officers Compensation update. Item 9.01 Financial Statements and Exhibits",
                ],
            }
        )

        sections = extract_scoring_documents(filings, min_section_chars=40)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections.loc[0, "ticker"], "AAA")
        self.assertEqual(sections.loc[0, "section_type"], "item_2_02")

    def test_extract_periodic_sections_returns_target_sections(self) -> None:
        filing = pd.Series(
            {
                "form": "10-Q",
                "raw_text": (
                    "Item 1A Risk Factors Demand volatility remains a risk factor for the business and supply conditions can change quickly across several product lines, "
                    "which may pressure margins, inventory planning, promotional cadence, and customer traffic during the remainder of the fiscal year. "
                    "Item 2 Management's Discussion and Analysis of Financial Condition and Results of Operations "
                    "Revenue improved year over year. Liquidity and Capital Resources Cash remained strong and debt stayed low. "
                    "Management expects demand to remain healthy and the outlook for the next quarter improved. "
                    "The company expects margin expansion, expects customer demand to stay firm, and believes the current guidance range remains achievable over the coming quarters. "
                    "Item 3 Quantitative and Qualitative Disclosures About Market Risk"
                ),
            }
        )

        sections = extract_periodic_sections(filing, min_section_chars=40)
        section_types = {section["section_type"] for section in sections}

        self.assertIn("mda", section_types)
        self.assertIn("risk_factors", section_types)
        self.assertIn("liquidity", section_types)
        self.assertIn("forward_guidance", section_types)

    def test_extract_mda_block_prefers_actual_section_over_table_of_contents(self) -> None:
        text = (
            "Table of Contents Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations 13 "
            "Item 3. Quantitative and Qualitative Disclosures About Market Risk 18 "
            "PART I - FINANCIAL INFORMATION "
            "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations "
            "Revenue improved and services growth accelerated materially during the quarter. "
            "Management expects demand to remain healthy through the next quarter and operating margin to expand. "
            "Item 3. Quantitative and Qualitative Disclosures About Market Risk"
        )

        extracted = _extract_mda_block(text, "10-Q")

        self.assertIn("Revenue improved and services growth accelerated materially", extracted)


if __name__ == "__main__":
    unittest.main()
