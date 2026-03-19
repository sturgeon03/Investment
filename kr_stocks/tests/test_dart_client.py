from __future__ import annotations

import unittest
from datetime import date

from kr_invest_ai.dart_client import (
    DARTAPIError,
    DARTListFilingsRequest,
    DARTOpenAPIClient,
    build_dart_filing_url,
    map_dart_list_row_to_payload,
)
from kr_invest_ai.market_config import load_kr_market_adapter_config


class _StubDARTClient(DARTOpenAPIClient):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(api_key="test-key", adapter_config=load_kr_market_adapter_config())
        self.payload = payload
        self.last_params: dict[str, str] | None = None

    def _request_json(self, params: dict[str, str]) -> dict[str, object]:
        self.last_params = params
        return self.payload


class DARTClientTests(unittest.TestCase):
    def test_request_builds_expected_params(self) -> None:
        request = DARTListFilingsRequest(
            begin_date=date(2026, 3, 1),
            end_date=date(2026, 3, 19),
            corp_code="00126380",
            filing_type="A",
            filing_detail_type="A001",
            corporation_class="Y",
        )

        params = request.to_params("abc123")

        self.assertEqual(params["crtfc_key"], "abc123")
        self.assertEqual(params["bgn_de"], "20260301")
        self.assertEqual(params["end_de"], "20260319")
        self.assertEqual(params["corp_code"], "00126380")
        self.assertEqual(params["pblntf_ty"], "A")
        self.assertEqual(params["pblntf_detail_ty"], "A001")
        self.assertEqual(params["corp_cls"], "Y")
        self.assertEqual(params["last_reprt_at"], "Y")

    def test_map_dart_list_row_to_payload_adds_source_url(self) -> None:
        payload = map_dart_list_row_to_payload(
            {
                "rcept_no": "20260319000123",
                "corp_name": "Samsung Electronics",
                "stock_code": "005930",
                "report_nm": "QUARTERLY REPORT",
                "rcept_dt": "20260319153000",
            }
        )

        self.assertEqual(payload["corp_name"], "Samsung Electronics")
        self.assertEqual(payload["report_name"], "QUARTERLY REPORT")
        self.assertEqual(
            payload["source_url"],
            build_dart_filing_url("20260319000123"),
        )

    def test_list_filings_raises_on_non_success_status(self) -> None:
        client = _StubDARTClient({"status": "013", "message": "No data."})

        with self.assertRaises(DARTAPIError):
            client.list_filings(
                DARTListFilingsRequest(
                    begin_date=date(2026, 3, 1),
                    end_date=date(2026, 3, 19),
                )
            )

    def test_fetch_normalized_filings_routes_rows_through_adapter(self) -> None:
        client = _StubDARTClient(
            {
                "status": "000",
                "list": [
                    {
                        "rcept_no": "20260320000123",
                        "corp_code": "00126380",
                        "corp_name": "Samsung Electronics",
                        "stock_code": "005930",
                        "report_nm": "EARNINGS",
                        "rm": "Preliminary",
                        "rcept_dt": "20260320160500",
                    }
                ],
            }
        )

        filings = client.fetch_normalized_filings(
            DARTListFilingsRequest(
                begin_date=date(2026, 3, 1),
                end_date=date(2026, 3, 20),
            )
        )

        self.assertEqual(client.last_params["page_count"], "100")
        self.assertEqual(len(filings), 1)
        filing = filings[0]
        self.assertEqual(filing.issuer_name, "Samsung Electronics")
        self.assertEqual(filing.category, "earnings")
        self.assertEqual(filing.session_date, date(2026, 3, 23))
        self.assertEqual(
            filing.source_url,
            "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260320000123",
        )


if __name__ == "__main__":
    unittest.main()
