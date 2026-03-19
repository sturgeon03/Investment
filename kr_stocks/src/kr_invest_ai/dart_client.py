from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from kr_invest_ai.dart_adapter import NormalizedDARTFiling, normalize_dart_filings
from kr_invest_ai.market_config import KRMarketAdapterConfig, load_kr_market_adapter_config


DART_LIST_ENDPOINT = "https://opendart.fss.or.kr/api/list.json"


class DARTAPIError(RuntimeError):
    """Raised when the DART API returns a non-success status."""


@dataclass(frozen=True, slots=True)
class DARTListFilingsRequest:
    begin_date: date
    end_date: date
    corp_code: str | None = None
    page_no: int = 1
    page_count: int = 100
    last_report_only: bool = True
    filing_type: str | None = None
    filing_detail_type: str | None = None
    corporation_class: str | None = None

    def to_params(self, api_key: str) -> dict[str, str]:
        params = {
            "crtfc_key": api_key,
            "bgn_de": self.begin_date.strftime("%Y%m%d"),
            "end_de": self.end_date.strftime("%Y%m%d"),
            "page_no": str(self.page_no),
            "page_count": str(self.page_count),
            "last_reprt_at": "Y" if self.last_report_only else "N",
        }
        if self.corp_code:
            params["corp_code"] = self.corp_code
        if self.filing_type:
            params["pblntf_ty"] = self.filing_type
        if self.filing_detail_type:
            params["pblntf_detail_ty"] = self.filing_detail_type
        if self.corporation_class:
            params["corp_cls"] = self.corporation_class
        return params


def build_dart_filing_url(receipt_no: str) -> str:
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}"


def map_dart_list_row_to_payload(row: dict[str, Any]) -> dict[str, Any]:
    receipt_no = str(row["rcept_no"])
    payload = {
        "rcept_no": receipt_no,
        "corp_code": row.get("corp_code"),
        "corp_name": row.get("corp_name"),
        "stock_code": row.get("stock_code"),
        "report_name": row.get("report_nm"),
        "report_subtype": row.get("rm"),
        "receipt_datetime": row.get("rcept_dt"),
        "source_url": build_dart_filing_url(receipt_no),
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


class DARTOpenAPIClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DART_LIST_ENDPOINT,
        adapter_config: KRMarketAdapterConfig | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DART_API_KEY")
        if not self.api_key:
            raise ValueError("DART API key is required. Provide api_key or set DART_API_KEY.")
        self.base_url = base_url
        self.adapter_config = adapter_config or load_kr_market_adapter_config()

    def list_filings(self, request: DARTListFilingsRequest) -> list[dict[str, Any]]:
        response = self._request_json(request.to_params(self.api_key))
        status = str(response.get("status", ""))
        if status != "000":
            message = str(response.get("message", "Unknown DART API error."))
            raise DARTAPIError(f"DART API error {status}: {message}")
        rows = response.get("list", [])
        if not isinstance(rows, list):
            raise DARTAPIError("DART API response 'list' field must be a list.")
        return [row for row in rows if isinstance(row, dict)]

    def fetch_normalized_filings(
        self,
        request: DARTListFilingsRequest,
        *,
        holiday_dates: set[date] | None = None,
    ) -> list[NormalizedDARTFiling]:
        rows = self.list_filings(request)
        payloads = [map_dart_list_row_to_payload(row) for row in rows]
        return normalize_dart_filings(
            payloads,
            adapter_config=self.adapter_config,
            holiday_dates=holiday_dates,
        )

    def _request_json(self, params: dict[str, str]) -> dict[str, Any]:
        query = urlencode(params)
        with urlopen(f"{self.base_url}?{query}") as response:
            return json.loads(response.read().decode("utf-8"))
