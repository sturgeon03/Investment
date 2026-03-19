from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from kr_invest_ai.calendar import align_filing_timestamp_to_session_date
from kr_invest_ai.market_config import KRMarketAdapterConfig, load_kr_market_adapter_config
from kr_invest_ai.tickers import CanonicalKRTicker, canonicalize_kr_ticker


@dataclass(frozen=True, slots=True)
class NormalizedDARTFiling:
    receipt_no: str | None
    issuer_code: str | None
    issuer_name: str
    canonical_ticker: CanonicalKRTicker | None
    filing_type: str
    filing_subtype: str | None
    title: str
    filed_at: datetime
    session_date: date
    category: str
    body_text: str | None
    source_url: str | None
    raw_fields: dict[str, str]


ANNUAL_PATTERNS = ("사업보고서", "ANNUAL REPORT")
SEMIANNUAL_PATTERNS = ("반기보고서", "SEMIANNUAL REPORT")
QUARTERLY_PATTERNS = ("분기보고서", "QUARTERLY REPORT")
EARNINGS_PATTERNS = ("영업(잠정)실적", "EARNINGS", "실적")
MAJOR_CONTRACT_PATTERNS = ("단일판매", "공급계약", "MAJOR CONTRACT")
CAPITAL_PATTERNS = ("유상증자", "무상증자", "전환사채", "BW", "CB", "RIGHTS OFFERING")
GOVERNANCE_PATTERNS = ("주주총회", "대표이사", "임원", "이사", "감사", "GOVERNANCE", "BOARD")


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_first(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _parse_dart_timestamp(value: Any, timezone_name: str) -> datetime:
    text = _clean_text(value)
    if text is None:
        raise ValueError("DART filing timestamp is required.")

    timezone = ZoneInfo(timezone_name)
    normalized = text.replace("Z", "+00:00")
    for parser in (
        lambda s: datetime.fromisoformat(s),
        lambda s: datetime.strptime(s, "%Y%m%d%H%M%S"),
        lambda s: datetime.strptime(s, "%Y%m%d%H%M"),
        lambda s: datetime.strptime(s, "%Y%m%d"),
    ):
        try:
            parsed = parser(normalized)
            return parsed.replace(tzinfo=timezone) if parsed.tzinfo is None else parsed.astimezone(timezone)
        except ValueError:
            continue
    raise ValueError(f"Unsupported DART filing timestamp format: {text}")


def classify_dart_filing_category(
    filing_type: str | None,
    filing_subtype: str | None,
    title: str | None,
) -> str:
    haystack = " ".join(
        value.upper()
        for value in (filing_type, filing_subtype, title)
        if _clean_text(value) is not None
    )

    for patterns, category in (
        (ANNUAL_PATTERNS, "annual_report"),
        (SEMIANNUAL_PATTERNS, "semiannual_report"),
        (QUARTERLY_PATTERNS, "quarterly_report"),
        (EARNINGS_PATTERNS, "earnings"),
        (MAJOR_CONTRACT_PATTERNS, "major_contract"),
        (CAPITAL_PATTERNS, "capital_event"),
        (GOVERNANCE_PATTERNS, "governance"),
    ):
        if any(pattern.upper() in haystack for pattern in patterns):
            return category
    return "other"


def normalize_dart_filing(
    payload: dict[str, Any],
    adapter_config: KRMarketAdapterConfig | None = None,
    holiday_dates: set[date] | None = None,
) -> NormalizedDARTFiling:
    config = adapter_config or load_kr_market_adapter_config()
    issuer_name = _clean_text(_pick_first(payload, ("corp_name", "issuer_name", "company_name")))
    if issuer_name is None:
        raise ValueError("DART filing requires an issuer name.")

    filing_type = _clean_text(_pick_first(payload, ("report_name", "filing_type", "form")))
    filing_subtype = _clean_text(_pick_first(payload, ("report_subtype", "filing_subtype", "subcategory")))
    title = _clean_text(_pick_first(payload, ("report_title", "title", "headline"))) or issuer_name
    filed_at = _parse_dart_timestamp(
        _pick_first(payload, ("receipt_datetime", "filed_at", "rcept_dt", "date")),
        config.market_assumptions.market.timezone,
    )
    session_date = align_filing_timestamp_to_session_date(
        filed_at,
        config.market_assumptions,
        trading_calendar=config.trading_calendar,
        holiday_dates=holiday_dates,
    )

    raw_ticker = _pick_first(payload, ("stock_code", "listing_code", "ticker"))
    canonical_ticker = (
        canonicalize_kr_ticker(str(raw_ticker), conventions=config.ticker_conventions)
        if _clean_text(raw_ticker) is not None
        else None
    )

    normalized_fields = {
        str(key): str(value)
        for key, value in payload.items()
        if value is not None
    }
    return NormalizedDARTFiling(
        receipt_no=_clean_text(_pick_first(payload, ("receipt_no", "rcept_no"))),
        issuer_code=_clean_text(_pick_first(payload, ("corp_code", "issuer_code"))),
        issuer_name=issuer_name,
        canonical_ticker=canonical_ticker,
        filing_type=filing_type or "UNKNOWN",
        filing_subtype=filing_subtype,
        title=title,
        filed_at=filed_at,
        session_date=session_date,
        category=classify_dart_filing_category(filing_type, filing_subtype, title),
        body_text=_clean_text(_pick_first(payload, ("body_text", "text", "document_text"))),
        source_url=_clean_text(_pick_first(payload, ("source_url", "url", "link"))),
        raw_fields=normalized_fields,
    )


def normalize_dart_filings(
    payloads: list[dict[str, Any]],
    adapter_config: KRMarketAdapterConfig | None = None,
    holiday_dates: set[date] | None = None,
) -> list[NormalizedDARTFiling]:
    config = adapter_config or load_kr_market_adapter_config()
    return [
        normalize_dart_filing(payload, adapter_config=config, holiday_dates=holiday_dates)
        for payload in payloads
    ]


def filings_to_frame(filings: list[NormalizedDARTFiling]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filing in filings:
        rows.append(
            {
                "receipt_no": filing.receipt_no,
                "issuer_code": filing.issuer_code,
                "issuer_name": filing.issuer_name,
                "ticker": filing.canonical_ticker.vendor_ticker if filing.canonical_ticker is not None else None,
                "listing_code": filing.canonical_ticker.listing_code if filing.canonical_ticker is not None else None,
                "vendor_suffix": filing.canonical_ticker.vendor_suffix if filing.canonical_ticker is not None else None,
                "filing_type": filing.filing_type,
                "filing_subtype": filing.filing_subtype,
                "title": filing.title,
                "filed_at": filing.filed_at,
                "session_date": filing.session_date,
                "category": filing.category,
                "body_text": filing.body_text,
                "source_url": filing.source_url,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "receipt_no",
                "issuer_code",
                "issuer_name",
                "ticker",
                "listing_code",
                "vendor_suffix",
                "filing_type",
                "filing_subtype",
                "title",
                "filed_at",
                "session_date",
                "category",
                "body_text",
                "source_url",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["filed_at"] = pd.to_datetime(frame["filed_at"])
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    return frame.sort_values(["session_date", "issuer_name", "title"]).reset_index(drop=True)
