from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}"
SEC_USER_AGENT_ENV = "SEC_USER_AGENT"
DEFAULT_FORMS = ("10-K", "10-Q", "8-K")
ALLOWED_8K_ITEMS = ("2.02", "7.01", "8.01")
EXCLUDED_8K_ITEMS = ("1.01", "5.02")
GUIDANCE_KEYWORDS = (
    "guidance",
    "outlook",
    "expect",
    "expects",
    "expected",
    "forecast",
    "forecasts",
    "anticipate",
    "anticipates",
    "anticipated",
    "believe",
    "believes",
    "forward-looking",
    "will",
    "trend",
)
BOILERPLATE_SENTENCE_MARKERS = (
    "forward-looking statements",
    "private securities litigation reform act",
    "assumes no obligation to revise or update",
    "see accompanying notes",
    "this item and other sections of this quarterly report",
    "this item and other sections of this annual report",
)
NOISE_8K_PATTERNS = (
    "underwriting agreement",
    "officers' certificate",
    "officers certificate",
    "notes due",
    "floating rate notes",
    "aggregate principal amount",
    "registration statement on form s-3",
)


@dataclass(slots=True)
class FilingMetadata:
    ticker: str
    company_name: str
    cik: int
    form: str
    filing_date: pd.Timestamp
    accession_number: str
    primary_document: str
    title: str
    items: str
    url: str


def resolve_user_agent(user_agent: str | None) -> str:
    candidate = user_agent or os.environ.get(SEC_USER_AGENT_ENV)
    if not candidate or not candidate.strip():
        raise ValueError(
            "Set SEC_USER_AGENT or pass --user-agent. "
            "SEC requests should identify the application and contact email."
        )
    return candidate.strip()


def build_sec_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def fetch_company_tickers(session: requests.Session) -> pd.DataFrame:
    response = session.get(SEC_TICKERS_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    companies = pd.DataFrame(payload.values())
    companies["ticker"] = companies["ticker"].str.upper()
    companies["cik_str"] = companies["cik_str"].astype(int)
    return companies.rename(columns={"title": "company_name"})


def lookup_companies(companies: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    requested = [ticker.upper() for ticker in tickers]
    matched = companies.loc[companies["ticker"].isin(requested)].copy()
    missing = sorted(set(requested).difference(matched["ticker"].tolist()))
    if missing:
        raise ValueError(f"Tickers not found in SEC company_tickers.json: {missing}")
    return matched.sort_values("ticker").reset_index(drop=True)


def fetch_submissions(session: requests.Session, cik: int) -> dict:
    response = session.get(SEC_SUBMISSIONS_URL.format(cik=cik), timeout=30)
    response.raise_for_status()
    return response.json()


def extract_recent_filings(
    submissions: dict,
    ticker: str,
    forms: tuple[str, ...] = DEFAULT_FORMS,
    start_date: str | None = None,
    limit_per_ticker: int | None = None,
) -> list[FilingMetadata]:
    recent = pd.DataFrame(submissions["filings"]["recent"])
    if recent.empty:
        return []

    recent["filingDate"] = pd.to_datetime(recent["filingDate"]).dt.normalize()
    recent["form"] = recent["form"].astype(str)
    recent = recent.loc[recent["form"].isin(set(forms))].copy()
    if start_date:
        recent = recent.loc[recent["filingDate"] >= pd.Timestamp(start_date)]

    recent = recent.sort_values("filingDate", ascending=False)
    if limit_per_ticker is not None:
        recent = recent.head(limit_per_ticker)

    cik = int(submissions["cik"])
    company_name = submissions["name"]
    records: list[FilingMetadata] = []
    for _, row in recent.iterrows():
        accession_number = str(row["accessionNumber"])
        accession_no_dashes = accession_number.replace("-", "")
        primary_document = str(row.get("primaryDocument") or "").strip() or f"{accession_no_dashes}.txt"
        title = str(row.get("primaryDocDescription") or "").strip() or f"{row['form']} filing"
        items = str(row.get("items") or "").strip()
        url = SEC_ARCHIVES_URL.format(
            cik=cik,
            accession_no_dashes=accession_no_dashes,
            primary_document=primary_document,
        )
        records.append(
            FilingMetadata(
                ticker=ticker.upper(),
                company_name=company_name,
                cik=cik,
                form=str(row["form"]),
                filing_date=row["filingDate"],
                accession_number=accession_number,
                primary_document=primary_document,
                title=title,
                items=items,
                url=url,
            )
        )

    return records


def _strip_hidden_and_inline_xbrl(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all(True)):
        name = (getattr(tag, "name", "") or "").lower()
        attrs = dict(getattr(tag, "attrs", {}) or {})
        style = str(attrs.get("style", "")).lower()
        aria_hidden = str(attrs.get("aria-hidden", "")).lower()
        classes = (
            " ".join(attrs.get("class", []))
            if isinstance(attrs.get("class"), list)
            else str(attrs.get("class", ""))
        )
        if (
            ":" in name
            or attrs.get("hidden") is not None
            or "display:none" in style
            or aria_hidden == "true"
            or "hidden" in classes.lower()
        ):
            tag.decompose()

    for name in ["script", "style", "head", "meta", "link", "title", "noscript"]:
        for tag in soup.find_all(name):
            tag.decompose()


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u2019", "'").replace("\u2014", "-")


def _remove_common_noise(text: str) -> str:
    cleaned = _normalize_unicode(text)
    cleaned = re.sub(r"Table of Contents", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bPage\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_filing_text(raw_text: str, is_html: bool) -> str:
    if is_html:
        soup = BeautifulSoup(raw_text, "html.parser")
        _strip_hidden_and_inline_xbrl(soup)
        text = soup.get_text(" ", strip=True)
    else:
        text = raw_text

    return _remove_common_noise(text)


def download_filing_text(
    session: requests.Session,
    filing: FilingMetadata,
    max_chars: int,
) -> str:
    response = session.get(filing.url, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or filing.primary_document.lower().endswith(".pdf"):
        raise ValueError("pdf filing documents are not supported")
    is_html = (
        "html" in content_type
        or "xml" in content_type
        or filing.primary_document.lower().endswith((".htm", ".html", ".xml"))
    )
    cleaned = clean_filing_text(response.text, is_html=is_html)
    return cleaned[:max_chars]


def fetch_filing_documents(
    session: requests.Session,
    filings: list[FilingMetadata],
    max_chars: int,
    pause_seconds: float,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, str | int]] = []
    errors: list[str] = []

    for index, filing in enumerate(filings):
        try:
            raw_text = download_filing_text(session, filing, max_chars=max_chars)
            if not raw_text:
                raise ValueError("empty filing text")
            rows.append(
                {
                    "date": filing.filing_date.date().isoformat(),
                    "ticker": filing.ticker,
                    "doc_type": "sec_filing_raw",
                    "source": "SEC",
                    "title": f"{filing.form} {filing.title}",
                    "text": raw_text,
                    "form": filing.form,
                    "items": filing.items,
                    "company_name": filing.company_name,
                    "cik": f"{filing.cik:010d}",
                    "accession_number": filing.accession_number,
                    "url": filing.url,
                    "raw_title": filing.title,
                    "raw_text": raw_text,
                }
            )
        except Exception as exc:
            errors.append(f"{filing.ticker} {filing.form} {filing.accession_number}: {exc}")

        if index < len(filings) - 1 and pause_seconds > 0:
            time.sleep(pause_seconds)

    return pd.DataFrame(rows), errors


def save_documents(documents: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents.to_csv(output_path, index=False)


def load_raw_filings(path: str | Path) -> pd.DataFrame:
    filings = pd.read_csv(path)
    required = {
        "date",
        "ticker",
        "form",
        "items",
        "accession_number",
        "url",
        "company_name",
        "raw_title",
        "raw_text",
    }
    missing = required.difference(filings.columns)
    if missing:
        raise ValueError(f"Raw SEC filings file is missing columns: {sorted(missing)}")

    filings = filings.copy()
    filings["date"] = pd.to_datetime(filings["date"]).dt.normalize()
    filings["ticker"] = filings["ticker"].str.upper()
    filings["form"] = filings["form"].astype(str).str.upper()
    filings["items"] = filings["items"].fillna("").astype(str)
    filings["raw_title"] = filings["raw_title"].fillna("").astype(str)
    filings["raw_text"] = filings["raw_text"].fillna("").astype(str)
    filings["url"] = filings["url"].fillna("").astype(str)
    filings["source"] = filings.get("source", "SEC")
    return filings


def normalize_items(items: str) -> list[str]:
    normalized = re.findall(r"\d+\.\d+", str(items))
    deduped: list[str] = []
    for item in normalized:
        if item not in deduped:
            deduped.append(item)
    return deduped


def extract_item_section(text: str, item_number: str) -> str:
    item_pattern = re.compile(rf"\bItem\s+{re.escape(item_number)}\.?\b", re.IGNORECASE)
    matches = list(item_pattern.finditer(text))
    candidates: list[str] = []
    for match in matches:
        trailing_text = text[match.start() :]
        next_match = re.search(
            r"\bItem\s+\d+\.\d+\.?\b",
            trailing_text[len(match.group(0)) :],
            re.IGNORECASE,
        )
        end = len(trailing_text)
        if next_match:
            end = len(match.group(0)) + next_match.start()
        section = trailing_text[:end]
        section = re.split(r"\bSIGNATURES?\b", section, maxsplit=1, flags=re.IGNORECASE)[0]
        section = re.split(r"\bItem\s+9\.01\.?\b", section, maxsplit=1, flags=re.IGNORECASE)[0]
        section = re.sub(r"\s+", " ", section).strip()
        if section:
            candidates.append(section)
    return max(candidates, key=len, default="")


def extract_8k_sections(
    filing: pd.Series,
    min_section_chars: int,
) -> list[dict[str, str]]:
    items = normalize_items(filing["items"])
    if not any(item in ALLOWED_8K_ITEMS for item in items):
        return []
    if any(item in EXCLUDED_8K_ITEMS for item in items) and not any(
        item in ALLOWED_8K_ITEMS for item in items
    ):
        return []

    extracted: list[dict[str, str]] = []
    for item in items:
        if item not in ALLOWED_8K_ITEMS:
            continue
        section_text = extract_item_section(filing["raw_text"], item)
        if len(section_text) < min_section_chars:
            continue
        if item == "8.01" and any(pattern in section_text.lower() for pattern in NOISE_8K_PATTERNS):
            continue
        extracted.append(
            {
                "section_type": f"item_{item.replace('.', '_')}",
                "items": item,
                "title": f"{filing['form']} Item {item}",
                "text": section_text,
            }
        )
    return extracted


def _extract_item_block(text: str, item_number: str) -> str:
    item_pattern = re.compile(
        rf"\bItem\s+{re.escape(item_number)}[\.:\-\s]",
        re.IGNORECASE,
    )
    matches = list(item_pattern.finditer(text))
    candidates: list[str] = []
    for match in matches:
        trailing_text = text[match.start() :]
        next_match = re.search(
            r"\bItem\s+(?:1A|1B|2|3|4|5|6|7|7A|8|9|9A|9B|10|11|12|13|14|15)\b",
            trailing_text[len(match.group(0)) :],
            re.IGNORECASE,
        )
        end = len(trailing_text)
        if next_match:
            end = len(match.group(0)) + next_match.start()
        candidate = trailing_text[:end].strip()
        candidate = re.split(r"\bSIGNATURES?\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate:
            candidates.append(candidate)
    return max(candidates, key=len, default="")


def _extract_mda_block(text: str, form: str) -> str:
    heading_pattern = re.compile(
        r"Management'?s Discussion and Analysis of Financial Condition and Results of Operations",
        re.IGNORECASE,
    )
    matches = list(heading_pattern.finditer(text))
    candidates: list[str] = []
    end_pattern = (
        re.compile(r"\bItem\s+(?:3|4)\b", re.IGNORECASE)
        if form == "10-Q"
        else re.compile(r"\bItem\s+(?:7A|8)\b", re.IGNORECASE)
    )
    for match in matches:
        trailing_text = text[match.start() :]
        next_match = end_pattern.search(trailing_text[match.end() - match.start() :])
        end = len(trailing_text)
        if next_match:
            end = (match.end() - match.start()) + next_match.start()
        candidate = trailing_text[:end].strip()
        candidate = re.split(r"\bSIGNATURES?\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate:
            candidates.append(candidate)

    if candidates:
        return max(candidates, key=len)
    fallback_item = "2" if form == "10-Q" else "7"
    return _extract_item_block(text, fallback_item)


def extract_named_subsection(text: str, heading: str, max_chars: int = 4000) -> str:
    pattern = re.compile(re.escape(heading), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    trailing_text = text[match.start() :]
    end_candidates = [
        re.search(r"\b[A-Z][A-Za-z/&,\-\s]{8,80}\b", trailing_text[match.end() - match.start() + 50 :]),
        re.search(r"\bQuantitative and Qualitative Disclosures About Market Risk\b", trailing_text, re.IGNORECASE),
        re.search(r"\bControls and Procedures\b", trailing_text, re.IGNORECASE),
    ]
    end = len(trailing_text)
    for candidate in end_candidates:
        if candidate:
            candidate_end = candidate.start()
            if candidate_end > 200:
                end = min(end, candidate_end)
    section = trailing_text[: min(end, max_chars)]
    return re.sub(r"\s+", " ", section).strip()


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def extract_guidance_sentences(text: str, min_sentence_chars: int = 40) -> str:
    sentences = split_sentences(text)
    selected = [
        sentence
        for sentence in sentences
        if len(sentence) >= min_sentence_chars
        and not any(marker in sentence.lower() for marker in BOILERPLATE_SENTENCE_MARKERS)
        and any(keyword in sentence.lower() for keyword in GUIDANCE_KEYWORDS)
    ]
    return " ".join(selected[:8]).strip()


def extract_periodic_sections(
    filing: pd.Series,
    min_section_chars: int,
) -> list[dict[str, str]]:
    form = filing["form"]
    text = filing["raw_text"]
    mda_item = "2" if form == "10-Q" else "7"
    mda_text = _extract_mda_block(text, form)
    risk_text = _extract_item_block(text, "1A")
    sections: list[dict[str, str]] = []

    if len(mda_text) >= min_section_chars:
        sections.append(
            {
                "section_type": "mda",
                "items": mda_item,
                "title": f"{form} MD&A",
                "text": mda_text,
            }
        )

        liquidity_text = extract_named_subsection(mda_text, "Liquidity and Capital Resources")
        if len(liquidity_text) >= max(120, min_section_chars // 2):
            sections.append(
                {
                    "section_type": "liquidity",
                    "items": mda_item,
                    "title": f"{form} Liquidity and Capital Resources",
                    "text": liquidity_text,
                }
            )

        guidance_text = extract_guidance_sentences(mda_text)
        if len(guidance_text) >= max(120, min_section_chars // 2):
            sections.append(
                {
                    "section_type": "forward_guidance",
                    "items": mda_item,
                    "title": f"{form} Forward-Looking Guidance",
                    "text": guidance_text,
                }
            )

    if len(risk_text) >= max(150, min_section_chars // 2):
        sections.append(
            {
                "section_type": "risk_factors",
                "items": "1A",
                "title": f"{form} Risk Factors",
                "text": risk_text,
            }
        )

    return sections


def extract_scoring_documents(
    filings: pd.DataFrame,
    min_section_chars: int = 250,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, filing in filings.iterrows():
        if not filing["raw_text"].strip():
            continue

        if filing["form"] == "8-K":
            extracted_sections = extract_8k_sections(filing, min_section_chars=min_section_chars)
        elif filing["form"] in {"10-Q", "10-K"}:
            extracted_sections = extract_periodic_sections(filing, min_section_chars=min_section_chars)
        else:
            extracted_sections = []

        for section in extracted_sections:
            rows.append(
                {
                    "date": filing["date"],
                    "ticker": filing["ticker"],
                    "doc_type": "sec_section",
                    "source": "SEC",
                    "title": section["title"],
                    "text": section["text"],
                    "form": filing["form"],
                    "section_type": section["section_type"],
                    "items": section["items"],
                    "company_name": filing["company_name"],
                    "accession_number": filing["accession_number"],
                    "source_url": filing["url"],
                    "document_id": filing["accession_number"],
                    "section_id": f"{filing['accession_number']}::{section['section_type']}",
                    "raw_title": filing["raw_title"],
                }
            )

    return pd.DataFrame(rows)
