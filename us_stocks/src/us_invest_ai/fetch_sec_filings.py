from __future__ import annotations

import argparse
from pathlib import Path

from us_invest_ai.config import load_config
from us_invest_ai.sec_filings import (
    DEFAULT_FORMS,
    build_sec_session,
    extract_recent_filings,
    fetch_company_tickers,
    fetch_filing_documents,
    fetch_submissions,
    lookup_companies,
    resolve_user_agent,
    save_documents,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch recent SEC filings into a raw metadata CSV.")
    parser.add_argument(
        "--config",
        default="us_stocks/config/base.yaml",
        help="Config file used to load the default ticker universe.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional explicit list of tickers. Defaults to the config universe.",
    )
    parser.add_argument(
        "--forms",
        default="10-K,10-Q,8-K",
        help="Comma-separated SEC form list.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional filing date floor in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--limit-per-ticker",
        type=int,
        default=3,
        help="Maximum number of filings to fetch per ticker after form/date filtering.",
    )
    parser.add_argument(
        "--output-csv",
        default="us_stocks/documents/sec_filings_recent.csv",
        help="Output CSV path for raw filing metadata and cleaned raw text.",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help="SEC user agent string. If omitted, SEC_USER_AGENT is used.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
        help="Pause between filing downloads to stay polite with SEC infrastructure.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200000,
        help="Maximum characters to keep per filing document.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    tickers = args.tickers or config.data.tickers
    forms = tuple(form.strip().upper() for form in args.forms.split(",") if form.strip()) or DEFAULT_FORMS
    user_agent = resolve_user_agent(args.user_agent)

    session = build_sec_session(user_agent)
    companies = fetch_company_tickers(session)
    matched_companies = lookup_companies(companies, tickers)

    filings = []
    for _, company in matched_companies.iterrows():
        submissions = fetch_submissions(session, int(company["cik_str"]))
        filings.extend(
            extract_recent_filings(
                submissions=submissions,
                ticker=company["ticker"],
                forms=forms,
                start_date=args.start_date,
                limit_per_ticker=args.limit_per_ticker,
            )
        )

    filings = sorted(filings, key=lambda filing: (filing.filing_date, filing.ticker), reverse=True)
    documents, errors = fetch_filing_documents(
        session=session,
        filings=filings,
        max_chars=args.max_chars,
        pause_seconds=args.pause_seconds,
    )
    save_documents(documents, args.output_csv)

    print(f"Fetched {len(documents)} raw filings into: {Path(args.output_csv)}")
    if errors:
        print(f"Skipped {len(errors)} filings due to errors.")
        for error in errors[:10]:
            print(f"  {error}")


if __name__ == "__main__":
    main()
