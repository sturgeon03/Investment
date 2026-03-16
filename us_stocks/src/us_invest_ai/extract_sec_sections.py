from __future__ import annotations

import argparse
from pathlib import Path

from us_invest_ai.sec_filings import extract_scoring_documents, load_raw_filings, save_documents


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract high-signal SEC filing sections for LLM scoring.")
    parser.add_argument(
        "--input-csv",
        default="us_stocks/documents/sec_filings_recent.csv",
        help="Raw SEC filings metadata CSV created by fetch_sec_filings.",
    )
    parser.add_argument(
        "--output-csv",
        default="us_stocks/documents/sec_sections.csv",
        help="Section-level document CSV ready for score_documents.",
    )
    parser.add_argument(
        "--min-section-chars",
        type=int,
        default=250,
        help="Minimum extracted section length to keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    filings = load_raw_filings(args.input_csv)
    sections = extract_scoring_documents(filings, min_section_chars=args.min_section_chars)
    save_documents(sections, args.output_csv)

    print(f"Loaded {len(filings)} raw filings.")
    print(f"Extracted {len(sections)} scoring sections into: {Path(args.output_csv)}")


if __name__ == "__main__":
    main()
