# DART Scaffold

This file defines the future DART adapter boundary for Korea-market research.

## Intended Inputs

- Filing metadata
- Filing body text
- Filing date and time
- Issuer code and issuer name
- Filing type and subtype

## Intended Outputs

- Normalized document rows for scoring
- Filing-category labels
- Event timestamp aligned to the trading calendar
- Evidence fields for later auditability

## Filing Categories To Reserve

- Annual report
- Semiannual report
- Quarterly report
- Earnings-related disclosure
- Major contract disclosure
- Capital increase or dilution events
- Governance and board-change notices

## Implementation Notes

- DART should be treated as an ingestion adapter, not as a strategy.
- Event timing must be aligned with the Korean trading calendar before any backtest.
- Filings that arrive after market close should not be treated as same-session signals without explicit handling.
