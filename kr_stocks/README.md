# KR Stocks

This folder is the Korea-market counterpart to `us_stocks/`.

It is intentionally a scaffold, not a copied US pipeline. The Korean market needs its own data vendors, filing sources, calendar rules, fee/tax assumptions, and execution conventions.

## Current Scope

- Define Korea-specific market assumptions in one place.
- Reserve DART-oriented document ingestion placeholders.
- Reserve ticker and exchange-convention mapping.
- Keep the future implementation separated from US logic.

## Proposed Layout

```text
kr_stocks/
  README.md
  config/
    market_assumptions.yaml
    fees_tax.yaml
    ticker_conventions.yaml
    trading_calendar.yaml
  docs/
    architecture.md
    dart.md
    market_rules.md
  src/
    kr_invest_ai/
  data/
  artifacts/
  tests/
```

## Korea-Market Notes

- Tickers are not directly interchangeable with US tickers.
- KRX-listed equities, ETFs, preferred shares, and warrants need separate handling.
- DART is the primary company-filing source, but it should be treated as a document adapter layer, not as the whole pipeline.
- Trading rules, session times, holiday handling, and fee/tax assumptions are market-specific.

## Design Principles

- Keep shared research logic abstract.
- Keep market-specific rules in config and adapter layers.
- Do not duplicate the US pipeline structure unless the Korea market actually needs the same shape.
- Prefer explicit placeholders over pretending the market assumptions are already implemented.

## Next Implementation Order

1. Build a shared-core interface only if both markets need it.
2. Add a Korea data adapter for market calendar and filing ingestion.
3. Add reproducible fee/tax and execution assumptions.
4. Add a Korean-market research lane only after the adapter layer is clear.
