# KR Stocks

This folder is the Korea-market counterpart to `us_stocks/`.

It is intentionally not a copied US pipeline. The Korean market needs its own data vendors, filing sources, calendar rules, fee/tax assumptions, and execution conventions.

## Current Scope

- Define Korea-specific market assumptions in one place.
- Expose a minimal executable adapter layer for KR market config and ticker normalization.
- Expose session-date alignment and regular-session helpers from KR calendar assumptions.
- Expose a first DART list client plus filing-row normalization layer.
- Expose a first historical daily OHLCV market-data client with canonical KR ticker normalization.
- Expose a first research-ready data bundle that combines KR daily prices and DART filing outputs.
- Expose a first executable KR data pipeline with bundle cache reuse, raw normalized CSV outputs, and provenance manifests.
- Expose a first KR feature-assembly layer that merges price bars, benchmark-relative context, and filing-event counts into a research-ready feature frame.
- Expose a first KR research/backtest lane that ranks names monthly and writes summary/equity/ranking artifacts.
- Expose optional benchmark handling plus a first KR walk-forward ridge baseline and comparison report with dedicated CLI coverage.
- Keep the future implementation separated from US logic while reusing the shared core where possible.

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
- Prefer small real adapters over large fake placeholders.

## Next Implementation Order

1. Extend the shared core only where the interface is genuinely market-agnostic.
2. Add reproducible fee/tax and execution assumptions to the execution layer.
3. Add a Korean-market research lane only after the adapter layer is clear.
4. Extend the first KR pipeline from raw bundle export and feature assembly into a fuller research pipeline.

## First Runnable Command

```powershell
$env:PYTHONPATH='C:\Users\sym89\Desktop\Investment\kr_stocks\src;C:\Users\sym89\Desktop\Investment\us_stocks\src'
python -m kr_invest_ai.main --tickers 005930 000660 --price-start-date 2026-01-01 --price-end-date 2026-03-19 --data-dir .\kr_stocks\data --artifacts-dir .\kr_stocks\artifacts
```

Add `--use-dart --corp-code-map-csv <path>` when DART credentials and corp-code mapping are available.

First research-lane command:

```powershell
$env:PYTHONPATH='C:\Users\sym89\Desktop\Investment\kr_stocks\src;C:\Users\sym89\Desktop\Investment\us_stocks\src'
python -m kr_invest_ai.research_main --tickers 005930 000660 --price-start-date 2026-01-01 --price-end-date 2026-03-19 --artifacts-dir .\kr_stocks\artifacts\research
```

First comparison-report command:

```powershell
$env:PYTHONPATH='C:\Users\sym89\Desktop\Investment\kr_stocks\src;C:\Users\sym89\Desktop\Investment\us_stocks\src'
python -m kr_invest_ai.report_main --tickers 005930 000660 --benchmark-ticker 069500 --price-start-date 2026-01-01 --price-end-date 2026-03-19 --artifacts-dir .\kr_stocks\artifacts\report
```
