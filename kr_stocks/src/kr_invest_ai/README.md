# kr_invest_ai

Placeholder package for a future Korea-market research engine.

The first shared-core split now exists in `invest_ai_core`, starting with:

- `invest_ai_core.config`
- `invest_ai_core.market_data`

KR-specific work should build adapters around that shared core instead of copying `us_invest_ai`.

Current executable modules:

- `market_config.py`: loads KR YAML assumptions into typed dataclasses.
- `tickers.py`: normalizes canonical 6-digit listing codes and vendor-style tickers.
- `calendar.py`: derives regular-session windows and filing-session alignment rules from KR config.
- `dart_adapter.py`: normalizes simple DART-style filing rows into typed filing records with category and session-date alignment.
- `dart_client.py`: builds DART list API requests, fetches filing rows, and routes them through the normalizer.
- `market_data_client.py`: builds a typed daily OHLCV request, fetches provider bars, and normalizes them into a research-ready tabular shape.
- `data_bundle.py`: assembles typed KR price and DART requests into one research-ready bundle with normalized DataFrame outputs.
- `pipeline.py`: adds cache/provenance around the bundle and saves normalized `prices.csv`, `filings.csv`, and a raw manifest.
- `main.py`: runnable CLI entry that executes the first KR data pipeline and writes a run manifest.
- `features.py`: turns KR price bars, optional benchmark bars, and filing-event counts into a first research-ready feature frame.
- `strategy.py`: applies a first monthly KR ranking rule over momentum, volatility, and filing-event context.
- `research.py`: runs the first KR research/backtest loop and saves artifacts.
- `research_main.py`: runnable CLI entry for the first KR research lane.
- `walkforward.py`: adds the first KR walk-forward split logic for learned-model validation.
- `ml_strategy.py`: adds a first KR ridge baseline plus a stricter KR walk-forward ridge path that trains only on already-known forward labels.
- `report_main.py`: compares the KR rules baseline against the first walk-forward ridge baseline and writes report artifacts.

Suggested submodules:

- `calendar`: KRX holiday/session handling
- `universe`: point-in-time ticker and exchange membership
- `data`: vendor adapters and cache/provenance
- `filings`: DART adapters and filing normalization
- `features`: market-specific feature engineering
- `strategy`: research strategies and baselines
- `backtest`: evaluation and portfolio accounting
- `report`: comparison and artifact generation

This package now has minimal adapter code for KR config, ticker, calendar, DART-row normalization, a first DART list client, a first historical daily market-data client, a first research-ready bundle assembler, a first executable raw-data pipeline with cache/provenance support, a first benchmark-aware feature-assembly layer on top of that raw pipeline, and a first rules-vs-walk-forward-ridge comparison path with optional benchmark handling.

Research-lane example:

```powershell
$env:PYTHONPATH='C:\Users\sym89\Desktop\Investment\kr_stocks\src;C:\Users\sym89\Desktop\Investment\us_stocks\src'
python -m kr_invest_ai.research_main --tickers 005930 000660 --price-start-date 2026-01-01 --price-end-date 2026-03-19
```

Comparison example:

```powershell
$env:PYTHONPATH='C:\Users\sym89\Desktop\Investment\kr_stocks\src;C:\Users\sym89\Desktop\Investment\us_stocks\src'
python -m kr_invest_ai.report_main --tickers 005930 000660 --benchmark-ticker 069500 --price-start-date 2026-01-01 --price-end-date 2026-03-19
```

Minimal runnable example:

```powershell
$env:PYTHONPATH='C:\Users\sym89\Desktop\Investment\kr_stocks\src;C:\Users\sym89\Desktop\Investment\us_stocks\src'
python -m kr_invest_ai.main --tickers 005930 --price-start-date 2026-01-01 --price-end-date 2026-03-19
```
