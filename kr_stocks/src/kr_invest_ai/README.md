# kr_invest_ai

Placeholder package for a future Korea-market research engine.

The first shared-core split now exists in `invest_ai_core`, starting with:

- `invest_ai_core.config`
- `invest_ai_core.market_data`

KR-specific work should build adapters around that shared core instead of copying `us_invest_ai`.

Suggested submodules:

- `calendar`: KRX holiday/session handling
- `universe`: point-in-time ticker and exchange membership
- `data`: vendor adapters and cache/provenance
- `filings`: DART adapters and filing normalization
- `features`: market-specific feature engineering
- `strategy`: research strategies and baselines
- `backtest`: evaluation and portfolio accounting
- `report`: comparison and artifact generation

This package is now a minimal Python package marker with shared-core references, but it still does not contain a Korea data adapter, filing adapter, or executable research pipeline.
