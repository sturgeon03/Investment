# KR Market Architecture

The Korea-market implementation should be built as an adapter around a shared research core, not as a direct copy of the US pipeline.

## Shared Candidates

- Walk-forward evaluation logic
- Backtest and portfolio accounting
- Experiment manifests and provenance
- Report generation
- Feature interface contracts

## Korea-Specific Layers

- Market calendar and session rules
- Ticker normalization and exchange codes
- DART filing ingestion and document parsing
- Fee and tax assumptions
- Vendor-specific price and universe history

## Boundary Rule

If a module encodes Korea-specific market rules, it belongs under `kr_stocks/`.
If a module is market-agnostic and reusable, it should eventually move into a shared core.

## Current Status

The repo now has a first shared package for market-agnostic config and market-data logic in `invest_ai_core`.

This folder still only defines the Korea-specific scaffold and assumptions. No production Korea data adapter or alpha model is implemented yet.
