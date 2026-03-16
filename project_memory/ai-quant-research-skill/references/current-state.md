# Current State

Last updated: 2026-03-17

## Repository Status

The repo is organized by market:

- `us_stocks/`: active research pipeline
- `kr_stocks/`: reserved only, not yet implemented as a full pipeline

The US project already includes:

- daily OHLCV download with `yfinance`
- feature generation for price-based ranking
- SEC filing fetch, cleanup, section extraction, and scoring pipeline
- OpenAI-compatible LLM scoring interface
- heuristic scoring fallback for local dry runs
- backtest engine and config-based comparisons
- paper portfolio and order preview workflow
- supervised ML baseline using labeled forward returns

## What The Project Is Right Now

This is currently:

- a usable research platform
- a baseline experimentation environment
- not yet a professional or paper-grade AI quant stack

It is not yet:

- a research-grade point-in-time institutional dataset
- a production trading system
- a validated deep learning alpha engine

## Latest Verified Findings

### Rule-based / heuristic side

Recent 1-year backtest window:

- window: `2025-03-13` to `2026-03-13`
- best heuristic/rules result: `with_llm_long`
- start capital: `$100,000`
- end capital: about `$171,906.61`

Important interpretation:

- most of that improvement came from the price/ranking stack and soft trend logic
- the heuristic LLM signal added little
- this should not be described as strong AI alpha

### True learned-model side

Recent 1-year out-of-sample supervised ML report:

- model family: ridge regression
- label: next 20 trading day return
- start capital: `$100,000`
- end capital: about `$150,734.03`
- baseline comparison over same window: about `$171,820.66`

Interpretation:

- the first true label-trained model made money
- but it underperformed the configured baseline
- there is no evidence yet that the repo has a superior AI alpha model

## Main Gaps

The biggest remaining gaps are:

- survivorship bias from a fixed modern large-cap universe
- `yfinance` quality and non-institutional data assumptions
- simplified execution model with fixed costs only
- coarse document timestamp handling
- no purged cross-validation or embargo-style validation
- no real deep learning baseline yet

## Immediate Conclusion

The repo is going in the right direction only in the sense that the platform exists.

The repo is not yet going far enough in the direction the user actually wants:

- the user wants AI quant investing
- the current system is still stronger as infrastructure than as AI alpha

The next stage should therefore focus on rigorous model research, not on more workflow polish.
