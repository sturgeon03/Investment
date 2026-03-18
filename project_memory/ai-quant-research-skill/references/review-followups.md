# Review Follow-Ups

Last updated: 2026-03-19

This file tracks which external review findings were already addressed in the local workspace and which ones remain valid backlog.

## Fixed Or Hardened Now

- Added regression coverage that `llm_score` survives the walk-forward learning frame after `attach_llm_scores`.
- Confirmed the local workspace strategy path already preserves LLM scores and still includes the price-vs-LLM override test in `test_strategy.py`.
- Backtests now apply `cash_buffer` and `max_position_weight`, reducing the gap between research weights and paper portfolio sizing.
- Backtest summaries now report `sortino`, `calmar`, `tracking_error`, `information_ratio`, and excess-return fields in addition to the existing return/risk metrics.
- Added run/report manifests with config and signal provenance hashes.
- Added a strong tree-based tabular control and confirmed it still underperforms the current price baseline on the latest 1-year OOS window.
- Expanded the default supervised feature set with richer price and volume signals; learned models improved but still trail the rules baseline.
- Added benchmark-relative and regime-aware features; ridge improved sharply and is now close to the rules baseline.
- Added a reusable 30-name large-cap universe via `tickers_file`; on the wider universe the learned models lag the rules baseline more clearly.
- Added sector-aware and cross-sectional context features; on the wider universe they did not improve the learned models enough to beat the rules baseline.
- Added phased universe snapshots through `universe_snapshots_file`; on the same last-year OOS window tree and MLP improved materially versus the static 30-name setup, while ridge worsened.
- Added a repeated-window stability report; the rules baseline still leads on average, ridge remains the strongest learned model overall, and phased universe history meaningfully changes learned-model rankings.
- Added `market_data_manifest.json` and request-signature cache validation, so reused raw data is now explicit and harder to misuse across mismatched universe definitions.
- Added retry/backoff hardening on `yfinance` and SEC fetch paths, plus regression tests for transient failures.
- Added point-in-time eligibility filters; these improved the latest learned-model results materially, but the rules baseline still leads on repeated-window averages.
- Added a free-approx dynamic monthly snapshot builder with sidecar manifests and linked those manifests into run/report provenance.
- Re-ran the stricter dynamic 60-name lane; the latest 1-year window still favors the rules baseline, but the tree model now leads the repeated-window average and beats the baseline in `3/4` windows.
- Added the first temporal convolution baseline; it underperforms the rules baseline and tree model, but it establishes a real sequence-model control in the stricter lane.
- Added a hybrid sequence-plus-static baseline; it improves materially on TCN and edges past MLP on repeated-window average, but it still trails tree, the rules baseline, and ridge.
- Added an LSTM recurrent baseline; it underperforms hybrid and does not improve on the current sequence controls, which is useful negative evidence against recurrence-only changes.
- Added a transformer-style baseline with a rolling training window; it is the first learned model in the repo to beat the rules baseline on the latest strict dynamic-lane 1-year report and in a focused repeated-window average comparison.
- Added focused transformer sweep tooling and confirmed the current best combo is a smaller transformer with a 252-trading-day rolling training window.
- Extended the focused transformer sweep to cover sequence-lookback length and target-clipping objective design, and confirmed that clipped targets beat the raw objective while `seq40` wins the latest-year slice and `seq20` wins repeated-window average.
- Added report-stack plumbing so the standard last-year report can take the stronger `seq40 + clip_q95` default and the repeated-window report can take the safer `seq20 + clip_q95` default.
- Added the first `kr_stocks` market-adapter scaffold instead of leaving Korea as a placeholder only.

## Still Valid Backlog

- Move from free-approx dynamic snapshots toward better point-in-time constituent history.
- Replace scalar `llm_score` with structured subfactors such as guidance, demand, legal risk, and liquidity pressure.
- Improve execution realism beyond fixed bps costs with spread/slippage and liquidity-aware assumptions.
- Promote the clipped-objective transformer winner into the standard report stack and resolve the `seq20` versus `seq40` tie-break.
- Rerun the standard reports on top of the new transformer defaults and start the first shared-core split between market-agnostic research code and US-specific adapters.

## Interpretation Rule

If an external review conflicts with the local workspace, trust the local code and tests first. Treat the review as a prioritization input, not as ground truth.
