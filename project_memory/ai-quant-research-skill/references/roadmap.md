# Roadmap

Last updated: 2026-03-19

## Direction Check

The user's direction is:

- build toward AI quant investing
- prefer real learned models over heuristic scoring
- move beyond infrastructure once the basics are stable

Current assessment:

- infrastructure progress: strong
- genuine AI quant progress: still early
- direction alignment: partial, but not complete

## Recommended Next Phase

The next phase should be:

`shared-core expansion plus KR market-adapter follow-through on top of the promoted transformer lane`

The repo now has a wider 60-name candidate pool, a generated free-approx monthly snapshot lane, repeated OOS windows, point-in-time eligibility filters, TCN/hybrid/LSTM controls, a transformer-style follow-up, and a focused transformer sweep tool. The strongest current transformer variants now use a clipped target objective on top of the smaller transformer with a 252-trading-day rolling training window. The refreshed standard reports confirm the split: `seq40 + clip_q95` remains strongest on the latest strict 1-year report, while `seq20 + clip_q95` remains strongest on repeated-window average. The standard report stack now has a canonical rerun path, and the first shared-core split exists in `invest_ai_core`. True constituent-history data is still backlog, but the immediate next step is now `shared-core expansion plus KR adapter build-out`, not more random model-family expansion and not RL.

## Ordered Next Steps

1. Continue hardening the evaluation framework.
   - keep purged walk-forward evaluation as the default supervised path
   - keep repeated OOS windows instead of relying on one latest 1-year slice
   - improve execution realism beyond fixed bps costs
   - keep manifests on every serious experiment

2. Improve universe realism before adding more model complexity.
   - keep the dynamic 60-name plus eligibility lane as the standing control
   - keep the phased 30-name lane as a secondary comparison lane
   - still treat better constituent-history data as a backlog item
   - keep caching and provenance around universe inputs so runs are reproducible
   - keep retry/backoff in the fetch layer so long experiments fail less often
   - keep the feature lane benchmark-aware and universe-aware

3. Harden the strongest sequence model now that TCN, hybrid, LSTM, and transformer controls exist.
   - keep the canonical report stack as the default validation path for the promoted clipped-objective defaults
   - keep the clipped-objective transformer as the current best candidate family
   - treat `seq20` as the safer repeated-window default and `seq40` as the more aggressive latest-window default until new evidence overturns that split
   - only revisit RL after the price-model lane is clearly credible

4. Build on the first shared-core split so KR can reuse the same research engine.
   - keep `invest_ai_core.config` and `invest_ai_core.market_data` as the initial market-agnostic surface
   - extend the shared core into evaluation/report helpers only where the interface is genuinely market-agnostic
   - keep SEC and future DART logic in market adapters, not in the shared core

5. Reposition LLM signals as auxiliary features.
   - use them as an input to the price model
   - do not let them become the center of the research program yet
   - eventually decompose `llm_score` into structured subfactors instead of one scalar

6. Move RL to the end.
   - only after price-model alpha and validation quality are credible
   - focus RL on sizing or allocation, not first-pass alpha discovery

## Default Decision Rules

When choosing between two possible next tasks:

- choose the one that improves validation rigor over convenience
- choose price-model work over more SEC/LLM prompt tuning
- choose interpretable baselines before complex architectures
- choose shared-core reuse over market-specific copy-paste
- choose one clean benchmark experiment over multiple shallow experiments

## What "Good Progress" Looks Like Next

The project will be moving in the right direction if the next implemented milestone produces:

- a standard report stack that uses the best clipped-objective transformer candidate rather than the older raw objective
- a canonical rerun path that refreshes the latest-year and repeated-window artifacts without ad hoc commands
- repeated window comparisons that clarify whether `seq20` or `seq40` is the better production research default
- a clearer answer on whether the remaining gap is now calibration/objective choice, feature design, or still mostly data realism
- continued reproducibility via manifests that include the dynamic snapshot builder sidecar
- learned-model results that remain credible under the stricter dynamic-universe and eligibility constraints
