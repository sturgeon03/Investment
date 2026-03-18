# Roadmap

Last updated: 2026-03-18

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

`a stronger second sequence baseline on top of the dynamic-universe control lane`

The repo now has a wider 60-name candidate pool, a generated free-approx monthly snapshot lane, repeated OOS windows, point-in-time eligibility filters, and a first TCN baseline. That TCN result is useful but weak. True constituent-history data is still backlog, but the immediate modeling next step is now `LSTM or hybrid sequence + static context`, not more tabular variants and not RL.

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

3. Add stronger sequence models now that the first TCN control exists.
   - LSTM or hybrid sequence-plus-static context next
   - transformer-based model only after at least one stronger sequence follow-up exists

4. Reposition LLM signals as auxiliary features.
   - use them as an input to the price model
   - do not let them become the center of the research program yet
   - eventually decompose `llm_score` into structured subfactors instead of one scalar

5. Move RL to the end.
   - only after price-model alpha and validation quality are credible
   - focus RL on sizing or allocation, not first-pass alpha discovery

## Default Decision Rules

When choosing between two possible next tasks:

- choose the one that improves validation rigor over convenience
- choose price-model work over more SEC/LLM prompt tuning
- choose interpretable baselines before complex architectures
- choose one clean benchmark experiment over multiple shallow experiments

## What "Good Progress" Looks Like Next

The project will be moving in the right direction if the next implemented milestone produces:

- a stronger sequence-model comparison on the same dynamic-universe lane
- repeated window comparisons vs rules, ridge, tree, MLP, TCN, and the next sequence follow-up
- a clearer answer on whether the remaining gap is architecture choice, feature design, or still data realism
- continued reproducibility via manifests that include the dynamic snapshot builder sidecar
- learned-model results that remain credible under the stricter dynamic-universe and eligibility constraints
