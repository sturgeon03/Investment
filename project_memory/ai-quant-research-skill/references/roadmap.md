# Roadmap

Last updated: 2026-03-17

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

`research-grade price-based AI modeling`

This means the project should stop prioritizing new workflow utilities and instead build the first real deep learning research lane.

## Ordered Next Steps

1. Harden the evaluation framework.
   - add purged walk-forward evaluation
   - keep the last year as a fixed final test window
   - add repeatable train/validation/test split logic

2. Build the first real deep learning baseline.
   - start with MLP on tabular price features
   - compare directly against ridge and the current rules baseline
   - require out-of-sample reports, not only in-sample improvement

3. Add sequence models only after the MLP benchmark is stable.
   - LSTM or temporal convolution first
   - transformer-based model after sequence baseline exists

4. Reposition LLM signals as auxiliary features.
   - use them as an input to the price model
   - do not let them become the center of the research program yet

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

- a true deep learning model
- strict out-of-sample evaluation
- direct comparison vs ridge and rules baseline
- a clear conclusion on whether AI is actually helping
