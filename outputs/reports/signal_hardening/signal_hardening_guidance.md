# Signal Hardening Guidance

## Inputs
- last-year summary: `us_stocks/artifacts/deep_learning_large_cap_60_dynamic_seq40_clip_q95_last_year/deep_learning_summary_last_year.csv`
- repeated-window summary: `us_stocks/artifacts/stability_large_cap_60_dynamic_seq20_clip_q95/stability_window_summary.csv`

## Robustness leaderboard
- `transformer_walkforward`: tier `promote`, average gap vs baseline `$5,123`, worst gap `$-12,020`, beat rate `50%`.
- `tree_walkforward`: tier `promote`, average gap vs baseline `$3,801`, worst gap `$-2,285`, beat rate `75%`.
- `configured_baseline`: tier `deprioritize`, average gap vs baseline `$0`, worst gap `$0`, beat rate `0%`.

## Degradation readout
- Top repeated-window model: `transformer_walkforward` with average gap vs baseline `$5,123`.
- Use repeated-window gap persistence and worst-window damage as the primary promotion gate.
- Treat positive latest-window spread without repeated-window support as a challenger, not a promotion.

## Transformer lane
- Repeated-window transformer gap vs baseline averages `$5,123` with a worst window of `$-12,020`.
- Latest promoted report gap vs baseline is `11,353` dollars; the delta versus the stability latest window is `396`.
- Keep `seq40 + clip_q95` for the headline latest-year view and `seq20 + clip_q95` for repeated-window control until a single setup wins both tests.

## Recommended follow-up
- Add the same degradation tracking to future promoted runs and block new model promotions when worst-window gap falls below the selected tolerance.
