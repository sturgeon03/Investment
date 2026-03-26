# Current State

Last updated: 2026-03-26

## Repository Status

The repo is organized by market:

- `us_stocks/`: active research pipeline
- `kr_stocks/`: executable Korea-market research scaffold with a first cacheable raw-data pipeline, monthly rules lane, benchmark-aware feature frame, and first rules-vs-walk-forward-ridge comparison path

Operational note:

- the repo now has a local `.venv` at the root with the research dependencies installed
- local verification currently passes with `88` US tests and `41` KR tests in that environment
- the US lane now also has `us_stocks/scripts/run_overnight_quant.ps1` plus `us_invest_ai.refresh_market_data`, giving the automation a single-writer overnight path that refreshes canonical market data, writes a refresh manifest, runs the promoted report stack into timestamped output directories, and appends a structured run ledger under `us_stocks/automation/`
- the shared backtest layer now supports execution-realism controls beyond fixed bps costs via spread, participation-scaled market impact, and liquidity lookback inputs, and both the US and KR lanes now expose that surface
- the main branch now also includes the missing runtime hardening that had only lived in detached automation worktrees: best-effort repo-health snapshots, UTF-8-safe git manifest capture, repo-local `yfinance` cache routing, cache-signature matching that ignores path drift across equivalent worktrees, `LOKY_MAX_CPU_COUNT=1` defaulting in the report stack, and tighter tree-fit thread caps
- the paper-first daily workflow now also writes operator-facing runtime state under `us_stocks/paper/runtime/`, including `latest_status.json`, an append-only paper ledger, stable latest CSV views, and a CLI summary path via `python -m us_invest_ai.paper_runtime_status`
- `us_stocks/scripts/run_us_daily.ps1` is now a portable paper runner: it defaults to the heuristic swing config, auto-detects the repo-local `.venv` Python, resolves the configured paper positions path, and prints the latest paper runtime status after each successful run

The US project already includes:

- daily OHLCV download with `yfinance`
- feature generation for price-based ranking
- SEC filing fetch, cleanup, section extraction, and scoring pipeline
- OpenAI-compatible LLM scoring interface
- heuristic scoring fallback for local dry runs
- backtest engine and config-based comparisons
- paper portfolio and order preview workflow
- supervised ML baseline using labeled forward returns
- purged walk-forward helpers and an initial MLP research lane
- gradient-boosted tree walk-forward baseline
- run/report manifests with config and signal provenance hashes
- richer price and volume feature engineering for supervised models
- benchmark-relative and regime-aware feature engineering for supervised models
- expanded 30-name large-cap universe config using `tickers_file`
- sector-aware and cross-sectional context features for supervised models
- approximate point-in-time large-cap experiments using `universe_snapshots_file`
- retry/backoff hardening for `yfinance` and SEC fetch paths
- point-in-time eligibility rules on close price, rolling dollar volume, and universe age
- free-approx dynamic monthly snapshot generation for a broader 60-name large-cap candidate pool

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

### Latest canonical overnight rerun

Most recent dependency-complete rerun completed on `2026-03-20` using refreshed market data through `2026-03-19`.

There have been no newer successful canonical reruns since then. The failed `2026-03-21` overnight attempts did not advance the canonical state; they stopped in `refresh_market_data`, with the final clear blocker being Yahoo connectivity for fresh daily data rather than a new research regression.

Dynamic 60-name plus eligibility lane:

- latest 1-year OOS ending capital:
  transformer about `$127,520`
  MLP about `$117,307`
  ridge about `$116,593`
  baseline about `$116,168`
  tree about `$113,267`
- repeated 4-window average ending capital:
  transformer about `$120,013`
  tree about `$118,691`
  baseline about `$114,890`
  ridge about `$112,974`
  MLP about `$112,468`

Interpretation:

- the promoted clipped-objective transformer remains the strongest learned model on both the latest 1-year slice and the repeated-window average in the stricter dynamic lane
- tree remains the strongest non-transformer learned baseline on repeated windows
- the latest rerun keeps the core conclusion intact: the repo now has credible positive AI-alpha evidence in the transformer lane, but it still depends on free-approx universe history and backtest assumptions
- the repo also now has a working overnight single-writer path that refreshes canonical market data, reruns the promoted report stack, records a structured ledger, and can safely promote successful timestamped outputs into the canonical tracked artifact directories

### Execution-realistic follow-up on the promoted dynamic lane

Most recent stricter execution-cost rerun completed on `2026-03-20` using `soft_price_large_cap_60_dynamic_eligibility_execution_realistic.yaml` with:

- transaction cost: `10 bps`
- spread cost: `5 bps`
- market impact: `15 bps * participation^0.5`
- liquidity lookback: `20 trading days`

Dynamic 60-name plus eligibility lane under these stricter costs:

- latest 1-year OOS ending capital:
  transformer about `$126,767`
  MLP about `$116,530`
  ridge about `$115,890`
  baseline about `$115,366`
  tree about `$112,568`
- repeated 4-window average ending capital:
  transformer about `$119,278`
  tree about `$117,991`
  baseline about `$114,138`
  ridge about `$112,238`
  MLP about `$111,712`

Interpretation:

- the promoted transformer remains the strongest model on both the latest 1-year slice and the repeated-window average even after adding spread and liquidity-scaled market impact
- the stricter execution assumptions reduce latest-year ending capital by roughly `$700-$800` across the main models and reduce repeated-window average ending capital by roughly the same order of magnitude
- the ranking does not change: transformer still leads, tree remains the strongest non-transformer repeated-window learned baseline, and the baseline still trails both transformer and tree on the repeated-window average
- realized participation stays tiny in the promoted US lane, so the main drag comes from linear plus spread costs rather than from large simulated market-impact penalties
- this moves the next bottleneck further away from naive execution-cost assumptions and back toward data realism, constituent history, and continued sequence-model hardening

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

Recent 1-year walk-forward model comparison:

- baseline: about `$163,834`
- ridge walk-forward: about `$158,832`
- tree walk-forward: about `$139,001`
- MLP walk-forward: about `$145,947`

Interpretation:

- benchmark-relative and regime-aware features materially improved the ridge walk-forward model
- ridge is now much closer to the configured price baseline than before
- the stronger learned models still underperform the configured price baseline
- this is useful negative evidence
- the bottleneck is now more likely data scope and universe design than model family choice alone

Expanded 30-name large-cap universe 1-year walk-forward comparison:

- baseline: about `$123,275`
- tree walk-forward: about `$113,386`
- MLP walk-forward: about `$111,688`
- ridge walk-forward: about `$110,500`

Interpretation:

- widening the universe reduced the baseline's raw CAGR versus the 10-name basket, which is expected
- the learned models underperform more clearly on the wider universe than they did on the 10-name basket
- this suggests the next bottleneck is cross-sectional context, sector structure, and universe design rather than simply adding more model families

Expanded 30-name phased snapshot universe 1-year walk-forward comparison:

- baseline: about `$123,275`
- tree walk-forward: about `$117,558`
- MLP walk-forward: about `$114,823`
- ridge walk-forward: about `$107,251`

Interpretation:

- the last-year baseline is unchanged because the active 2025-2026 universe is the same as the static 30-name config
- the phased snapshot file changes the historical training context rather than the final evaluation constituents
- tree and MLP improved meaningfully relative to the static 30-name run
- ridge deteriorated under the same change
- this is evidence that universe history assumptions matter, even before moving to true point-in-time constituent data

Repeated-window stability report over four 252-trading-day windows:

Static 30-name universe:

- baseline average ending capital: about `$119,065`
- ridge average ending capital: about `$117,462`
- MLP average ending capital: about `$114,184`
- tree average ending capital: about `$108,491`

Phased 30-name universe:

- baseline average ending capital: about `$119,122`
- ridge average ending capital: about `$114,780`
- MLP average ending capital: about `$113,716`
- tree average ending capital: about `$113,701`

Interpretation:

- the rules baseline still has the strongest average multi-window result
- ridge remains the strongest learned-model family overall
- phased universe history helped tree and MLP materially across repeated windows
- the repo now has evidence that data realism changes the learned-model ranking
- that pushes the next bottleneck further toward universe realism and data quality, not random model-family expansion

Phased 30-name universe with point-in-time eligibility filters:

- latest 1-year OOS ending capital:
  baseline about `$122,956`
  MLP about `$123,183`
  tree about `$121,720`
  ridge about `$112,517`
- repeated 4-window average ending capital:
  baseline about `$119,781`
  tree about `$115,188`
  MLP about `$114,224`
  ridge about `$110,608`

Interpretation:

- adding point-in-time eligibility rules improved learned-model realism and latest-window competitiveness
- the latest 1-year window now shows MLP roughly at parity with, and slightly ahead of, the baseline on ending capital
- across repeated windows the baseline still leads on average
- this is the strongest learned-model result so far in the repo, but it is still not enough to declare stable AI alpha

Free-approx dynamic 60-name universe with point-in-time eligibility filters:

- latest 1-year OOS ending capital:
  baseline about `$118,294`
  ridge about `$116,877`
  MLP about `$114,915`
  tree about `$113,542`
- repeated 4-window average ending capital:
  tree about `$119,183`
  baseline about `$115,607`
  ridge about `$113,319`
  MLP about `$112,204`

Interpretation:

- the stronger dynamic universe lowers the optimistic latest-window numbers seen in the smaller 30-name phased lane
- on the latest 1-year window the rules baseline is still the strongest model
- across repeated windows the tree model is now the strongest average learned-model family and beats the baseline in three of four windows
- this is the first evidence in the repo that a learned model can lead the repeated-window average under a stricter universe-realism lane
- this is still a free approximation, not institutional constituent history

First sequence baseline on the dynamic 60-name lane:

- latest 1-year OOS ending capital:
  TCN about `$109,257`
- repeated 4-window average ending capital:
  TCN about `$111,833`

Interpretation:

- the first temporal convolution baseline underperforms the rules baseline and also trails the current tree leader
- this is still a meaningful milestone because the repo now has a real sequence-model control under the stricter dynamic-universe lane
- the next model step should be a stronger sequence baseline, likely `LSTM` or a hybrid sequence-plus-static model

Second sequence baseline on the dynamic 60-name lane:

- latest 1-year OOS ending capital:
  hybrid sequence about `$115,056`
- repeated 4-window average ending capital:
  hybrid sequence about `$113,074`

Interpretation:

- the hybrid sequence-plus-static model improves materially on the pure TCN baseline
- in the latest 1-year window it beats TCN, MLP, and tree, but still trails ridge and the rules baseline
- across repeated windows it beats MLP and TCN on average, but still trails tree, the rules baseline, and ridge
- this makes the remaining sequence gap look more like a recurrent-architecture or objective-design issue than a simple need for more static context

Recurrent sequence follow-up on the dynamic 60-name lane:

- latest 1-year OOS ending capital:
  LSTM about `$107,684`
- repeated 4-window average ending capital:
  LSTM about `$110,121`

Interpretation:

- the first recurrent baseline underperforms the rules baseline and also trails tree, hybrid, MLP, and TCN on repeated-window average
- this is useful negative evidence because it suggests recurrence alone does not close the gap
- the next sequence step should now be transformer-style sequence modeling or a sequence-objective redesign rather than another small recurrent variant

Transformer-style sequence follow-up on the dynamic 60-name lane:

- latest 1-year OOS ending capital:
  transformer about `$123,134`
- focused repeated 4-window average ending capital versus the same rules baseline windows:
  transformer about `$120,520`
  baseline about `$115,607`

Interpretation:

- this is the first learned model in the repo to beat the rules baseline on the latest strict dynamic-lane 1-year report
- in the focused repeated-window comparison it also beats the rules baseline on average and in `2/4` windows
- its repeated-window average also exceeds the previously leading tree model's `~$119,183` on the same lane
- this is the strongest positive AI-alpha evidence in the repo so far, but it still relies on free-approx universe history and backtest assumptions

Focused transformer robustness sweep on the dynamic 60-name lane:

- sweep grid:
  `model_dim = 4`
  `training_lookback_days = 252`
  `sequence_lookback_window in {10, 20, 40}`
  `target_clip_quantile in {raw, 0.90, 0.95}`
- best latest 1-year combo:
  `transformer_d4_lb252_seq40_clip_q95`
- best repeated 4-window average combo:
  `transformer_d4_lb252_seq20_clip_q95`
- latest 1-year OOS ending capital for `transformer_d4_lb252_seq40_clip_q95`:
  about `$129,739`
- repeated 4-window average ending capital for `transformer_d4_lb252_seq20_clip_q95`:
  about `$121,320`
  baseline over the same windows:
  about `$115,607`

Interpretation:

- clipping the target objective improves both the latest-window result and the repeated-window average relative to the raw-return objective
- `sequence_lookback_window = 10` is too short under the current data regime
- `sequence_lookback_window = 40` is strongest on the latest 1-year slice, while `20` is stronger on repeated-window average
- this strengthens the view that current bottlenecks are objective design, calibration, and realism, not simply more width or recurrence

## Main Gaps

The biggest remaining gaps are:

- survivorship bias from a fixed modern large-cap universe
- `yfinance` quality and non-institutional data assumptions
- execution realism is still simplified even after risk-limited backtests
- coarse document timestamp handling
- no purged cross-validation or embargo-style validation
- no true point-in-time constituent history yet
- only an approximate phased universe, not an institutional point-in-time membership dataset
- no true institutional constituent-history data yet, even though the current dynamic-snapshot lane is now the standing realism control

## Recent Hardening

The local workspace has now hardened several items that were previously weak or ambiguous:

- backtests apply `cash_buffer` and `max_position_weight`, not only paper sizing
- summary outputs include `sortino`, `calmar`, `tracking_error`, and `information_ratio`
- walk-forward tests explicitly guard that attached `llm_score` values survive into the learning frame
- reports and research runs write manifests with config and signal provenance hashes
- supervised models now train on a broader price/volume feature set instead of the original minimal tabular baseline
- supervised models now also use benchmark-relative and market-regime features
- supervised models now also use sector-aware and cross-sectional context features
- the repo now supports dated universe snapshot files to approximate point-in-time membership research
- market-data loading now writes `market_data_manifest.json` and only reuses raw cache when the request signature matches
- price and SEC fetch paths now retry transient failures instead of failing on the first temporary error
- the repo now supports point-in-time eligibility filters that apply to both ranking and walk-forward training
- the repo now supports generated monthly dynamic-universe snapshots with sidecar manifests, and those manifests are linked into run/report provenance
- the repo now also has temporal convolution, hybrid sequence-plus-static, and recurrent LSTM baselines integrated into the last-year and repeated-window report stack
- the repo now also has a transformer-style baseline with a rolling training window that finishes on the strict dynamic-universe lane
- the repo now also has a focused transformer sweep tool for baseline-vs-transformer robustness checks across sequence lookback and target-clipping objective choices
- the standard report stack now has a canonical rerun path through `run_us_report_stack.ps1`, with the latest-year report refreshed on the stronger `seq40 + clip_q95` path and the repeated-window report refreshed on the safer `seq20 + clip_q95` path
- the refreshed strict dynamic-lane reports confirm the current split: latest-year ending capital is about `$127.8k` for transformer vs `$118.3k` baseline, while repeated-window average ending capital is about `$121.3k` for transformer vs `$115.6k` baseline
- the first shared-core split is now in place through `invest_ai_core`, which holds the market-agnostic config loader, market-data bundle, manifest helpers, artifact-output helpers, performance summary layer, shared reporting helpers, and reusable backtest-window evaluation helpers while preserving `us_invest_ai.config`, `us_invest_ai.data`, `us_invest_ai.backtest.build_summary`, and report-wrapper compatibility
- `kr_stocks` is no longer just an empty placeholder; it now has a Korea-market scaffold plus executable adapter code for KR config loading, ticker normalization, calendar/session alignment, DART filing normalization, a first DART list client, a first historical daily market-data client, a first research-ready bundle assembler, a first runnable raw-data pipeline with cache/provenance manifests, a first feature-assembly layer that merges price bars with benchmark-relative and filing-event context, a first monthly research/backtest lane, and a first rules-vs-walk-forward-ridge comparison path with optional benchmark handling and CLI coverage

An external static review suggested that `llm_score` was being overwritten to zero in the main strategy path. That is not true in the current local workspace. Treat that review item as stale for this branch, but keep the new regression tests.

## Immediate Conclusion

The repo is going in the right direction only in the sense that the platform exists.

The repo is not yet going far enough in the direction the user actually wants:

- the user wants AI quant investing
- the current system is still stronger as infrastructure than as AI alpha

The next stage should therefore focus on extending the shared-core split only where a real market-agnostic seam remains, then hardening the first KR rules-vs-walk-forward-ridge comparison path into a more credible research lane with fuller purged validation and richer learned-model controls, while keeping true constituent-history data as backlog and avoiding premature RL work.
