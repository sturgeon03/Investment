# US Stocks AI Investing Starter

This project now has a practical SEC-first research loop for US equities:

1. Download daily OHLCV data with `yfinance`.
2. Fetch raw SEC filings for the stock universe.
3. Extract high-signal sections from 8-K, 10-Q, and 10-K filings.
4. Score those sections across `short_term`, `swing`, and `long_term` horizons.
5. Attach one selected horizon to the ranking model.
6. Compare the baseline against multi-horizon LLM variants.
7. Convert the latest signal into a paper portfolio and order preview.

The project is still a research baseline, not a production trading bot.

Recent research additions:

- risk-aware backtests that apply `cash_buffer` and `max_position_weight`
- run and report manifests with config and signal hashes
- raw market-data cache and `market_data_manifest.json` provenance
- retry/backoff on `yfinance` and SEC fetch paths
- walk-forward `ridge`, `tree`, and `MLP` model lanes for out-of-sample comparison
- richer price and volume features for supervised model research
- benchmark-relative and regime-aware features for supervised model research
- sector-aware and cross-sectional context features for supervised model research
- phased universe snapshots for approximate point-in-time large-cap experiments
- free-approx dynamic monthly snapshot generation for a broader 60-name large-cap candidate pool

## Layout

- `config/base.yaml`: price-only baseline.
- `config/soft_price.yaml`: price-only strategy with a soft trend penalty instead of a hard trend filter.
- `config/soft_price_large_cap_30.yaml`: expanded 30-name large-cap research universe driven by `universes/liquid_large_cap_30.txt`.
- `config/soft_price_large_cap_30_phased.yaml`: the same 30-name universe, but activated through dated snapshot membership in `universes/liquid_large_cap_30_phased_snapshots.csv`.
- `config/soft_price_large_cap_30_phased_eligibility.yaml`: phased universe plus point-in-time eligibility filters on price, rolling dollar volume, and universe age.
- `config/soft_price_large_cap_60_dynamic_eligibility.yaml`: 60-name candidate pool plus generated monthly liquidity snapshots in `universes/generated/liquid_large_cap_60_dynamic_snapshots.csv`.
- `config/with_llm_short.yaml`: short-term LLM horizon.
- `config/with_llm_swing.yaml`: swing-horizon LLM config.
- `config/with_llm_long.yaml`: long-term LLM config.
- `runs/`: dated workflow runs with SEC inputs, merged signals, research outputs, and paper orders.
- `documents/sec_filings_recent.csv`: raw SEC filings metadata and raw text.
- `documents/sec_sections.csv`: extracted high-signal sections used for scoring.
- `signals/llm_scores.generated.csv`: aggregated multi-horizon LLM scores.
- `artifacts/`: backtest outputs and strategy comparisons.
- `universes/`: reusable ticker lists, sector metadata, and dated snapshot files for larger experiments.

## Quickstart

Install the package from the repository root:

```powershell
pip install -e .\us_stocks
```

Fetch recent SEC filings:

```powershell
$env:SEC_USER_AGENT="InvestmentResearch your_email@example.com"
python -m us_invest_ai.fetch_sec_filings --start-date 2025-01-01 --limit-per-ticker 2
```

The SEC and price-download paths now retry transient failures with backoff, so repeated research runs are less likely to fail on a single 429/5xx or temporary download issue.

Extract high-signal filing sections:

```powershell
python -m us_invest_ai.extract_sec_sections --input-csv .\us_stocks\documents\sec_filings_recent.csv --output-csv .\us_stocks\documents\sec_sections.csv
```

Score sections with a local dry-run scorer:

```powershell
python -m us_invest_ai.score_documents --provider heuristic --input-csv .\us_stocks\documents\sec_sections.csv
```

Score sections with an OpenAI-compatible API:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
python -m us_invest_ai.score_documents --provider openai-compatible --base-url https://api.deepseek.com --model deepseek-chat --api-key-env DEEPSEEK_API_KEY --input-csv .\us_stocks\documents\sec_sections.csv
```

Run the swing-horizon backtest:

```powershell
python -m us_invest_ai.main --config .\us_stocks\config\with_llm_swing.yaml
```

Build the new free-approx dynamic universe snapshots:

```powershell
python -m us_invest_ai.build_universe_snapshots --candidate-tickers-file .\us_stocks\universes\liquid_large_cap_60.txt --metadata-file .\us_stocks\universes\liquid_large_cap_60_metadata.csv --start 2018-01-01 --data-dir .\us_stocks\data_large_cap_60_dynamic_eligibility --output-csv .\us_stocks\universes\generated\liquid_large_cap_60_dynamic_snapshots.csv
```

Run the new dynamic-universe control lane:

```powershell
python -m us_invest_ai.main --config .\us_stocks\config\soft_price_large_cap_60_dynamic_eligibility.yaml
python -m us_invest_ai.deep_learning_report --config .\us_stocks\config\soft_price_large_cap_60_dynamic_eligibility.yaml --transformer-sequence-lookback-window 40 --transformer-target-clip-quantile 0.95 --output-dir .\us_stocks\artifacts\deep_learning_large_cap_60_dynamic_seq40_clip_q95_last_year
python -m us_invest_ai.stability_report --config .\us_stocks\config\soft_price_large_cap_60_dynamic_eligibility.yaml --transformer-sequence-lookback-window 20 --transformer-target-clip-quantile 0.95 --output-dir .\us_stocks\artifacts\stability_large_cap_60_dynamic_seq20_clip_q95
```

Run the promoted clipped-objective report stack with one command:

```powershell
powershell -ExecutionPolicy Bypass -File .\us_stocks\scripts\run_us_report_stack.ps1 -PythonExe "C:\Path\To\python.exe"
```

The standard latest-year report now uses `seq40 + clip_q95`, while the repeated-window report uses `seq20 + clip_q95` until new evidence overturns that split.
Add `-RunSweep` if you also want to rerun the slower focused transformer sweep.

Run the focused transformer robustness sweep:

```powershell
python -m us_invest_ai.transformer_sweep --config .\us_stocks\config\soft_price_large_cap_60_dynamic_eligibility.yaml --transformer-model-dims 4 --transformer-training-lookback-days 252 --sequence-lookback-windows 10,20,40 --target-clip-quantiles none,0.9,0.95 --output-dir .\us_stocks\artifacts\transformer_sweep_v2
```

Generate a portfolio snapshot and order preview from an optional positions file:

```powershell
python -m us_invest_ai.main --config .\us_stocks\config\with_llm_swing.yaml --current-positions-csv .\us_stocks\paper\current_positions.csv
```

Run the full daily research-to-paper workflow:

```powershell
$env:SEC_USER_AGENT="InvestmentResearch your_email@example.com"
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_swing.yaml --provider heuristic --run-label daily_check
```

Run the portable daily paper runner with repo-local defaults and an automatic paper runtime status check:

```powershell
powershell -ExecutionPolicy Bypass -File .\us_stocks\scripts\run_us_daily.ps1 -RunLabel daily_check
```

Submit the generated paper orders into the broker-shaped local paper OMS:

```powershell
powershell -ExecutionPolicy Bypass -File .\us_stocks\scripts\run_us_daily.ps1 -RunLabel daily_check -SubmitPaperOrders
```

Run the same workflow with a real OpenAI-compatible API:

```powershell
$env:SEC_USER_AGENT="InvestmentResearch your_email@example.com"
$env:DEEPSEEK_API_KEY="your-key"
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_swing.yaml --provider openai-compatible --base-url https://api.deepseek.com --model deepseek-chat --api-key-env DEEPSEEK_API_KEY --run-label deepseek_live
```

Run DeepSeek using the config defaults and `.env` file:

```powershell
Copy-Item .\us_stocks\.env.us.deepseek.example .\us_stocks\.env.us.deepseek
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_deepseek.yaml --run-label deepseek_config
```

Run a local Qwen OpenAI-compatible server from config:

```powershell
Copy-Item .\us_stocks\.env.us.qwen_local.example .\us_stocks\.env.us.qwen_local
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_qwen_local.yaml --run-label qwen_local
```

Write the paper portfolio state forward automatically:

```powershell
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_deepseek.yaml --apply-paper-orders
```

Submit the paper orders into the local OMS directly from the workflow:

```powershell
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_swing.yaml --provider heuristic --submit-paper-orders
```

Prepare an Alpaca paper env file for a real broker-backed paper adapter:

```powershell
Copy-Item .\us_stocks\.env.us.alpaca.paper.example .\us_stocks\.env.us.alpaca.paper
```

Submit the workflow orders through the Alpaca paper adapter instead of the local OMS:

```powershell
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_swing.yaml --provider heuristic --submit-paper-orders --paper-broker-backend alpaca --paper-broker-env-file .\us_stocks\.env.us.alpaca.paper
```

Require a live readiness probe before the broker-backed paper submit:

```powershell
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_swing.yaml --provider heuristic --submit-paper-orders --paper-broker-backend alpaca --paper-broker-env-file .\us_stocks\.env.us.alpaca.paper --paper-broker-live-readiness-check
```

Add submission guardrails so broker-backed paper runs refuse duplicates or oversized order sets:

```powershell
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_swing.yaml --provider heuristic --submit-paper-orders --paper-broker-backend local --max-paper-order-count 8 --max-paper-total-trade-notional 35000 --max-paper-single-order-notional 12000
```

Manually stop or resume broker-backed paper submission with the kill switch:

```powershell
python -m us_invest_ai.paper_broker_control --broker-root .\us_stocks\paper\broker --activate --reason "manual review"
python -m us_invest_ai.paper_broker_control --broker-root .\us_stocks\paper\broker --deactivate
```

Summarize paper runtime incidents and fail when any error-level issue exists:

```powershell
python -m us_invest_ai.paper_runtime_incidents --positions-path .\us_stocks\paper\current_positions.csv --fail-on-error
```

Submit a previously saved `recommended_orders.csv` file into the OMS:

```powershell
python -m us_invest_ai.paper_broker_oms --orders-csv .\us_stocks\runs\<timestamp>\paper\recommended_orders.csv --latest-prices-csv .\us_stocks\data\raw\prices.csv --positions-path .\us_stocks\paper\current_positions.csv --capital-base 100000 --transaction-cost-bps 10
```

Compare the hard baseline, soft price-only control, and all three LLM horizons:

```powershell
python -m us_invest_ai.compare_configs --eval-start 2026-01-01
```

Run local tests:

```powershell
$env:PYTHONPATH=".\us_stocks\src"
python -m unittest discover -s .\us_stocks\tests
```

## SEC Refinement Rules

The SEC pipeline is tuned for signal quality before model complexity.

- `8-K`: only `2.02`, `7.01`, and `8.01` are kept by default.
- `8-K`: `1.01` and `5.02` are excluded because they are often financing or compensation noise.
- `10-Q` and `10-K`: the extractor targets `MD&A`, `Risk Factors`, `Liquidity and Capital Resources`, and forward-looking or guidance-heavy sentence sets.
- Hidden iXBRL, table-of-contents noise, exhibit lists, and signatures are stripped before section scoring.

## Multi-Horizon Signals

The scoring output is now multi-horizon.

- Detailed document scores contain one row per section per horizon.
- Aggregated signal scores use the schema:
  `date,ticker,horizon_bucket,llm_score,document_count,section_count,avg_confidence,avg_risk_flag`
- The backtest chooses the horizon through `llm.horizon_bucket` in the config.
- `with_llm_short.yaml`, `with_llm_swing.yaml`, and `with_llm_long.yaml` all use the same strategy, universe, and costs. Only the selected LLM horizon changes.

## Outputs

Running `main.py` with LLM enabled saves:

- `summary.csv`: backtest summary for the selected config.
- `data/raw/market_data_manifest.json`: source, query, file hashes, and cache-vs-download provenance for the price inputs.
- `ranking_history.csv`: rebalance snapshots with attached LLM signals.
- `llm_scores_used.csv`: filtered signal file for the selected horizon.
- `router_training_frame.csv`: future meta-router training frame with `next_5d`, `next_20d`, and `next_60d` realized returns.
- `target_portfolio.csv`: latest risk-limited holdings with target shares and notionals.
- `recommended_orders.csv`: optional order preview if `--current-positions-csv` is provided.
- `next_positions_preview.csv`: optional post-trade holdings preview.
- `run_manifest.json`: config snapshot, input hashes, git state, and output file hashes for that run.

Running `build_universe_snapshots.py` saves:

- `universes/generated/liquid_large_cap_60_dynamic_snapshots.csv`: monthly free-approx membership with `selection_rank`, `selection_score`, `avg_dollar_volume_60`, `close`, and `universe_age_days`.
- `universes/generated/liquid_large_cap_60_dynamic_snapshots.manifest.json`: candidate-pool hashes, snapshot parameters, and the backing `market_data_manifest.json` hash.

Running `daily_workflow.py` also saves:

- `documents/sec_filings_recent.csv`: raw SEC filings fetched in that run.
- `documents/sec_sections.csv`: extracted scoring sections for that run.
- `signals/llm_scores.merged.csv`: existing signal store plus the new run's scores.
- `paper/target_portfolio.csv`: current paper target holdings.
- `paper/recommended_orders.csv`: delta versus the supplied current positions file.
- `paper/next_positions_preview.csv`: optional post-trade holdings preview.
- `paper/runtime/latest_status.json`: latest operator-facing paper runtime summary, including whether paper state advanced and whether the run bootstrapped from zero holdings.
- `paper/runtime/latest_incidents.json`: latest structured incident summary over freshness, kill switch, readiness, guardrails, and reconciliation.
- `paper/runtime/ledger/paper_run_ledger.jsonl`: append-only paper run ledger for automation and morning review.
- `paper/runtime/ledger/incidents.jsonl`: append-only incident history for operator review and automation gating.
- `paper/runtime/latest_*.csv`: stable latest copies of target holdings, recommended orders, next positions, and the current paper state.
- `paper/broker/latest_account_state.json`: broker-shaped paper account snapshot with cash, equity, fees, and fill counts.
- `paper/broker/latest_orders.csv`: latest submitted OMS orders with fill status.
- `paper/broker/latest_fills.csv`: latest paper fills.
- `paper/broker/ledger/orders.jsonl`, `fills.jsonl`, `account_snapshots.jsonl`: append-only OMS ledgers.
- `paper/broker/latest_kill_switch.json`: latest kill-switch status snapshot used by the workflow.
- `paper/broker/latest_readiness.json`: latest broker readiness evaluation, including optional live connectivity checks.
- `paper/broker/latest_guardrails.json`: latest pre-submit guardrail evaluation, including duplicate-market-date protection and order/notional limits.
- `paper/broker/latest_reconciliation.json`: latest post-submit reconciliation against `paper/current_positions.csv` and the paper runtime status.
- `paper/broker/ledger/reconciliation.jsonl`, `readiness.jsonl`, `kill_switch_events.jsonl`: append-only operational histories.
- `paper/broker/` can now be driven either by the local OMS backend or by the new Alpaca paper adapter, selected with `--paper-broker-backend`.
- `workflow_manifest.json`: workflow settings, provider/model metadata, counts, and output file hashes.

Inspect the latest paper runtime status:

```powershell
python -m us_invest_ai.paper_runtime_status --positions-path .\us_stocks\paper\current_positions.csv
```

`run_us_daily.ps1` now defaults to the heuristic swing config, auto-detects the repo-local `.venv` Python when available, resolves the configured paper positions path, and prints this runtime summary after each successful run.
When you add `-SubmitPaperOrders`, it also routes the recommended orders into the local broker-shaped paper OMS and leaves the resulting account snapshot under `paper/broker/`.
You can switch that path to Alpaca paper with `-PaperBrokerBackend alpaca -PaperBrokerEnvFile .\us_stocks\.env.us.alpaca.paper`.
You can add `-PaperBrokerLiveReadinessCheck` when you want the run to probe the live paper account before submission.
The submit path can also enforce `-MaxPaperOrderCount`, `-MaxPaperTotalTradeNotional`, and `-MaxPaperSingleOrderNotional` through `daily_workflow.py`, and the runtime status now reports broker kill-switch, readiness, guardrail, and reconciliation outcomes.
You can add `-FailOnPaperIncident` when the wrapper should exit non-zero on any error-level paper incident after the run.

## Comparison Metrics

`compare_configs` now reports:

- `signal_coverage`: share of rebalance rows with a non-zero attached LLM signal.
- `avg_llm_abs_score`: average absolute LLM signal size during the evaluation window.
- `changed_rebalance_count`: number of rebalance dates that differ from the baseline.
- backtest summaries now also include `sortino`, `calmar`, and `information_ratio` when a benchmark is present.

Use `soft_price.yaml` as the control when you want to separate the impact of `soft trend filtering` from the impact of `LLM signals`.

## Model Research Reports

The supervised model reports now write a `report_manifest.json` alongside their CSV and SVG outputs.

- `us-invest-ai-ml-report`: rule baseline vs supervised ridge
- `us-invest-ai-dl-report`: rule baseline vs walk-forward ridge, gradient-boosted tree, MLP, TCN, hybrid sequence, LSTM, and transformer
- `us-invest-ai-stability-report`: repeated OOS windows across rules, ridge, tree, MLP, TCN, hybrid sequence, LSTM, and transformer
- `us-invest-ai-transformer-sweep`: focused baseline-vs-transformer robustness sweep across model size and rolling training-window choices

Those report manifests now also include the `market_data_manifest.json` path and hash, so repeated runs can tell whether they used a matching raw cache or a fresh download.

The supervised model lane now uses a richer default feature set, including:

- multi-horizon returns (`1d`, `5d`, `20d`, `60d`, `120d`)
- rolling volatility and volatility-ratio features
- moving-average distance and moving-average spread features
- rolling drawdown features
- rolling volume and dollar-volume features
- average intraday range features
- benchmark-relative return, volatility, and market-regime features
- universe-relative and sector-relative context features
- optional phased universe membership via `data.universe_snapshots_file`
- optional point-in-time eligibility filters via `eligibility.min_close_price`, `eligibility.min_dollar_volume_20`, and `eligibility.min_universe_age_days`

The current wider-universe finding is important:

- static 30-name universe, last 1-year OOS: baseline `$123,275`, tree `$105,305`, MLP `$109,814`, ridge `$111,471`
- phased 30-name snapshot universe, same last 1-year OOS: baseline `$123,275`, tree `$117,558`, MLP `$114,823`, ridge `$107,251`

Interpretation:

- the last-year baseline is unchanged because the active 2025-2026 universe is the same in both configs
- phased membership changes the training history, not the final-year constituents
- tree and MLP improved materially under the phased history
- the rules baseline is still the strongest model in this repo

The multi-window stability report now gives the stricter answer:

- static 30-name, 4 repeated 252-trading-day windows: baseline average ending capital `$119,065`, ridge `$117,462`, MLP `$114,184`, tree `$108,491`
- phased 30-name, same repeated windows: baseline average ending capital `$119,122`, ridge `$114,780`, MLP `$113,716`, tree `$113,701`

Interpretation:

- the configured rules baseline is still the best average performer
- ridge is still the strongest learned model overall
- phased universe history helped tree and MLP a lot, but not enough to beat the rules baseline on average
- the repo is now at the stage where better universe realism matters more than adding random new model classes

Run the repeated-window stability report:

```powershell
python -m us_invest_ai.stability_report --config .\us_stocks\config\soft_price_large_cap_30_phased.yaml --output-dir .\us_stocks\artifacts\stability_large_cap_30_phased
```

The new eligibility lane extends that same realism push:

- phased + eligibility, latest 1-year OOS ending capital: baseline `$122,956`, MLP `$123,183`, tree `$121,720`, ridge `$112,517`
- phased + eligibility, repeated 4-window average ending capital: baseline `$119,781`, tree `$115,188`, MLP `$114,224`, ridge `$110,608`

Interpretation:

- point-in-time eligibility rules improved the latest learned-model results materially
- the latest 1-year window now has MLP roughly at parity with, and slightly above, the rules baseline on ending capital
- across repeated windows the rules baseline still leads, so this is progress but not victory
- the next gain is still more likely to come from better constituent-history assumptions than from random model-family expansion

The new dynamic 60-name lane is the stricter control:

- dynamic 60-name + eligibility, latest 1-year OOS ending capital: baseline `$118,294`, ridge `$116,877`, MLP `$114,915`, tree `$113,542`
- dynamic 60-name + eligibility, repeated 4-window average ending capital: tree `$119,183`, baseline `$115,607`, ridge `$113,319`, MLP `$112,204`

Interpretation:

- the broader dynamic universe lowers the latest-window results versus the smaller 30-name phased lane, which is the expected realism penalty
- on the latest 1-year window the configured rules baseline is still the strongest model
- across repeated windows the tree model now leads on average and beats the baseline in `3/4` windows
- this is the first learned-model family in the repo to lead the repeated-window average under a stricter universe-realism lane
- it is still only a free approximation, not institutional point-in-time constituent history

The first sequence baseline is now also in place on that same lane:

- dynamic 60-name + eligibility + TCN, latest 1-year OOS ending capital: `TCN $109,257`
- dynamic 60-name + eligibility + TCN, repeated 4-window average ending capital: `TCN $111,833`

Interpretation:

- the first temporal convolution baseline does not beat the rules baseline, ridge, or the current tree leader
- this is still useful because it gives the repo a real sequence-model control instead of only tabular learned models
- the next sequence step should be `LSTM or hybrid sequence + static context`, not RL and not more workflow work

The second sequence baseline is now also in place on that same lane:

- dynamic 60-name + eligibility + hybrid sequence, latest 1-year OOS ending capital: `hybrid $115,056`
- dynamic 60-name + eligibility + hybrid sequence, repeated 4-window average ending capital: `hybrid $113,074`

Interpretation:

- the hybrid sequence model materially improves on the pure TCN baseline
- in the latest 1-year window it beats TCN, MLP, and tree, but still trails ridge and the rules baseline
- across repeated windows it beats MLP and TCN on average, but it still trails the tree leader, the rules baseline, and ridge
- the next model step should now be a recurrent sequence follow-up, most likely `LSTM`, not RL

The recurrent sequence follow-up is now also in place on that same lane:

- dynamic 60-name + eligibility + LSTM, latest 1-year OOS ending capital: `LSTM $107,684`
- dynamic 60-name + eligibility + LSTM, repeated 4-window average ending capital: `LSTM $110,121`

Interpretation:

- the first recurrent baseline underperforms not only the rules baseline and tree model, but also hybrid, MLP, and TCN on repeated-window average
- this is still useful negative evidence because it shows that adding recurrence alone does not close the gap
- the next model step should now be a transformer-style sequence baseline or a sequence-objective redesign, not RL

The transformer-style sequence follow-up is now also in place on that same lane:

- dynamic 60-name + eligibility + transformer, latest 1-year OOS ending capital: `transformer $123,134`
- dynamic 60-name + eligibility + transformer, focused repeated 4-window average ending capital: `transformer $120,520` vs baseline `$115,607`

Interpretation:

- this is the first learned model in the repo to beat the rules baseline on the latest strict 1-year dynamic-lane report
- in the focused repeated-window comparison against the same baseline windows, it also beats the rules baseline on average and in `2/4` windows
- its repeated-window average also exceeds the previously leading tree model's `~$119,183` on the same lane
- the next stage should now focus on hardening the transformer lane and adding lighter-weight focused comparison tooling, not jumping to RL

The focused transformer hardening sweep is now also in place:

- sweep artifact root: `artifacts/transformer_sweep_v2`
- tested grid:
  `model_dim = 4`
  `training_lookback_days = 252`
  `sequence_lookback_window in {10, 20, 40}`
  `target_clip_quantile in {raw, 0.90, 0.95}`
- best latest 1-year combo: `transformer_d4_lb252_seq40_clip_q95`
- best repeated-window average combo: `transformer_d4_lb252_seq20_clip_q95`
- `transformer_d4_lb252_seq40_clip_q95`, latest 1-year OOS ending capital: `$129,739` vs baseline `$118,294`
- `transformer_d4_lb252_seq20_clip_q95`, repeated 4-window average ending capital: `$121,320` vs baseline `$115,607`

Interpretation:

- clipping the target objective improves both the latest-window result and repeated-window average versus the raw-return objective
- `sequence_lookback_window = 10` is too short under the current data regime
- `sequence_lookback_window = 40` is strongest on the latest 1-year slice, while `20` is stronger on repeated-window average
- the next gain is more likely to come from resolving the `seq20` versus `seq40` trade-off and promoting the clipped-objective transformer into the standard report stack than from simply making the model wider

The lane-to-lane comparison artifacts are:

- `artifacts/universe_lane_comparison/last_year_lane_comparison.csv`
- `artifacts/universe_lane_comparison/stability_lane_comparison.csv`

## Risk And Paper Trading Layer

The configs now include a `risk` block:

- `capital_base`: notional portfolio size used for target sizing.
- `cash_buffer`: fraction of capital intentionally left in cash.
- `max_position_weight`: cap applied before share sizing.
- `min_trade_notional`: ignore tiny rebalance trades.
- `allow_fractional_shares`: use fractional sizing for broker models that support it.

The research backtest now applies `cash_buffer` and `max_position_weight` to target weights as well, so the backtest is closer to the paper portfolio sizing path. `min_trade_notional` and share-rounding still matter only in the order preview layer.

The configs also include:

- `scoring`: provider defaults for heuristic, DeepSeek, or a local OpenAI-compatible server.
- `workflow`: SEC fetch defaults, paper positions path, run output root, and whether to advance the paper state automatically.

## Real LLM Configs

- [with_llm_deepseek.yaml](C:/Users/sym89/Desktop/Investment/us_stocks/config/with_llm_deepseek.yaml) uses `https://api.deepseek.com` and reads [\.env.us.deepseek.example](C:/Users/sym89/Desktop/Investment/us_stocks/.env.us.deepseek.example).
- [with_llm_qwen_local.yaml](C:/Users/sym89/Desktop/Investment/us_stocks/config/with_llm_qwen_local.yaml) targets a local OpenAI-compatible server at `http://127.0.0.1:8000/v1` and reads [\.env.us.qwen_local.example](C:/Users/sym89/Desktop/Investment/us_stocks/.env.us.qwen_local.example).

## Windows Automation

Run once manually with logging:

```powershell
powershell -ExecutionPolicy Bypass -File .\us_stocks\scripts\run_us_daily.ps1 -ConfigPath .\us_stocks\config\with_llm_deepseek.yaml -RunLabel manual
```

Register a daily scheduled task:

```powershell
powershell -ExecutionPolicy Bypass -File .\us_stocks\scripts\register_us_daily_task.ps1 -TaskName USStocksDailyWorkflow -ConfigPath C:\Users\sym89\Desktop\Investment\us_stocks\config\with_llm_deepseek.yaml -StartTime 22:30
```

The helper scripts are [run_us_daily.ps1](C:/Users/sym89/Desktop/Investment/us_stocks/scripts/run_us_daily.ps1) and [register_us_daily_task.ps1](C:/Users/sym89/Desktop/Investment/us_stocks/scripts/register_us_daily_task.ps1).

Use [run_us_report_stack.ps1](C:/Users/sym89/Desktop/Investment/us_stocks/scripts/run_us_report_stack.ps1) when you want the canonical clipped-transformer report rerun rather than the daily SEC workflow.

## Current Research Direction

The recommended order remains:

1. Keep the dynamic 60-name lane and repeated-window stability report as the default reality check.
2. Keep true point-in-time constituent data as backlog, but treat the current dynamic lane as the standing control until then.
3. Continue hardening the transformer lane next by promoting the clipped-objective setup into the standard report stack and resolving the `seq20` versus `seq40` trade-off.
4. Treat LLM signals as auxiliary inputs, not as the center of the research program.
5. Leave reinforcement learning until the price-model lane is clearly credible.
