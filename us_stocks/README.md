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

## Layout

- `config/base.yaml`: price-only baseline.
- `config/soft_price.yaml`: price-only strategy with a soft trend penalty instead of a hard trend filter.
- `config/with_llm_short.yaml`: short-term LLM horizon.
- `config/with_llm_swing.yaml`: swing-horizon LLM config.
- `config/with_llm_long.yaml`: long-term LLM config.
- `runs/`: dated workflow runs with SEC inputs, merged signals, research outputs, and paper orders.
- `documents/sec_filings_recent.csv`: raw SEC filings metadata and raw text.
- `documents/sec_sections.csv`: extracted high-signal sections used for scoring.
- `signals/llm_scores.generated.csv`: aggregated multi-horizon LLM scores.
- `artifacts/`: backtest outputs and strategy comparisons.

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

Generate a portfolio snapshot and order preview from an optional positions file:

```powershell
python -m us_invest_ai.main --config .\us_stocks\config\with_llm_swing.yaml --current-positions-csv .\us_stocks\paper\current_positions.csv
```

Run the full daily research-to-paper workflow:

```powershell
$env:SEC_USER_AGENT="InvestmentResearch your_email@example.com"
python -m us_invest_ai.daily_workflow --config .\us_stocks\config\with_llm_swing.yaml --provider heuristic --run-label daily_check
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
- `ranking_history.csv`: rebalance snapshots with attached LLM signals.
- `llm_scores_used.csv`: filtered signal file for the selected horizon.
- `router_training_frame.csv`: future meta-router training frame with `next_5d`, `next_20d`, and `next_60d` realized returns.
- `target_portfolio.csv`: latest risk-limited holdings with target shares and notionals.
- `recommended_orders.csv`: optional order preview if `--current-positions-csv` is provided.
- `next_positions_preview.csv`: optional post-trade holdings preview.

Running `daily_workflow.py` also saves:

- `documents/sec_filings_recent.csv`: raw SEC filings fetched in that run.
- `documents/sec_sections.csv`: extracted scoring sections for that run.
- `signals/llm_scores.merged.csv`: existing signal store plus the new run's scores.
- `paper/target_portfolio.csv`: current paper target holdings.
- `paper/recommended_orders.csv`: delta versus the supplied current positions file.

## Comparison Metrics

`compare_configs` now reports:

- `signal_coverage`: share of rebalance rows with a non-zero attached LLM signal.
- `avg_llm_abs_score`: average absolute LLM signal size during the evaluation window.
- `changed_rebalance_count`: number of rebalance dates that differ from the baseline.

Use `soft_price.yaml` as the control when you want to separate the impact of `soft trend filtering` from the impact of `LLM signals`.

## Risk And Paper Trading Layer

The configs now include a `risk` block:

- `capital_base`: notional portfolio size used for target sizing.
- `cash_buffer`: fraction of capital intentionally left in cash.
- `max_position_weight`: cap applied before share sizing.
- `min_trade_notional`: ignore tiny rebalance trades.
- `allow_fractional_shares`: use fractional sizing for broker models that support it.

The research backtest remains unchanged for comparability. The `risk` block is applied when generating paper portfolio outputs.

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

## Current Research Direction

The recommended order remains:

1. Improve section extraction and signal quality.
2. Validate multi-horizon behavior against the price-only baseline.
3. Build a meta-router on top of `router_training_frame.csv`.
4. Only after that, consider supervised deep learning or reinforcement learning for horizon selection or sizing.
