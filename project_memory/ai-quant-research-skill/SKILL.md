---
name: ai-quant-research-skill
description: Continue the Investment repository as a rigorous AI quant research program rather than a generic trading bot project. Use when working on this repo to decide next steps, implement new research modules, evaluate model progress, or update project memory. Prioritize research rigor, leakage control, out-of-sample validation, and the roadmap of price-based deep learning first, LLM signals second, and reinforcement learning last.
---

# AI Quant Research Skill

Read `references/current-state.md` first to understand what already exists and what the latest verified results mean.

Read `references/principles.md` before planning or implementing any strategy or model change.

Read `references/roadmap.md` when deciding what to build next.

Read `references/review-followups.md` when external feedback or old notes need to be reconciled with the local workspace.

## Workflow

1. Treat the existing repo as a research platform, not as a production trading system.
2. Preserve the current pipeline unless a change clearly improves research rigor or enables the roadmap.
3. Before adding a new model, confirm whether the change belongs to:
   - data and validation hardening
   - price-based AI alpha modeling
   - text/LLM signal integration
   - RL or execution research
4. Prefer the roadmap order in `references/roadmap.md`. Do not skip ahead to RL-first work unless the user explicitly redirects the project.
5. When reporting performance, separate:
   - heuristic or rules-based results
   - true learned-model results
   - real trading vs paper trading vs backtest
6. After meaningful code or result changes, update the references so future work can resume without reconstructing context from scratch.

## Guardrails

- Do not present heuristic SEC scoring as "real AI alpha".
- Do not treat a backtest as real profits.
- Do not claim research-grade rigor unless point-in-time data, leakage controls, and strict out-of-sample validation are in place.
- Bias toward fewer, stronger experiments over many weak ones.

## References

- `references/current-state.md`
- `references/principles.md`
- `references/roadmap.md`
- `references/review-followups.md`
