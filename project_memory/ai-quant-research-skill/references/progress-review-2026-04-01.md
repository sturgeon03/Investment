# Progress Review

Review date: 2026-04-01

This file estimates progress against the current roadmap rather than against a hypothetical full production trading stack.

## Research Progress

- evaluation framework and reporting rigor: about `85%`
- universe realism and point-in-time discipline: about `70%`
- price-model and sequence-model alpha lane: about `75%`
- shared-core reuse plus KR adapter follow-through: about `60%`
- LLM signal repositioning as auxiliary features: about `30%`
- RL and live-promotion stage: about `5%`
- total research-program progress: about `62%`

Interpretation:

- the repo is strong on repeatable evaluation, manifests, repeated OOS windows, and the promoted transformer validation path
- the main remaining gap is not "more plumbing"; it is still data realism, structured LLM feature design, and stronger proof that learned alpha survives beyond the current free-approx universe history
- RL is intentionally near-zero because the roadmap explicitly keeps it last

## Automation Progress

- overnight canonical research rerun: about `90%`
- cache-first overnight fallback lane: about `85%`
- paper daily workflow runner: about `80%`
- broker-backed paper safety layer: about `75%`
- bounded night-batch orchestrator: about `70%`
- scheduler and Codex-native automation operationalization: about `40%`
- total automation-stack progress: about `73%`

Interpretation:

- the repo-native PowerShell automation is mostly real and not just placeholder scripts
- the lowest automation score is the operationalization layer because the Windows task registration is scriptable but not state-tracked in-repo, and there is still no Codex-native automation configured under `$CODEX_HOME/automations`

## Automation Review Findings

- latest successful repo-native overnight canonical rerun: `2026-03-30 02:11 +09:00`
- latest market date in that successful overnight rerun: `2026-03-19`
- latest successful night batch completion: `2026-03-30 10:03 +09:00`
- observed night-batch task mix on `2026-03-30`: `5/6` tasks succeeded; the one recorded failure happened after report artifacts were generated

The failed research task was not a model or report regression. The artifacts for `20260330_052820` were written, but `run_summary.json` was missing because the overnight wrapper's final JSON persistence could choke on PowerShell runtime objects and skip the summary write.

## Fix Applied On 2026-04-01

- hardened `us_stocks/scripts/run_overnight_quant.ps1` so final status writing first normalizes non-JSON-friendly PowerShell runtime objects
- guaranteed that the single-writer lock is still removed even if summary persistence fails
- preserved the existing "fail the job if summary persistence breaks" behavior so automation health is still honest

## Remaining Operational Gaps

- fresh Yahoo-backed refreshes are still less reliable than the cache-backed overnight path
- there is still no lightweight smoke coverage for the PowerShell automation wrappers themselves
- Codex-native automation is not configured yet, so the automation layer currently depends on repo-local scripts plus optional Windows Task Scheduler registration
