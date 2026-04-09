# Progress Review

Review date: 2026-04-09

This file updates progress after a mainline integration audit and the promotion of the detached `signal_hardening` work into `main`.

## Overall Progress

- total project progress, weighted across research, automation, and mainline integration: about `69%`
- total research-program progress, weighted toward roadmap priorities rather than operations alone: about `63%`
- total automation-stack progress: about `76%`
- committed-work mainline integration progress: about `98%`

Interpretation:

- the project is now much better integrated and operationally repeatable than it was a few weeks ago
- the real bottleneck is still stable learned alpha under stricter realism, not whether the repo can run overnight scripts
- the high mainline-integration score does not mean the research is nearly done; it means the existing work is now mostly in one place

## Mainline Integration Progress

- `main` and `origin/main` baseline sync before this update: `100%`
- detached committed-work audit coverage: `100%`
- unique detached commit promotion into `main`: `100%`
- superseded detached variants disposition: `100%`
- stale detached worktree hygiene: about `35%`

Interpretation:

- `codex/provide-ai` was already an ancestor of `main`, so it was not a missing merge target
- the only committed gap outside `main` was the newer signal-hardening branch tip, and it is now promoted into `main`
- many detached worktrees still exist under Codex bookkeeping, but they point to old snapshots rather than additional unique commits

## Research Progress

- evaluation framework and reporting rigor: about `88%`
- manifests, provenance, and artifact promotion: about `95%`
- repeated out-of-sample window evaluation: about `90%`
- signal-hardening degradation diagnostics: about `80%`
- execution-realism in evaluation: about `60%`

- universe realism and point-in-time discipline: about `72%`
- phased snapshots and eligibility filters: about `85%`
- dynamic 60-name realism control lane: about `85%`
- true constituent-history realism: about `25%`

- price-model and sequence-model alpha lane: about `77%`
- tabular baseline/control coverage: about `85%`
- sequence-model control coverage: about `82%`
- stable learned-alpha proof under strict realism: about `45%`

- shared-core reuse plus KR adapter follow-through: about `63%`
- shared-core split: about `72%`
- KR raw-data and adapter stack: about `78%`
- KR learned-model validation lane: about `40%`

- LLM signal repositioning as auxiliary features: about `32%`
- LLM ingestion and scoring plumbing: about `70%`
- structured LLM subfactor design: about `10%`
- price-model feature integration rigor for LLM signals: about `15%`

- RL and live-promotion stage: about `5%`
- RL sizing/allocation research: about `0%`
- live-trading promotion: about `10%`

Interpretation:

- the strongest increase since the last review is in reporting rigor because the repo now has a dedicated degradation layer on top of the promoted transformer stack
- price-model progress is real, but the gap between "best backtest evidence" and "credible durable alpha" is still large
- KR is no longer a placeholder, but it is still in the first serious research-lane stage rather than a mature market stack

## Market-Lane Progress

- US market research lane: about `81%`
- KR market research lane: about `58%`
- cross-market shared-core layer: about `63%`
- paper workflow and broker safety lane: about `74%`

Interpretation:

- US is the only lane with enough evidence to support sequence-model promotion decisions
- KR now has enough structure to be real work, but not enough validation depth to claim parity with the US lane

## Automation Progress

- overnight canonical research rerun: about `92%`
- cache-first overnight fallback lane: about `87%`
- report-stack promotion plus signal-hardening outputs: about `90%`
- paper daily workflow runner: about `82%`
- broker-backed paper safety layer: about `78%`
- bounded night-batch orchestrator: about `72%`
- Windows scheduler registration path: about `55%`
- Codex-native automation operationalization: about `0%`

Interpretation:

- the repo-native automation stack is now fairly complete for overnight reruns and paper-ops guardrails
- the weakest operational area is still stateful scheduler management because the automation is scriptable but not yet Codex-native or centrally tracked

## Immediate Next Gaps

- move from free-approx universe history toward better constituent-history realism
- define explicit promotion thresholds for worst-window degradation and repeated-window damage
- deepen the KR learned-model lane past the first walk-forward ridge comparison
- decompose scalar `llm_score` into structured auxiliary features
- keep RL and live promotion parked until the price-model evidence is stronger
