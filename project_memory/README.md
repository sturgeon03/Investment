# Project Memory

This folder is the persistent operating context for the Investment repository.

Use it as the single place to maintain:

- project direction
- current verified state
- next recommended steps
- Codex working rules for this repo

## Structure

- `ai-quant-research-skill/`: repo-local Codex skill and its references
- `sync_skill_to_codex.ps1`: sync the repo-local skill into `C:\Users\sym89\.codex\skills`

## Maintenance Rule

Update the references in `ai-quant-research-skill/references/` after any substantial change to:

- research direction
- model family or validation policy
- workflow architecture
- reported results or benchmark conclusions

Keep these files concise, factual, and decision-oriented.

## Sync Rule

The repo-local copy is the source of truth.

After updating the skill or its references, re-install it into the Codex skills directory with:

```powershell
powershell -ExecutionPolicy Bypass -File .\project_memory\sync_skill_to_codex.ps1
```
