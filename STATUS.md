# STATUS — domino-rl

> Single source of truth for where this project stands. Last updated: 2026-10-06.
> Read this before starting work. Update it in the same PR when reality changes.

## Done
- 2026-09-26 — Engine + Gymnasium env (obs 143-dim, 111 masked actions, shared-policy self-play) + baselines + tournament eval.
- 2026-09-26 — MAPPO 1v1 validated: 60.2% vs random / 48.2% vs greedy. 23/23 tests green.
- 2026-09-26 — 2v2 teams curriculum chosen by Roger (one shared net, unbiased team reward); warm-start run launched.

## In progress / blocked
- 2v2 teams warm-start run (launched 2026-09-26) — final eval pending.

## Next
- Per PLAN.md Fases 0–9; Fase 7 writes discovered heuristics to `docs/hallazgos.md`.
- Deep CFR deferred (vanilla CFR is 2-player zero-sum); curriculum goes 1v1 → 4p individual → 2v2 teams.
