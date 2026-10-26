# domino-rl — Project Brief

> Full project context in one file. Hand this to ANOTHER AI (GPT, etc.) for planning
> and ideation, then bring the refined specs back. Keep this file accurate — it is the handoff doc.
> For the current timeline see `STATUS.md`. For how to work in this repo see `AGENTS.md`.

## One-liner
Domino RL teaches an AI to play Cuban double-9 domino (55 tiles, parejas) — Roger's deep-RL learning project with verifiable ML evidence.

## Problem & audience
Roger's ML baseline is fine-tuning; he wants a deep dive into ML starting from RL foundations (MDPs, Q-learning, policy gradients) before agent post-training (RLHF, DPO, RLVR) — learned by building, not by reading.

## Product (what it is / is not)
A full RL stack for double-9 domino: engine, Gymnasium env, baselines, tournament eval, MAPPO training. Dual purpose: strong play AND knowledge discovery. It is NOT a heuristics engine — human heuristics are banned by design.

## Key decisions (locked)
- STRICT bitter lesson: zero human heuristics, sparse +1/-1/0 reward only.
- Framed as Dec-POMDP (blind 2v2 cooperation).
- Obs 143-dim = hand 55 + board ends 20 + played-tiles one-hot 55 (full public history) + remaining 4 + passes 1 + seat 4 + teammate 4. Pip-counts REMOVED per Roger's 'counting must emerge from complete history' thesis. Set invariant: 495 total pips (test asserts).
- Roger requires verifiable ML evidence: raw logs, checkpoints, fixed seeds, reproduction commands, source paths.

## Stack
Python (uv), PyTorch, Gymnasium. Analyzer: `src/domino_rl/analyze.py`.

## Business model
Learning project — no business model. Feeds Roger's #1-in-AI goal as public RL depth.

## Open questions
- 2v2 warm-start final eval results.
- Which heuristics emerge in Fase 7 (`docs/hallazgos.md`).
