# Domino RL

Teach an AI to play **Cuban double-9 domino** (55 tiles, 2v2 partnerships) with **reinforcement learning** — strict bitter lesson, sparse rewards only, CPU-only.

## Principles (non-negotiable)

1. **Strict bitter lesson** — zero human heuristics. Sparse +1/−1/0 reward only; no hand-coded strategy.
2. **Dual purpose** — strong play AND knowledge discovery: learned heuristics are written up in `docs/hallazgos.md` (Fase 7).
3. **Dec-POMDP framing** — blind 2v2 cooperation. Counting must emerge from complete public history, never be hardcoded (pip-counts were deliberately removed from the observation).

## What it is

- **Engine** — full double-9 rules: 4 players × 10 tiles (15 in the pool), highest double leads (forced), partnerships (0&2 vs 1&3) or free-for-all, domino or tranca wins, pip-count scoring.
- **Gymnasium-style env** — 143-dim observation (hand 55 + board ends 20 + **full public tile history 55** + remaining 4 + passes 1 + seat 4 + teammate 4), 111 masked actions, shared-policy self-play.
- **Baselines** — random and greedy (heuristic) reference agents.
- **Training** — PPO/IPPO with self-play on CPU. The net is tiny (~35k params) and the sim is pure Python — the free Colab tier is plenty, no GPU needed.
- **Evaluation** — tournaments with win-rate, points, domino-vs-tranca stats (`evaluate.py`); style analysis + game transcripts (`analyze.py`).
- **Demo table** — humans (phones) vs AI: `PYTHONPATH=src python demo/server.py`, then open `http://<ip>:8000` on each phone.

## Status

- **1v1 MAPPO validated (2026-09-26):** 60.2% vs random / 48.2% vs greedy. 23/23 tests green.
- **2v2 teams curriculum in progress:** one shared net, unbiased team reward (owner's choice); warm-start run launched 2026-09-26 — final eval pending.
- **Next:** per `PLAN.md` Fases 0–9 (curriculum 1v1 → 4p individual → 2v2 teams; vanilla CFR deferred — it's 2-player zero-sum).

## Quickstart

```bash
uv sync --group train && source .venv/bin/activate

# 1. Sanity check: greedy vs random tournament
PYTHONPATH=src python -m domino_rl.evaluate --rounds 200 --a greedy --b random

# 2. Train 2v2 partnerships with IPPO (CPU)
#    (--iters = TOTAL iterations; LR and entropy decay linearly)
PYTHONPATH=src python -m domino_rl.ippo --iters 1000 --manos-per-iter 256 \
  --mode teams --n-players 4 --match-target 100 \
  --lr 1e-4 --lr-end 0.0 --ent-coef 0.1 --ent-end 0.01 \
  --epochs 2 --seed 42 \
  --out checkpoints/ippo_2v2 --ckpt-every 100 --eval-every 100 --eval-manos 200

# 3. Evaluate a checkpoint vs greedy (500 hands)
PYTHONPATH=src python -m domino_rl.evaluate --rounds 500 \
  --a ippo:checkpoints/ippo_2v2/ckpt_500.pt --b greedy \
  --mode teams --n-players 4

# 4. See HOW it plays, not just how much it wins (style stats + transcripts)
PYTHONPATH=src python -m domino_rl.analyze \
  --policy checkpoints/ippo_2v2/ckpt_500.pt \
  --manos 200 --opponent greedy --transcripts 3
```

Everything needed to reproduce is in the repo: code, `pyproject.toml` + `uv.lock`, fixed seeds, checkpoints.

## Structure

```
src/domino_rl/
├── game.py        # Double-9 engine: rules, deal, scoring
├── env.py         # Gymnasium-style env: obs[143], 111 masked actions
├── baselines.py   # Random + greedy reference agents
├── evaluate.py    # Tournaments: win-rate, points, domino vs tranca
├── train.py       # PPO training with self-play (CPU)
├── ippo.py        # IPPO for 2v2 team play
└── analyze.py     # Style stats + game transcripts
tests/             # Engine tests (TDD — written before the engine)
demo/              # Digital table: humans vs AI
PLAN.md            # Full step-by-step plan (Fases 0–9)
```

## License

No license file yet.
