"""Evaluation harness: tournaments between policies.

A policy is any callable `(obs, mask) -> action`.

Usage:
    python -m domino_rl.evaluate --rounds 200 --a greedy --b random
    python -m domino_rl.evaluate --rounds 200 --a ppo:checkpoints/best.zip --b greedy
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from .baselines import AGENTS
from .env import DominoEnv


def load_policy(spec: str):
    """'greedy' / 'random', 'ppo:<sb3 .zip>' or 'mappo:<.pt>'."""
    if spec in AGENTS:
        agent = AGENTS[spec]()
        return agent.act
    if spec.startswith("ppo:"):
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(spec[len("ppo:"):])
        def ppo_act(obs, mask):
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)
        return ppo_act
    if spec.startswith("mappo:"):
        import torch
        from .mappo import load_policy as load_mappo

        policy = load_mappo(spec[len("mappo:"):])
        def mappo_act(obs, mask):
            import numpy as np
            ot = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            mt = torch.as_tensor(np.asarray(mask, dtype=bool)).unsqueeze(0)
            a, _, _ = policy.act(ot, mt, deterministic=True)
            return int(a.item())
        return mappo_act
    raise ValueError(f"unknown policy spec {spec!r}")


def play_round(policy_a, policy_b, seats_a=(0, 2), mode="teams",
               n_players=4, seed=None):
    env = DominoEnv(mode=mode, reward_mode="sparse", n_players=n_players,
                    seed=seed)
    obs, _ = env.reset()
    done = False
    while not done:
        seat = env.round.turn
        policy = policy_a if seat in seats_a else policy_b
        action = policy(obs, env.action_masks())
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def tournament(policy_a, policy_b, n_rounds=200, mode="teams",
               n_players=4, seed=0):
    rng = np.random.default_rng(seed)
    wins = Counter()   # "A" / "B" / "draw"
    points = Counter()
    kinds = Counter()  # domino vs tranca
    for i in range(n_rounds):
        # alternate which seats each policy controls (fairness)
        if n_players == 2:
            seats_a = (0,) if i % 2 == 0 else (1,)
        else:
            seats_a = (0, 2) if i % 2 == 0 else (1, 3)
        info = play_round(policy_a, policy_b, seats_a,
                          mode=mode, n_players=n_players,
                          seed=int(rng.integers(0, 2**31 - 1)))
        w = info["winner_team"]
        if w is None:
            wins["draw"] += 1
        else:
            # winner_team is a team index (teams) or seat (individual);
            # seats_a tells which side policy A controlled
            side = "A" if w in seats_a else "B"
            wins[side] += 1
            points[side] += info["points"]
        kinds["domino" if info["winner_seat"] is not None else "tranca"] += 1
    return {"wins": dict(wins), "points": dict(points),
            "endings": dict(kinds), "rounds": n_rounds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="greedy")
    ap.add_argument("--b", default="random")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--mode", default="teams", choices=["teams", "individual"])
    ap.add_argument("--n-players", type=int, default=4, choices=[2, 4])
    args = ap.parse_args()

    pa, pb = load_policy(args.a), load_policy(args.b)
    res = tournament(pa, pb, n_rounds=args.rounds, mode=args.mode,
                     n_players=args.n_players)
    w = res["wins"]
    a, b = w.get("A", 0), w.get("B", 0)
    print(f"A ({args.a}) vs B ({args.b}) — {res['rounds']} rounds [{args.mode}]")
    print(f"  wins:  A={a} ({a/res['rounds']:.1%})  "
          f"B={b} ({b/res['rounds']:.1%})  draw={w.get('draw', 0)}")
    print(f"  points: A={res['points'].get('A',0)}  B={res['points'].get('B',0)}")
    print(f"  endings: {res['endings']}")


if __name__ == "__main__":
    main()
