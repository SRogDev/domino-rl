"""Shared-parameter IPPO self-play for Cuban double-9 domino (option B).

Naming note (professional precision): this is IPPO — Independent PPO with
parameter sharing — NOT MAPPO. Each seat learns its own value function from
the shared actor-critic; there is no centralized critic with global state
(which is what defines MAPPO, Yu et al. 2021). The shared trunk + seat
one-hots in the observation let one network play every chair.

Why this exists
--------------
The first attempt (MaskablePPO over ONE trajectory that walks through every
seat in turn order) never learned: the terminal +1/-1 was attached only to the
LAST transition of the hand, so every earlier move — by winners and losers
alike — was reinforced with the same sign. A policy cannot tell a good move
from a bad one that way (200k steps: 49.8% vs random, and the 200k checkpoint
lost to the 100k one).

This module implements the correct formulation:
- ONE shared actor-critic (parameter sharing across seats), but
- PER-SEAT trajectories: each seat's own turns inside a hand form one episode,
- PER-SEAT team reward at hand end, from that seat's own perspective.

Reward = the real game score, no heuristics (bitter lesson intact):
- every hand ("mano"):  r = tantos_won_by_my_team / match_target
- match bonus: a match ends when a team reaches `match_target` tantos
  (100, the Cuban way). On that final hand every seat gets an extra
  +1.0 (winning team) / -1.0 (losing team).

Professional instrumentation: linear LR annealing, entropy-coef annealing
(0.1 -> 0.01: high early exploration, low late noise), approx-KL logging,
and full checkpoints (policy + optimizer + RNG + iter) so --resume is exact.

Everything runs on CPU.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from .env import N_ACTIONS, OBS_DIM, DominoEnv

NEG_INF = -1e9


# ------------------------------------------------------------------ model

class ActorCritic(nn.Module):
    """Shared trunk, masked categorical actor, scalar critic."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.actor = nn.Linear(d, n_actions)
        self.critic = nn.Linear(d, 1)
        # small actor init -> near-uniform policy at start -> real exploration
        nn.init.normal_(self.actor.weight, std=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        h = self.trunk(obs)
        logits = self.actor(h).masked_fill(~mask.bool(), NEG_INF)
        value = self.critic(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor,
            deterministic: bool = False):
        logits, value = self.forward(obs, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


def save_policy(policy: ActorCritic, path: str):
    """Minimal checkpoint: policy weights only (for eval/analysis)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(),
                "obs_dim": OBS_DIM, "n_actions": N_ACTIONS}, path)


def save_checkpoint(policy: ActorCritic, opt: torch.optim.Optimizer,
                    rng: np.random.Generator, it: int, path: str):
    """Full checkpoint: policy + optimizer + RNG states + iter.

    Makes --resume exact: Adam moments, mano sequences and LR/entropy
    schedules all continue where they left off.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(),
                "obs_dim": OBS_DIM, "n_actions": N_ACTIONS,
                "opt": opt.state_dict(),
                "np_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "iter": it}, path)


def load_policy(path: str, device: str = "cpu") -> ActorCritic:
    """Load policy weights from a policy-only OR a full checkpoint."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = ActorCritic(obs_dim=ckpt["obs_dim"],
                         n_actions=ckpt["n_actions"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (policy, opt_state_dict, np_rng_state, torch_rng_state, it)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = ActorCritic(obs_dim=ckpt["obs_dim"],
                         n_actions=ckpt["n_actions"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    return (policy, ckpt.get("opt"), ckpt.get("np_rng"),
            ckpt.get("torch_rng"), ckpt.get("iter", 0))


# ------------------------------------------------------------------ rollout

def _team_of(env: DominoEnv, seat: int) -> int:
    return env.round.team_of(seat)


def play_manos(policy: ActorCritic, n_manos: int, mode: str, n_players: int,
               match_target: int, rng: np.random.Generator,
               device: str = "cpu"):
    """Play `n_manos` hands. Returns (trajectories, stats).

    One trajectory per (hand, seat): that seat's own turns, with the reward
    attached to its last turn from its own team's perspective.
    """
    trajs: list[dict] = []
    tantos = np.zeros(n_players, dtype=np.int64)  # indexed by team id
    n_won = {0: 0, 1: 0}          # hands won per team id (teams mode)
    seat_wins = np.zeros(n_players, dtype=np.int64)
    total_tantos = 0
    policy.eval()

    for _ in range(n_manos):
        env = DominoEnv(mode=mode, n_players=n_players,
                        seed=int(rng.integers(0, 2 ** 31 - 1)))
        env.reset()
        per_seat: dict[int, list] = {s: [] for s in range(n_players)}
        done = False
        while not done:
            seat = env.round.turn
            obs = env.observe(seat)
            mask = env.action_masks()
            ot = torch.as_tensor(obs, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            mt = torch.as_tensor(mask, dtype=torch.bool,
                                 device=device).unsqueeze(0)
            a, logp, v = policy.act(ot, mt)
            _, _, term, trunc, info = env.step(int(a.item()))
            per_seat[seat].append((obs, int(a.item()),
                                   float(logp.item()), float(v.item()), mask))
            done = bool(term or trunc)

        wt = info["winner_team"]          # team id or None (draw)
        pts = int(info["points"])
        if wt is not None:
            tantos[wt] += pts
            total_tantos += pts
            n_won[wt] = n_won.get(wt, 0) + 1
            for s in range(n_players):
                if _team_of(env, s) == wt:
                    seat_wins[s] += 1

        match_over = bool((tantos >= match_target).any())
        match_winner = int(np.argmax(tantos)) if match_over else None

        for s in range(n_players):
            team = _team_of(env, s)
            r = (pts / match_target) if (wt is not None and wt == team) else 0.0
            if match_over:
                r += 1.0 if team == match_winner else -1.0
            steps = per_seat[s]
            if not steps:
                continue
            rew = np.zeros(len(steps), dtype=np.float32)
            rew[-1] = r
            trajs.append({
                "seat": s, "team": team,
                "obs": np.stack([x[0] for x in steps]).astype(np.float32),
                "act": np.array([x[1] for x in steps], dtype=np.int64),
                "logp": np.array([x[2] for x in steps], dtype=np.float32),
                "val": np.array([x[3] for x in steps], dtype=np.float32),
                "mask": np.stack([x[4] for x in steps]).astype(bool),
                "rew": rew,
            })
        if match_over:
            tantos = np.zeros(n_players, dtype=np.int64)

    stats = {"manos": n_manos, "team_hands_won": dict(n_won),
             "seat_hands_won": seat_wins, "avg_tantos_per_mano":
             total_tantos / max(1, n_manos)}
    return trajs, stats


# ------------------------------------------------------------------ PPO math

def compute_gae(rew: np.ndarray, val: np.ndarray,
                gamma: float = 0.99, lam: float = 0.95):
    """GAE over one seat-episode (it always ends at the hand's end)."""
    t_len = len(rew)
    adv = np.zeros(t_len, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(t_len)):
        last = t == t_len - 1
        next_v = 0.0 if last else val[t + 1]
        not_done = 0.0 if last else 1.0
        delta = rew[t] + gamma * next_v * not_done - val[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    return adv, (adv + val).astype(np.float32)


def ppo_update(policy: ActorCritic, opt: torch.optim.Optimizer,
               batch: dict[str, torch.Tensor],
               epochs: int = 2, minibatch: int = 512,
               clip: float = 0.2, vf_coef: float = 0.5,
               ent_coef: float = 0.1, max_grad_norm: float = 0.5,
               device: str = "cpu"):
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ds = TensorDataset(batch["obs"], batch["act"], batch["logp"],
                       batch["mask"], adv, batch["ret"], batch["val"])
    loader = DataLoader(ds, batch_size=minibatch, shuffle=True)
    tot = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "n": 0}
    policy.train()
    for _ in range(epochs):
        for obs_b, act_b, logp_b, mask_b, adv_b, ret_b, val_b in loader:
            obs_b, mask_b = obs_b.to(device), mask_b.to(device)
            act_b, logp_b = act_b.to(device), logp_b.to(device)
            adv_b, ret_b, val_b = (adv_b.to(device), ret_b.to(device),
                                   val_b.to(device))
            logits, v = policy(obs_b, mask_b)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act_b)
            log_ratio = new_logp - logp_b
            ratio = torch.exp(log_ratio)
            # approx KL (Schulman k1 estimator): watch for policy collapse
            approx_kl = ((ratio - 1) - log_ratio).mean()
            pg = -torch.min(ratio * adv_b,
                            torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b
                            ).mean()
            v_clipped = val_b + torch.clamp(v - val_b, -clip, clip)
            vf = torch.max((v - ret_b) ** 2,
                           (v_clipped - ret_b) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + vf_coef * vf - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()
            tot["pg"] += float(pg.detach()); tot["vf"] += float(vf.detach())
            tot["ent"] += float(ent.detach())
            tot["kl"] += float(approx_kl.detach()); tot["n"] += 1
    return {k: tot[k] / max(1, tot["n"]) for k in ("pg", "vf", "ent", "kl")}


def _stack(trajs: list[dict], device: str):
    advs, rets = [], []
    for t in trajs:
        a, r = compute_gae(t["rew"], t["val"])
        advs.append(a)
        rets.append(r)
    to = lambda x, dt: torch.as_tensor(x, dtype=dt)  # noqa: E731
    return {
        "obs": to(np.concatenate([t["obs"] for t in trajs]), torch.float32),
        "act": to(np.concatenate([t["act"] for t in trajs]), torch.int64),
        "logp": to(np.concatenate([t["logp"] for t in trajs]), torch.float32),
        "mask": to(np.concatenate([t["mask"] for t in trajs]), torch.bool),
        "val": to(np.concatenate([t["val"] for t in trajs]), torch.float32),
        "adv": to(np.concatenate(advs), torch.float32),
        "ret": to(np.concatenate(rets), torch.float32),
    }


# ------------------------------------------------------------------ schedules

def lerp(a: float, b: float, frac: float) -> float:
    """Linear interpolation used by the LR / entropy annealing schedules."""
    return a + (b - a) * frac


# ------------------------------------------------------------------ eval hook

def eval_vs(policy: ActorCritic, baseline: str, manos: int, mode: str,
            n_players: int, seed: int, device: str = "cpu"):
    """Win-rate of `policy` (deterministic) vs a baseline. The honest metric."""
    from .baselines import AGENTS
    base = AGENTS[baseline]()
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    policy.eval()
    for i in range(manos):
        seats_mine = ((0,) if n_players == 2 else (0, 2)) if i % 2 == 0 else \
                     ((1,) if n_players == 2 else (1, 3))
        env = DominoEnv(mode=mode, n_players=n_players,
                        seed=int(rng.integers(0, 2 ** 31 - 1)))
        env.reset()
        done = False
        while not done:
            seat = env.round.turn
            if seat in seats_mine:
                ot = torch.as_tensor(env.observe(seat), dtype=torch.float32,
                                     device=device).unsqueeze(0)
                mt = torch.as_tensor(env.action_masks(), dtype=torch.bool,
                                     device=device).unsqueeze(0)
                a, _, _ = policy.act(ot, mt, deterministic=True)
                action = int(a.item())
            else:
                action = int(base.act(env.observe(seat), env.action_masks()))
            _, _, term, trunc, info = env.step(action)
            done = bool(term or trunc)
        wt = info["winner_team"]
        if wt is None:
            draws += 1
        elif wt in {env.round.team_of(s) for s in seats_mine}:
            wins += 1
        else:
            losses += 1
    return wins / manos, draws / manos


# ------------------------------------------------------------------ training

def train(n_iters: int = 300, manos_per_iter: int = 64, mode: str = "teams",
          n_players: int = 4, match_target: int = 100, seed: int = 0,
          lr: float = 3e-4, lr_end: float = 0.0,
          ent_coef: float = 0.1, ent_end: float = 0.01, epochs: int = 2,
          out: str = "checkpoints/ippo",
          ckpt_every: int = 50, log_every: int = 10,
          eval_every: int = 0, eval_manos: int = 200,
          device: str = "cpu", resume: str | None = None):
    """n_iters = TOTAL iterations; with --resume, trains up to that number."""
    policy = ActorCritic().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    start_it = 1
    if resume and os.path.exists(resume):
        print("resuming from", resume)
        policy, opt_state, np_rng, torch_rng, last_it = \
            load_checkpoint(resume, device)
        restored = []
        if opt_state is not None:
            opt.load_state_dict(opt_state)
            restored.append("optimizer")
        if np_rng is not None:
            rng.bit_generator.state = np_rng
            restored.append("numpy RNG")
        if torch_rng is not None:
            torch.set_rng_state(torch_rng.cpu())
            restored.append("torch RNG")
        start_it = last_it + 1
        if restored:
            print(f"  continued at iter {start_it} "
                  f"({' + '.join(restored)} restored)")
        else:
            print(f"  warm start at iter 1 from policy weights only "
                  f"(optimizer/RNG fresh)")
    if start_it > n_iters:
        print(f"nothing to do: checkpoint already at iter {start_it - 1} "
              f">= n_iters {n_iters}")
        return resume
    os.makedirs(out, exist_ok=True)

    for it in range(start_it, n_iters + 1):
        # linear annealing schedules (CleanRL-style)
        frac = it / n_iters
        lr_now = lerp(lr, lr_end, frac)
        ent_now = lerp(ent_coef, ent_end, frac)
        opt.param_groups[0]["lr"] = lr_now
        trajs, stats = play_manos(policy, manos_per_iter, mode, n_players,
                                  match_target, rng, device)
        batch = _stack(trajs, device)
        losses = ppo_update(policy, opt, batch, ent_coef=ent_now,
                            epochs=epochs, device=device)
        mean_rew = float(np.mean([t["rew"].sum() for t in trajs]))
        if it % log_every == 0 or it == 1:
            line = (f"iter {it:4d}/{n_iters}  manos={stats['manos']}  "
                    f"mean_seat_rew={mean_rew:+.3f}  "
                    f"hands_won={stats['team_hands_won']}  "
                    f"avg_tantos={stats['avg_tantos_per_mano']:.1f}  "
                    f"pg={losses['pg']:+.4f} vf={losses['vf']:.4f} "
                    f"ent={losses['ent']:.3f} kl={losses['kl']:.4f} "
                    f"lr={lr_now:.0e} entc={ent_now:.3f}")
            if eval_every and it % eval_every == 0:
                wr, dr = eval_vs(policy, "random", eval_manos, mode,
                                 n_players, seed + it, device)
                wg, dg = eval_vs(policy, "greedy", eval_manos, mode,
                                 n_players, seed + it + 999, device)
                line += (f"  | EVAL vs random {wr:.1%} (draw {dr:.1%})"
                         f" vs greedy {wg:.1%} (draw {dg:.1%})")
            print(line, flush=True)
        if it % ckpt_every == 0:
            save_checkpoint(policy, opt, rng, it,
                            os.path.join(out, f"ckpt_{it}.pt"))
    final = os.path.join(out, "final.pt")
    save_checkpoint(policy, opt, rng, n_iters, final)
    print("saved:", final)
    return final


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Shared-parameter IPPO self-play (option B)")
    ap.add_argument("--iters", type=int, default=300,
                    help="TOTAL iterations; with --resume, trains up to this")
    ap.add_argument("--manos-per-iter", type=int, default=64)
    ap.add_argument("--mode", default="teams", choices=["teams", "individual"])
    ap.add_argument("--n-players", type=int, default=4, choices=[2, 4])
    ap.add_argument("--match-target", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="initial LR (linearly annealed to --lr-end)")
    ap.add_argument("--lr-end", type=float, default=0.0,
                    help="final LR after annealing")
    ap.add_argument("--ent-coef", type=float, default=0.1,
                    help="initial entropy coef (annealed to --ent-end)")
    ap.add_argument("--ent-end", type=float, default=0.01,
                    help="final entropy coef after annealing")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="checkpoints/ippo")
    ap.add_argument("--resume", default=None,
                    help="full checkpoint: restores policy+opt+RNG+iter")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="run eval vs random/greedy every N iters (0=off)")
    ap.add_argument("--eval-manos", type=int, default=200)
    args = ap.parse_args()
    train(n_iters=args.iters, manos_per_iter=args.manos_per_iter,
          mode=args.mode, n_players=args.n_players,
          match_target=args.match_target, seed=args.seed, lr=args.lr,
          lr_end=args.lr_end, ent_coef=args.ent_coef, ent_end=args.ent_end,
          epochs=args.epochs, out=args.out,
          resume=args.resume, ckpt_every=args.ckpt_every,
          log_every=args.log_every, eval_every=args.eval_every,
          eval_manos=args.eval_manos)


if __name__ == "__main__":
    main()
