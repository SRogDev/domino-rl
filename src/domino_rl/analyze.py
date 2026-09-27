"""Behavioral analysis: HOW does a trained policy play, not just how much it wins.

Stats + human-readable hand transcripts for qualitative observation
(cooperation signs, style, endgame). Usage:
    python -m domino_rl.analyze --policy checkpoints/mappo_2v2/final.pt \
        --manos 200 --opponent greedy --transcripts 2
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from .env import DominoEnv, PASS_ACTION
from .game import TILES, pips
from .mappo import load_policy
from .baselines import AGENTS


def tile_str(t: int) -> str:
    a, b = TILES[t]
    return f"{a}-{b}"


def act_str(action: int) -> str:
    if action == PASS_ACTION:
        return "pasa"
    t, side = action // 2, action % 2
    return f"{tile_str(t)} {'izq' if side == 0 else 'der'}"


def policy_act(policy, env, seat, device):
    o = torch.as_tensor(env.observe(seat), dtype=torch.float32,
                        device=device).unsqueeze(0)
    m = torch.as_tensor(env.action_masks(), dtype=torch.bool,
                        device=device).unsqueeze(0)
    a, _, _ = policy.act(o, m, deterministic=True)
    return int(a.item())


def play_hand(policy, policy_seats: tuple[int, ...], opp_name: str,
              seed: int, device: str):
    """One hand. Returns (transcript lines, result dict)."""
    opp = AGENTS[opp_name]()
    env = DominoEnv(mode="teams", n_players=4, seed=seed)
    env.reset()
    policy.eval()
    lines: list[str] = []
    moves: list[dict] = []
    # track board ends manually for the transcript
    ends: list[int | None] = [None, None]
    done = False
    while not done:
        seat = env.round.turn
        hand_before = len(env.round.hands[seat])
        if seat in policy_seats:
            action = policy_act(policy, env, seat, device)
        else:
            action = int(opp.act(env.observe(seat), env.action_masks()))
        if action == PASS_ACTION:
            desc = "pasa"
        else:
            t, side = action // 2, action % 2
            a, b = TILES[t]
            if ends[0] is None:
                ends = [a, b]
                desc = f"abre con {tile_str(t)}"
            else:
                assert a == ends[side] or b == ends[side], \
                    f"illegal transcript: {tile_str(t)} vs ends {ends}"
                ends[side] = b if a == ends[side] else a
                desc = f"{act_str(action)}"
        e = f"{ends[0]}-{ends[1]}" if ends[0] is not None else "-"
        tag = "*" if seat in policy_seats else " "
        lines.append(f"[t{len(moves):02d}]{tag} S{seat} ({hand_before}f): "
                     f"{desc}   | puntas {e}")
        moves.append({"seat": seat, "action": action,
                      "mine": seat in policy_seats})
        _, _, term, trunc, info = env.step(action)
        done = bool(term or trunc)
    wt = info["winner_team"]
    mine_won = wt in {0} if policy_seats == (0, 2) else wt in {1}
    result = {"winner_team": wt, "winner_seat": info["winner_seat"],
              "points": info["points"], "n_moves": info["n_moves"],
              "mine_won": mine_won,
              "domino": info["winner_seat"] is not None}
    return lines, moves, result


def behavior_stats(policy_path: str, manos: int, opponent: str,
                   seed: int = 0, device: str = "cpu") -> dict:
    policy = load_policy(policy_path, device)
    rng = np.random.default_rng(seed)
    s = {"hands": 0, "team_hands_won": 0, "domino_wins": 0,
         "tranca_wins": 0, "tantos_won": [], "tantos_lost": [],
         "pips_played": [], "doubles_played": 0, "tiles_played": 0,
         "passes": 0, "turns": 0,
         "after_partner_pass_played": 0, "after_partner_pass_turns": 0}
    for i in range(manos):
        policy_seats = (0, 2) if i % 2 == 0 else (1, 3)
        lines, moves, res = play_hand(
            policy, policy_seats, opponent,
            int(rng.integers(0, 2 ** 31 - 1)), device)
        s["hands"] += 1
        if res["mine_won"]:
            s["team_hands_won"] += 1
            s["tantos_won"].append(res["points"])
            if res["domino"]:
                s["domino_wins"] += 1
            else:
                s["tranca_wins"] += 1
        else:
            s["tantos_lost"].append(res["points"])
        last_pass: dict[int, bool] = {}
        for mv in moves:
            seat = mv["seat"]
            if mv["mine"]:
                partner = seat ^ 2  # 0<->2, 1<->3
                if last_pass.get(partner, False):
                    # partner is stuck: passed on their last turn
                    s["after_partner_pass_turns"] += 1
                    if mv["action"] != PASS_ACTION:
                        s["after_partner_pass_played"] += 1
                s["turns"] += 1
                if mv["action"] == PASS_ACTION:
                    s["passes"] += 1
                else:
                    t = mv["action"] // 2
                    s["pips_played"].append(pips(t))
                    s["tiles_played"] += 1
                    a, b = TILES[t]
                    if a == b:
                        s["doubles_played"] += 1
            last_pass[seat] = (mv["action"] == PASS_ACTION)
    return s


def report(s: dict, label: str) -> str:
    h = s["hands"]
    out = [f"--- {label} ({h} manos) ---"]
    out.append(f"manos ganadas por el equipo: "
               f"{s['team_hands_won']/h:.1%}  "
               f"(dominó {s['domino_wins']/h:.1%} / tranca {s['tranca_wins']/h:.1%})")
    if s["tantos_won"]:
        out.append(f"tantos/manoganada: {np.mean(s['tantos_won']):.1f}  | "
                   f"tantos/manoperdida: {np.mean(s['tantos_lost']):.1f}")
    if s["tiles_played"]:
        out.append(f"tantos por ficha jugada: {np.mean(s['pips_played']):.1f}  | "
                   f"dobles jugados: {s['doubles_played']/s['tiles_played']:.1%}  | "
                   f"pases: {s['passes']/s['turns']:.1%} de los turnos")
    if s["after_partner_pass_turns"]:
        out.append(f"con el compañero trancado juega: "
                   f"{s['after_partner_pass_played']/s['after_partner_pass_turns']:.1%} "
                   f"({s['after_partner_pass_turns']} casos)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--manos", type=int, default=200)
    ap.add_argument("--opponent", default="greedy",
                    choices=["greedy", "random"])
    ap.add_argument("--transcripts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    policy = load_policy(args.policy, "cpu")
    s = behavior_stats(args.policy, args.manos, args.opponent,
                       seed=args.seed)
    print(report(s, f"policy vs {args.opponent}"))
    print()
    rng = np.random.default_rng(args.seed + 1)
    for k in range(args.transcripts):
        ps = (0, 2) if k % 2 == 0 else (1, 3)
        lines, _, res = play_hand(
            policy, ps, args.opponent,
            int(rng.integers(0, 2 ** 31 - 1)), "cpu")
        tag = "GANA" if res["mine_won"] else "PIERDE"
        print(f"=== mano {k+1} ({tag}, {res['points']} tantos, "
              f"{res['n_moves']} jugadas) ===")
        print("\n".join(lines[:14]))
        if len(lines) > 14:
            print(f"... (+{len(lines)-14} jugadas)")
        print()


if __name__ == "__main__":
    main()
