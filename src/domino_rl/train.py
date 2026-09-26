"""Train the domino agent with self-play PPO (masked actions).

Phase 5 of the plan. Everything runs on CPU: the network is tiny
(143 -> 128 -> 128 -> 111) and the simulator is pure Python, so the free
Colab tier (or even a laptop) is enough.

Usage (local):
    pip install -e ".[train]"
    python -m domino_rl.train --timesteps 2000000 --n-envs 8

Usage (Colab, free tier):
    !git clone https://github.com/SRogDev/domino-rl.git
    %cd domino-rl
    !pip install -e ".[train]"
    !python -m domino_rl.train --timesteps 2000000 --n-envs 4 --out /content/drive/MyDrive/domino-rl

Evaluate later with:
    python -m domino_rl.evaluate --a ppo:checkpoints/final.zip --b greedy --rounds 500
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def make_vec_env(mode: str, reward_mode: str, shaping_coef: float,
                 seed: int, n_envs: int, n_players: int):
    """DummyVecEnv of masked domino envs (single process, no pickling pain)."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .env import DominoEnv

    def _thunk(rank: int):
        def _init():
            env = DominoEnv(mode=mode, reward_mode=reward_mode,
                            shaping_coef=shaping_coef, n_players=n_players,
                            seed=seed + rank)
            env = Monitor(env)  # episode stats -> ep_rew_mean in logs
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    class MaskableDummyVecEnv(DummyVecEnv):
        def action_masks(self) -> np.ndarray:  # read by MaskablePPO
            return np.stack([env.action_masks() for env in self.envs])

    return MaskableDummyVecEnv([_thunk(i) for i in range(n_envs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--mode", default="teams", choices=["teams", "individual"])
    ap.add_argument("--n-players", type=int, default=4, choices=[2, 4],
                    help="2 = curriculum step 1 (1v1, hidden info); "
                         "4 = full game")
    ap.add_argument("--reward-mode", default="sparse", choices=["sparse", "shaped"],
                    help="sparse = pure win/loss (bitter lesson default); "
                         "shaped = experimental dense nudge")
    ap.add_argument("--shaping-coef", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--net", type=int, nargs=2, default=[128, 128])
    ap.add_argument("--resume", action="store_true",
                    help="continue training from --out/final.zip if it exists")
    ap.add_argument("--ckpt-every", type=int, default=250_000,
                    help="save a checkpoint every N timesteps (Colab safety)")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(args.out, exist_ok=True)
    vec_env = make_vec_env(args.mode, args.reward_mode, args.shaping_coef,
                           args.seed, args.n_envs, args.n_players)

    final_path = os.path.join(args.out, "final.zip")
    if args.resume and os.path.exists(final_path):
        print("resuming from", final_path)
        model = MaskablePPO.load(final_path, env=vec_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(net_arch=list(args.net)),
            n_steps=1024,
            batch_size=512,
            n_epochs=4,
            gamma=0.995,          # rounds are short; value the actual outcome
            learning_rate=3e-4,
            seed=args.seed,
            verbose=1,
        )
    # CheckpointCallback counts VecEnv steps (1 call = n_envs timesteps)
    ckpt = CheckpointCallback(
        save_freq=max(1, args.ckpt_every // args.n_envs),
        save_path=args.out, name_prefix="ckpt")
    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    path = os.path.join(args.out, "final")
    model.save(path)
    print("saved:", path + ".zip")


if __name__ == "__main__":
    main()
