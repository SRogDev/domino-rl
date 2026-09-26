"""Gymnasium-style environment for Cuban double-9 domino.

Design for RL training (shared-policy self-play):
  - The env drives all 4 seats. Every `step` acts for whoever's turn it is,
    and the observation is always from *that seat's* perspective.
  - A single policy therefore learns to play any seat / any team. The reward
    of each transition is computed from the acting seat's team perspective,
    which is the standard trick that makes single-agent PPO/DQN work for
    multiplayer self-play.
  - Episode = one round ("mano"). `truncated` is only a safety cap.

Observation (Box[98], float32), from the acting seat's perspective:
  - hand one-hot ......................... 55
  - board ends L one-hot, R one-hot ...... 20
  - pips of each value already played .... 10   (/11)
  - tiles remaining per seat ............. 4    (/10)
  - consecutive passes ................... 1    (/4)
  - acting seat one-hot .................. 4
  - teammate seat one-hot ................ 4    (zeros in individual mode)

Action (Discrete[111]): 55 tiles x {left, right} + pass (110).
`action_masks()` returns the legal-move mask (for sb3-contrib MaskablePPO).
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym, spaces = None, None

from .game import N_TILES, TILES, Round, new_round

N_ACTIONS = N_TILES * 2 + 1
PASS_ACTION = N_TILES * 2
OBS_DIM = 98
MAX_STEPS = 500


def _encode_action(tile: int, side: int) -> int:
    return tile * 2 + side


def _decode_action(action: int) -> tuple[int, int] | None:
    if action == PASS_ACTION:
        return None
    return action // 2, action % 2


class DominoEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self, mode: str = "teams", reward_mode: str = "shaped",
                 shaping_coef: float = 0.1, seed: int | None = None):
        if reward_mode not in ("sparse", "shaped"):
            raise ValueError("reward_mode must be 'sparse' or 'shaped'")
        self.mode = mode
        self.reward_mode = reward_mode
        self.shaping_coef = shaping_coef
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.round: Round | None = None
        self._steps = 0
        self._mask = np.zeros(N_ACTIONS, dtype=bool)
        if gym:
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(
                0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------ masks
    def action_masks(self) -> np.ndarray:
        return self._mask

    def _update_mask(self) -> None:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        moves = self.round.legal_moves(self.round.turn)
        if moves:
            for tile, side in moves:
                mask[_encode_action(tile, side)] = True
        else:
            mask[PASS_ACTION] = True
        self._mask = mask

    # ---------------------------------------------------------------- gym api
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        round_seed = int(self._rng.integers(0, 2**31 - 1))
        self.round = new_round(seed=round_seed, mode=self.mode)
        self._steps = 0
        self._update_mask()
        return self.observe(self.round.turn), {}

    def step(self, action: int):
        r = self.round
        seat = r.turn
        if not self._mask[action]:
            raise ValueError(f"illegal action {action} for seat {seat}")
        r.apply(seat, _decode_action(action))
        self._steps += 1

        reward = 0.0
        terminated = r.done
        if terminated:
            reward = self._terminal_reward(seat)
            obs = self.observe(seat)  # acting seat's final view
        else:
            obs = self.observe(r.turn)
        truncated = self._steps >= MAX_STEPS
        self._update_mask()
        info = {
            "winner_team": r.winner_team,
            "winner_seat": r.winner_seat,
            "points": r.points_awarded,
            "n_moves": r.n_moves,
        }
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------- rewards
    def _terminal_reward(self, seat: int) -> float:
        r = self.round
        my_team = r.team_of(seat)
        if r.winner_team is None:
            base = 0.0
        elif r.winner_team == my_team:
            base = 1.0
        else:
            base = -1.0
        if self.reward_mode == "sparse" or r.winner_team is None:
            return base
        # small dense nudge: leave fewer pips than the opponents
        my_pips = sum(r.hand_pips(s) for s in range(4)
                      if r.team_of(s) == my_team)
        opp_pips = sum(r.hand_pips(s) for s in range(4)
                       if r.team_of(s) != my_team)
        return base + self.shaping_coef * (opp_pips - my_pips) / 100.0

    # ------------------------------------------------------------------ obs
    def observe(self, seat: int) -> np.ndarray:
        r = self.round
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        o = 0
        # hand one-hot (55)
        for t in r.hands[seat]:
            obs[o + t] = 1.0
        o += 55
        # board ends (20)
        if r.ends is not None:
            L, R = r.ends
            obs[o + L] = 1.0
            obs[o + 10 + R] = 1.0
        o += 20
        # pips of each value already played (10), /11
        for t in r.board:
            a, b = TILES[t]
            obs[o + a] += 1.0
            obs[o + b] += 1.0
        obs[o:o + 10] /= 11.0
        o += 10
        # tiles remaining per seat (4), /10
        for s in range(4):
            obs[o + s] = len(r.hands[s]) / 10.0
        o += 4
        # consecutive passes (1), /4
        obs[o] = r.passes_consecutive / 4.0
        o += 1
        # acting seat one-hot (4)
        obs[o + seat] = 1.0
        o += 4
        # teammate one-hot (4); zeros in individual mode
        if r.mode == "teams":
            mate = next(s for s in range(4)
                        if s != seat and r.team_of(s) == r.team_of(seat))
            obs[o + mate] = 1.0
        o += 4
        assert o == OBS_DIM
        return obs
