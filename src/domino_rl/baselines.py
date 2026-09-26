"""Baseline agents: the opponents our RL agent must learn to beat.

Every RL project needs these FIRST: without a baseline you can't tell
whether the agent learned anything or is just flailing.
"""
from __future__ import annotations

import numpy as np

from .env import PASS_ACTION
from .game import TILES, pips


class RandomAgent:
    """Plays a uniformly random legal move."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


class GreedyAgent:
    """Heuristic: always play the highest-pip legal tile.

    Not a great domino player (it ignores defense, counting and partner
    play), but it punishes a policy that plays randomly. A trained agent
    should beat it clearly.
    """

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            return int(legal[0])
        best, best_key = int(legal[0]), (-1, -1)
        for a in legal:
            if a == PASS_ACTION:
                continue
            tile = a // 2
            # tie-break: doubles first (harder to place late)
            key = (pips(tile), 1 if TILES[tile][0] == TILES[tile][1] else 0)
            if key > best_key:
                best_key, best = key, int(a)
        return best


AGENTS = {"random": RandomAgent, "greedy": GreedyAgent}
