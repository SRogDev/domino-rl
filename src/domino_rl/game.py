"""Cuban double-9 domino engine.

Pure-Python, dependency-free and fast: this is the *simulator* that will
generate the millions of synthetic games our RL agent learns from.

Rules implemented (standard Cuban "dominó de 9"):
  - 55 tiles (0-0 .. 9-9). 4 seats, 10 tiles each, 15 stay in the pozo.
  - Salida: the seat holding the highest double starts and MUST play it.
    (If no double was dealt, the highest tile by pips starts instead.)
  - Turn order is seat+1 mod 4. A tile (a, b) can be played on an end
    whose value equals a or b. No legal moves -> "paso" (pass).
  - A round ("mano") ends by "dominó" (a seat plays its last tile) or by
    "tranca" (4 consecutive passes, table locked).
  - Scoring ("tantos"): the winning team scores the sum of the pips left
    in the opponents' hands. Tranca: the team with fewer pips left wins and
    scores the opponents' pips. Exact tie -> draw, nobody scores.
  - Modes: "teams" (seats 0&2 vs 1&3, the Cuban way) or "individual"
    (every seat for itself).

An episode for RL = one round. Match-level play (first to 100) is left to
the evaluation scripts.
"""
from __future__ import annotations

import random

# All 55 tiles of the double-9 set, canonical order a <= b.
TILES: list[tuple[int, int]] = [(a, b) for a in range(10) for b in range(a, 10)]
N_TILES = len(TILES)
_TILE_INDEX = {t: i for i, t in enumerate(TILES)}
DOUBLES = [i for i, (a, b) in enumerate(TILES) if a == b]

LEFT, RIGHT = 0, 1
N_SEATS = 4
HAND_SIZE = 10


def tile_index(a: int, b: int) -> int:
    """Index of tile (a, b), order-independent."""
    if a > b:
        a, b = b, a
    return _TILE_INDEX[(a, b)]


def pips(t: int) -> int:
    a, b = TILES[t]
    return a + b


def new_round(seed: int | None = None, mode: str = "teams",
            n_players: int = 4) -> "Round":
    return Round(seed=seed, mode=mode, n_players=n_players)


class Round:
    """One 'mano' of Cuban double-9 domino."""

    def __init__(self, seed: int | None = None, mode: str = "teams",
                 n_players: int = 4):
        if mode not in ("teams", "individual"):
            raise ValueError(f"unknown mode {mode!r}")
        if n_players not in (2, 4):
            raise ValueError(f"n_players must be 2 or 4, got {n_players}")
        if mode == "teams" and n_players != 4:
            raise ValueError("teams mode needs 4 players")
        self.mode = mode
        self.n_players = n_players
        rng = random.Random(seed)
        deck = list(range(N_TILES))
        rng.shuffle(deck)
        self.hands: list[list[int]] = [sorted(deck[i * HAND_SIZE:(i + 1) * HAND_SIZE])
                                       for i in range(n_players)]
        self.pozo: list[int] = deck[n_players * HAND_SIZE:]

        self.board: list[int] = []          # tile indices in placement order
        self.ends: tuple[int, int] | None = None

        # --- salida: highest double in hand starts and must play it ---
        best, holder = -1, -1
        for s, hand in enumerate(self.hands):
            for t in hand:
                if t in DOUBLES and t > best:
                    best, holder = t, s
        if best < 0:  # no double dealt (possible: 15 tiles stay in the pozo)
            best_key, holder = (-1, -1), -1
            for s, hand in enumerate(self.hands):
                for t in hand:
                    key = (pips(t), t)
                    if key > best_key:
                        best_key, holder = key, s
            best = best_key[1]
        self.salida_tile: int = best
        self.turn: int = holder
        self.salida_pending: bool = True

        self.passes_consecutive = 0
        self.n_moves = 0
        self.done = False
        self.winner_seat: int | None = None
        self.winner_team: int | None = None
        self.points_awarded = 0
        self.loser_pips_at_end = 0

    # ------------------------------------------------------------------ teams
    def team_of(self, seat: int) -> int:
        if self.mode == "individual":
            return seat
        return 0 if seat in (0, 2) else 1

    def hand_pips(self, seat: int) -> int:
        return sum(pips(t) for t in self.hands[seat])

    # ------------------------------------------------------------------ moves
    def legal_moves(self, seat: int) -> list[tuple[int, int]]:
        """Legal (tile_idx, side) moves for `seat`. Empty list -> must pass."""
        if self.done or seat != self.turn:
            return []
        if self.salida_pending:
            return [(self.salida_tile, LEFT)]  # side irrelevant, canonical
        L, R = self.ends
        moves: list[tuple[int, int]] = []
        for t in self.hands[seat]:
            a, b = TILES[t]
            if a == L or b == L:
                moves.append((t, LEFT))
            if a == R or b == R:
                moves.append((t, RIGHT))
        return moves

    def apply(self, seat: int, move: tuple[int, int] | None) -> None:
        """Play `move` ((tile_idx, side)) or pass (None) for `seat`."""
        if self.done:
            raise ValueError("round is over")
        if seat != self.turn:
            raise ValueError(f"not seat {seat}'s turn (turn={self.turn})")
        legal = self.legal_moves(seat)
        if move is None:
            if legal:
                raise ValueError("cannot pass while legal moves exist")
            self.passes_consecutive += 1
            if self.passes_consecutive >= self.n_players:
                self._finish_tranca()
            else:
                self.turn = (self.turn + 1) % self.n_players
            return

        if move not in legal:
            raise ValueError(f"illegal move {move} for seat {seat}")
        tile, side = move
        self.hands[seat].remove(tile)
        self.board.append(tile)
        a, b = TILES[tile]
        if self.salida_pending:
            self.ends = (a, b)
            self.salida_pending = False
        else:
            L, R = self.ends
            if side == LEFT:
                self.ends = (b if a == L else a, R)
            else:
                self.ends = (L, b if a == R else a)
        self.passes_consecutive = 0
        self.n_moves += 1

        if not self.hands[seat]:
            self._finish_domino(seat)
        else:
            self.turn = (self.turn + 1) % self.n_players

    # -------------------------------------------------------------- finishing
    def _finish_domino(self, seat: int) -> None:
        team = self.team_of(seat)
        opponents = [s for s in range(self.n_players) if self.team_of(s) != team]
        loser_pips = sum(self.hand_pips(s) for s in opponents)
        self.done = True
        self.winner_seat = seat
        self.winner_team = team
        self.loser_pips_at_end = loser_pips
        self.points_awarded = loser_pips

    def _finish_tranca(self) -> None:
        if self.mode == "teams":
            t0 = self.hand_pips(0) + self.hand_pips(2)
            t1 = self.hand_pips(1) + self.hand_pips(3)
            if t0 == t1:
                self.done = True  # draw
                return
            winner = 0 if t0 < t1 else 1
            loser_pips = t1 if winner == 0 else t0
        else:
            seat_pips = [(self.hand_pips(s), s) for s in range(self.n_players)]
            seat_pips.sort()
            if seat_pips[0][0] == seat_pips[1][0]:
                self.done = True  # draw
                return
            winner = seat_pips[0][1]
            loser_pips = sum(p for p, s in seat_pips[1:])
        self.done = True
        self.winner_seat = None
        self.winner_team = winner
        self.loser_pips_at_end = loser_pips
        self.points_awarded = loser_pips

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"Round(turn={self.turn} ends={self.ends} "
                f"hands={[len(h) for h in self.hands]} done={self.done})")
