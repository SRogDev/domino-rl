#!/usr/bin/env python3
"""Domino RL - la mesa digital: humanos (desde el movil) vs IA.

Cada humano abre http://<ip-del-servidor>:8000 en su telefono y se une
a la mesa. La IA ocupa las sillas restantes. En modo parejas, los dos
humanos son pareja contra dos IAs (Fase 10 del plan).

Run from the repo root:
    PYTHONPATH=src python demo/server.py
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from domino_rl.baselines import AGENTS
from domino_rl.env import PASS_ACTION, DominoEnv
from domino_rl.game import TILES

HERE = os.path.dirname(os.path.abspath(__file__))
PPO_PATH = os.path.join(HERE, "models", "ppo_smoke.zip")
MATCH_TARGET = 100


class PPOStudent:
    """The learning agent. Weak today, champion later - same interface."""

    def __init__(self, path: str = PPO_PATH):
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(path)

    def act(self, obs, mask) -> int:
        action, _ = self.model.predict(obs, action_masks=mask,
                                      deterministic=True)
        return int(action)


def decode(action: int):
    if action == PASS_ACTION:
        return None
    return action // 2, action % 2  # tile, side (0=left, 1=right)


def seat_order_for(mode: str) -> list[int]:
    # humans join in partner seats first (teams: 0 & 2 are partners)
    return [0, 2] if mode == "teams" else [0, 1, 2, 3]


class DemoGame:
    def __init__(self):
        self.lock = threading.Lock()
        self.match = [0, 0]
        self.mode = "teams"
        self.opponent = "greedy"
        self.env: DominoEnv | None = None
        self.log: list[str] = []
        self.student: PPOStudent | None = None
        self.human_seats: list[int] = []

    # ------------------------------------------------------------ opponents
    def make_opp(self, name: str):
        if name == "ppo":
            if self.student is None:
                self.student = PPOStudent()
            return self.student
        return AGENTS[name]()

    def role_of(self, seat: int) -> str:
        if seat in self.human_seats:
            return f"Humano {self.human_seats.index(seat) + 1}"
        return "IA"

    # ------------------------------------------------------------------ flow
    def create_table(self, mode: str, opponent: str) -> int:
        """First human creates the table and takes the first seat."""
        self.mode = mode
        self.opponent = opponent
        self.match = [0, 0]
        self.human_seats = [seat_order_for(mode)[0]]
        self.new_round()
        return self.human_seats[0]

    def join_table(self) -> int:
        """Another human joins an existing table."""
        if self.env is None:
            raise ValueError("no hay mesa creada todavia")
        for s in seat_order_for(self.mode):
            if s not in self.human_seats:
                self.human_seats.append(s)
                self.log.append(f"{self.role_of(s)} se unio a la mesa")
                return s
        raise ValueError("la mesa esta llena")

    def new_round(self):
        self.env = DominoEnv(mode=self.mode, reward_mode="sparse")
        self.env.reset()
        r = self.env.round
        self.log = [f"Mano nueva - sale {self.role_of(r.turn)}"]
        self.autoplay()

    def autoplay(self):
        """Play AI seats until a human's turn or the round ends."""
        opp = self.make_opp(self.opponent)
        r = self.env.round
        while not r.done and r.turn not in self.human_seats:
            seat = r.turn
            obs = self.env.observe(seat)
            mask = self.env.action_masks()
            a = opp.act(obs, mask)
            mv = decode(a)
            if mv is None:
                self.log.append(f"{self.role_of(seat)} paso")
            else:
                t, side = mv
                ta, tb = TILES[t]
                arrow = "<-" if side == 0 else "->"
                self.log.append(
                    f"{self.role_of(seat)} jugo {ta}-{tb} {arrow}")
            self.env.step(a)
        if r.done:
            self.score_round()

    def score_round(self):
        r = self.env.round
        w = r.winner_team
        pts = r.points_awarded
        if w is None:
            self.log.append("Tranca empatada - nadie anota")
            return
        while len(self.match) <= w:
            self.match.append(0)
        self.match[w] += pts
        if self.mode == "teams":
            us = "ganaron los humanos" if w == 0 else "ganaron las IAs"
        else:
            us = f"gano {self.role_of(w)}"
        self.log.append(f"Mano: {us} (+{pts} tantos)")

    def match_winner(self):
        for i, s in enumerate(self.match):
            if s >= MATCH_TARGET:
                who = self.role_of(i) if self.mode == "individual" else (
                    "los humanos" if i == 0 else "las IAs")
                return f"¡Ganaron {who} el partido!"
        return None

    # ----------------------------------------------------------------- state
    def state(self, seat: int) -> dict:
        r = self.env.round
        hand = [{"tile": t, "a": TILES[t][0], "b": TILES[t][1]}
                for t in sorted(r.hands[seat])]
        legal = [{"tile": t, "side": s} for t, s in r.legal_moves(seat)]
        return {
            "mode": self.mode,
            "opponent": self.opponent,
            "seat": seat,
            "you_are": self.role_of(seat),
            "seats": {str(s): self.role_of(s) for s in range(r.n_players)},
            "humans": len(self.human_seats),
            "hand": hand,
            "board": [[TILES[t][0], TILES[t][1]] for t in r.board],
            "ends": list(r.ends) if r.ends else None,
            "legal": legal,
            "must_pass": (not r.done and r.turn == seat and not legal),
            "your_turn": (not r.done and r.turn == seat),
            "turn_name": None if r.done else self.role_of(r.turn),
            "done": r.done,
            "log": self.log[-8:],
            "match": self.match,
            "target": MATCH_TARGET,
            "match_over": self.match_winner(),
            "remaining": [len(r.hands[s]) for s in range(r.n_players)],
            "ppo_available": os.path.exists(PPO_PATH),
        }

    def human_move(self, seat: int, tile: int | None,
                   side: int | None) -> None:
        r = self.env.round
        if r.done or r.turn != seat:
            raise ValueError("no es tu turno")
        if tile is None:
            action = PASS_ACTION
        else:
            if (tile, side) not in r.legal_moves(seat):
                raise ValueError("jugada ilegal")
            action = tile * 2 + side
        self.env.step(action)
        if tile is None:
            self.log.append(f"{self.role_of(seat)} paso")
        else:
            ta, tb = TILES[tile]
            arrow = "<-" if side == 0 else "->"
            self.log.append(f"{self.role_of(seat)} jugo {ta}-{tb} {arrow}")
        self.autoplay()


GAME = DemoGame()


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self):
        with open(os.path.join(HERE, "index.html"), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._page()
        elif self.path == "/api/info":
            with GAME.lock:
                self._json({"table_exists": GAME.env is not None,
                            "ppo_available": os.path.exists(PPO_PATH)})
        elif self.path.startswith("/api/state"):
            try:
                seat = int(self.path.split("seat=")[1].split("&")[0])
            except (IndexError, ValueError):
                return self._json({"error": "falta ?seat=N"}, 400)
            with GAME.lock:
                if GAME.env is None:
                    return self._json({"error": "no hay mesa"}, 400)
                self._json(GAME.state(seat))
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        with GAME.lock:
            try:
                if self.path == "/api/create":
                    mode = data.get("mode", "teams")
                    opponent = data.get("opponent", "greedy")
                    if opponent not in ("greedy", "random", "ppo"):
                        return self._json({"error": "oponente desconocido"},
                                           400)
                    seat = GAME.create_table(mode, opponent)
                    self._json({"seat": seat, **GAME.state(seat)})
                elif self.path == "/api/join":
                    seat = GAME.join_table()
                    self._json({"seat": seat, **GAME.state(seat)})
                elif self.path == "/api/next":
                    if GAME.match_winner():
                        return self._json({"error": "el partido termino"},
                                            400)
                    GAME.new_round()
                    self._json(GAME.state(int(data.get("seat", 0))))
                elif self.path == "/api/move":
                    GAME.human_move(int(data["seat"]), data.get("tile"),
                                    data.get("side"))
                    self._json(GAME.state(int(data["seat"])))
                else:
                    self.send_error(404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # e.g. missing ppo file
                self._json({"error": str(e)}, 400)

    def log_message(self, *args):
        pass  # quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Cada jugador abre en su telefono: http://<ip-de-esta-maquina>:{args.port}")
    print("Pulsa Ctrl+C para parar.")
    srv.serve_forever()


if __name__ == "__main__":
    main()
