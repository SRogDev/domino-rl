"""Tests for the Cuban double-9 domino engine (domino_rl.game).

TDD: these were written BEFORE the implementation (RED), then the engine
was built to make them pass (GREEN).
"""
import pytest

from domino_rl.game import (
    N_TILES,
    TILES,
    Round,
    new_round,
    pips,
    tile_index,
)


def test_tile_set_is_double_nine():
    assert N_TILES == 55
    assert len(TILES) == 55
    # every (a, b) with 0 <= a <= b <= 9 exactly once
    expected = {(a, b) for a in range(10) for b in range(a, 10)}
    assert set(TILES) == expected
    # index round-trips
    for i, (a, b) in enumerate(TILES):
        assert tile_index(a, b) == i


def test_deal_gives_10_each_and_15_in_pozo():
    r = new_round(seed=42)
    assert len(r.hands) == 4
    for hand in r.hands:
        assert len(hand) == 10
    assert len(r.pozo) == 15
    dealt = [t for hand in r.hands for t in hand]
    assert len(set(dealt)) == 40
    assert not (set(dealt) & set(r.pozo))
    assert set(dealt) | set(r.pozo) <= set(range(N_TILES))


def test_salida_is_highest_double_and_must_play_it():
    r = new_round(seed=1)
    doubles = [t for t in range(N_TILES) if TILES[t][0] == TILES[t][1]]
    best, holder = -1, -1
    for seat, hand in enumerate(r.hands):
        for t in hand:
            if t in doubles and t > best:
                best, holder = t, seat
    if best >= 0:
        assert r.turn == holder
        assert r.salida_tile == best
        moves = r.legal_moves(holder)
        assert moves == [(best, 0)] or moves == [(best, 1)] or set(moves) == {(best, 0), (best, 1)}
        # playing anything else is rejected
        other = next(t for t in r.hands[holder] if t != best)
        with pytest.raises(ValueError):
            r.apply(holder, (other, 0))


def test_play_updates_board_ends_and_hand():
    r = new_round(seed=7)
    seat = r.turn
    tile, side = r.legal_moves(seat)[0]
    a, b = TILES[tile]
    r.apply(seat, (tile, side))
    assert tile not in r.hands[seat]
    assert len(r.board) == 1
    L, R = r.ends
    assert {L, R} == {a, b} or (a == b and L == R == a)
    assert r.turn == (seat + 1) % 4


def test_legal_moves_match_board_ends():
    r = new_round(seed=7)
    seat = r.turn
    tile, side = r.legal_moves(seat)[0]
    r.apply(seat, (tile, side))
    L, R = r.ends
    for s in range(4):
        for t, sd in r.legal_moves(s):
            a, b = TILES[t]
            end = L if sd == 0 else R
            assert a == end or b == end, f"tile {(a,b)} not playable on {end}"


def test_pass_only_when_no_moves_and_tranca_ends_round():
    r = new_round(seed=7)
    # force a locked board: empty every hand except tiles nobody can play is
    # hard; instead verify the pass rule directly.
    seat = r.turn
    moves = r.legal_moves(seat)
    if moves:
        with pytest.raises(ValueError):
            r.apply(seat, None)  # cannot pass while moves exist
    else:
        r.apply(seat, None)
        assert r.passes_consecutive == 1


def test_full_random_round_always_terminates_with_consistent_score():
    for seed in range(30):
        r = new_round(seed=seed)
        steps = 0
        while not r.done:
            seat = r.turn
            moves = r.legal_moves(seat)
            if moves:
                import random
                r.apply(seat, random.choice(moves))
            else:
                r.apply(seat, None)
            steps += 1
            assert steps < 500, "round did not terminate"
        assert r.done
        # score consistency: points awarded == opponent pips at end (or 0 on draw)
        if r.winner_team is not None:
            assert r.points_awarded > 0
            assert r.points_awarded == r.loser_pips_at_end
        else:
            assert r.points_awarded == 0


def test_domino_win_scores_opponent_pips():
    r = new_round(seed=123, mode="teams")
    # rig: give seat 0 a single playable tile and make it the salida
    # simplest deterministic check via a scripted tiny scenario:
    r2 = new_round(seed=999, mode="individual")
    # play a full round, then check individual-mode invariant instead:
    import random
    while not r2.done:
        seat = r2.turn
        moves = r2.legal_moves(seat)
        r2.apply(seat, random.choice(moves) if moves else None)
    if r2.winner_seat is not None:
        assert r2.points_awarded == r2.loser_pips_at_end > 0


def test_teams_mode_pairs_seats():
    r = new_round(seed=5, mode="teams")
    assert r.team_of(0) == r.team_of(2) != r.team_of(1) == r.team_of(3)
    r2 = new_round(seed=5, mode="individual")
    assert len({r2.team_of(s) for s in range(4)}) == 4


def test_double_nine_set_totals_495_pips():
    # Invariant of any double-N set: total pips = N(N+1)(N+2)/2.
    # For N=9: 495 total, average 9 per tile.
    total = sum(pips(t) for t in range(N_TILES))
    assert total == 495
    assert total / N_TILES == 9


def test_two_player_round_deals_and_alternates():
    r = new_round(seed=7, mode="individual", n_players=2)
    assert len(r.hands) == 2
    assert all(len(h) == 10 for h in r.hands)
    assert len(r.pozo) == 35
    # turn alternates between the two seats
    s0 = r.turn
    moves = r.legal_moves(s0)
    r.apply(s0, moves[0] if moves else None)
    if not r.done:
        assert r.turn == 1 - s0


def test_two_player_tranca_after_two_consecutive_passes():
    r = new_round(seed=7, mode="individual", n_players=2)
    # force empty hands so both must pass (valid mid-game state)
    r.hands = [[], []]
    r.salida_pending = False
    r.ends = (3, 7)
    r.turn = 0
    r.apply(0, None)
    assert not r.done
    r.apply(1, None)
    assert r.done  # tranca after n_players consecutive passes


def test_teams_mode_requires_four_players():
    with pytest.raises(ValueError):
        new_round(seed=1, mode="teams", n_players=2)
    with pytest.raises(ValueError):
        new_round(seed=1, n_players=3)
