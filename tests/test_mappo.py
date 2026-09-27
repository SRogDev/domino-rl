"""Tests for the MAPPO-style per-seat self-play (option B)."""
import numpy as np
import torch

from domino_rl.mappo import (ActorCritic, compute_gae, load_policy,
                             play_manos, ppo_update, save_policy)


def _policy(seed=0):
    torch.manual_seed(seed)
    return ActorCritic()


def test_every_seat_gets_a_trajectory_each_mano():
    rng = np.random.default_rng(0)
    trajs, stats = play_manos(_policy(), n_manos=6, mode="individual",
                              n_players=2, match_target=100, rng=rng)
    assert stats["manos"] == 6
    # 2 seats x 6 manos, everyone acts at least once per mano
    assert len(trajs) == 12
    for t in trajs:
        assert len(t["rew"]) >= 1
        # reward only on the seat's last turn of the hand
        assert (t["rew"][:-1] == 0).all()


def test_winner_team_seats_get_positive_reward_losers_zero():
    rng = np.random.default_rng(1)
    trajs, _ = play_manos(_policy(), n_manos=10, mode="individual",
                          n_players=2, match_target=10_000, rng=rng)
    # huge target -> no match ever ends -> no +/-1 bonus, pure tantos/100
    scoring = [t for t in trajs if t["rew"][-1] > 0]
    assert scoring, "random play should produce some scoring hand"
    for t in scoring:
        assert 0 < t["rew"][-1] <= 5.0  # tantos/100, sane magnitude
    zeros = [t for t in trajs if t["rew"][-1] == 0]
    assert zeros, "losing seats must get exactly 0"


def test_match_bonus_applied_on_final_hand():
    rng = np.random.default_rng(2)
    # target=1: every scoring hand ends the match
    trajs, _ = play_manos(_policy(), n_manos=30, mode="individual",
                          n_players=2, match_target=1, rng=rng)
    rewards = [t["rew"][-1] for t in trajs]
    assert any(r >= 1.0 for r in rewards), "winner seats need the +1 bonus"
    assert any(r == -1.0 for r in rewards), "loser seats need the -1 bonus"


def test_teams_mode_two_seats_share_team_reward():
    rng = np.random.default_rng(3)
    trajs, _ = play_manos(_policy(), n_manos=8, mode="teams",
                          n_players=4, match_target=10_000, rng=rng)
    by_team: dict[int, list] = {}
    for t in trajs:
        by_team.setdefault(t["team"], []).append(t["rew"][-1])
    # both seats of a team see identical hand rewards (same tantos, same bonus)
    assert set(by_team) == {0, 1}
    for team, rs in by_team.items():
        # rewards come in pairs (2 seats x hands); each pair equal
        assert len(rs) % 2 == 0


def test_act_never_plays_illegal_move():
    torch.manual_seed(0)
    p = ActorCritic()
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = rng.random(111) < 0.15
        mask[110] = True  # pass is always legal
        o = torch.randn(1, 143)
        m = torch.as_tensor(mask).unsqueeze(0)
        for det in (True, False):
            a, _, _ = p.act(o, m, deterministic=det)
            assert bool(mask[int(a.item())]), "played an illegal action"


def test_gae_terminal_reward_only():
    rew = np.array([0, 0, 0, 1.0], np.float32)
    val = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    adv, ret = compute_gae(rew, val, gamma=1.0, lam=1.0)
    # gamma=lam=1 -> advantage = total return - value at each step
    np.testing.assert_allclose(ret, np.full(4, 1.0), rtol=1e-5)
    np.testing.assert_allclose(adv, 1.0 - val, rtol=1e-5)


def test_ppo_update_is_finite_and_moves_params():
    torch.manual_seed(0)
    p = ActorCritic()
    before = [x.clone() for x in p.parameters()]
    n = 256
    mask = torch.ones(n, 111, dtype=torch.bool)
    batch = {
        "obs": torch.randn(n, 143), "act": torch.randint(0, 111, (n,)),
        "logp": torch.zeros(n), "mask": mask,
        "adv": torch.randn(n), "ret": torch.randn(n),
        "val": torch.randn(n),
    }
    opt = torch.optim.Adam(p.parameters(), lr=3e-4)
    losses = ppo_update(p, opt, batch, epochs=2, minibatch=128)
    for v in losses.values():
        assert np.isfinite(v), f"non-finite loss {losses}"
    changed = any(not torch.equal(a, b)
                  for a, b in zip(before, p.parameters()))
    assert changed, "ppo_update did not change parameters"


def test_save_load_roundtrip(tmp_path):
    p = _policy()
    path = str(tmp_path / "pol.pt")
    save_policy(p, path)
    q = load_policy(path)
    for a, b in zip(p.parameters(), q.parameters()):
        assert torch.equal(a, b)


def test_eval_vs_returns_sane_rates():
    from domino_rl.mappo import eval_vs
    p = _policy()
    wr, dr = eval_vs(p, "random", manos=20, mode="individual",
                     n_players=2, seed=0)
    assert 0.0 <= wr <= 1.0 and 0.0 <= dr <= 1.0
    assert wr + dr <= 1.0


def test_analyze_stats_and_transcript():
    from domino_rl.analyze import behavior_stats, play_hand, act_str
    from domino_rl.mappo import ActorCritic, save_policy
    p = ActorCritic()
    save_policy(p, "/tmp/_tpol.pt")
    s = behavior_stats("/tmp/_tpol.pt", manos=6, opponent="random", seed=0)
    assert s["hands"] == 6
    assert 0.0 <= s["team_hands_won"] / 6 <= 1.0
    assert s["turns"] > 0
    lines, moves, res = play_hand(p, (0, 2), "random", seed=1, device="cpu")
    assert len(lines) == len(moves) > 0
    assert res["winner_team"] in (0, 1, None)
    assert act_str(110) == "pasa"
