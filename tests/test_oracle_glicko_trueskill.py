"""Orakel-Tests: Glicko gegen Glickmans Formeln in Punkten (q, g(RD), E mit Zehnerpotenz, d²), TrueSkill gegen das Paket `trueskill` in den Originaleinheiten
(25 / 8,33 / 4,17), Elo gegen die Lehrbuchformel 10^(ΔR/400). Alles auf Zufallsfolgen mit Neuling, Pause und ungleicher Aktivität; jeweils unter 10 Sekunden."""

import math
import random

import numpy as np
import pytest

import glt_constants as C
import glt_drift as D
import glt_elo as E
import glt_glicko as G
import glt_trueskill as T

Q = math.log(10.0) / 400.0
SCALE_ORIG = C.BETA / (25.0 / 6.0)


def matches(seed, n=6, rounds=40, **kw):
    args = dict(delta=0.3, sigma=0.05, p_rare=1.0, join_round=0, gap=0.0, pause_start=0, pause_len=0)
    args.update(kw)
    return D.simulate_matches(n, args["delta"], rounds, args["sigma"], args["p_rare"], args["join_round"], args["gap"], args["pause_start"], args["pause_len"], seed)


SCENARIOS = [dict(), dict(p_rare=0.3), dict(join_round=15, gap=1.0, sigma=0.0), dict(pause_start=10, pause_len=15), dict(p_rare=0.5, sigma=0.1)]


# --- Glicko nach Glickman (1999), in Punkten ---------------------------------------------------------------------------------------------------

def g_pts(rd):
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / math.pi ** 2)


def e_pts(r, r_j, rd_j):
    return 1.0 / (1.0 + 10.0 ** (-g_pts(rd_j) * (r - r_j) / 400.0))


def glicko_pts(r, rd, games):
    """r' und RD' für eine Periode; games = [(r_j, rd_j, s_j)]."""
    d2 = 1.0 / (Q * Q * sum(g_pts(rd_j) ** 2 * e_pts(r, r_j, rd_j) * (1.0 - e_pts(r, r_j, rd_j)) for r_j, rd_j, _s in games))
    r_new = r + Q / (1.0 / rd ** 2 + 1.0 / d2) * sum(g_pts(rd_j) * (s - e_pts(r, r_j, rd_j)) for r_j, rd_j, s in games)
    return r_new, math.sqrt(1.0 / (1.0 / rd ** 2 + 1.0 / d2))


def test_glicko_update_equals_the_textbook_formulas_in_points():
    rng = random.Random(1)
    for _ in range(300):
        r, rd = rng.uniform(1000, 2000), rng.uniform(30, 350)
        games = [(rng.uniform(1000, 2000), rng.uniform(30, 350), rng.choice((0.0, 0.5, 1.0))) for _ in range(rng.randint(1, 5))]
        want = glicko_pts(r, rd, games)
        mu, phi = G.glicko_update((r - 1500) / C.SCALE, rd / C.SCALE, [((rj - 1500) / C.SCALE, rdj / C.SCALE, s) for rj, rdj, s in games])
        assert 1500 + C.SCALE * mu == pytest.approx(want[0], abs=1e-6)
        assert C.SCALE * phi == pytest.approx(want[1], abs=1e-6)


def glicko_run_pts(m, c_pts, rd0_pts):
    """Eigenständiger Lauf nach Glickman in Punkten: Schritt 1 (RD wächst bis 350 bzw. RD0), Schritt 2 mit den Werten vom Periodenbeginn."""
    n = m.n
    r = [1500.0] * n
    rd = [rd0_pts] * n
    for t in range(m.rounds):
        if m.join_round and t == m.join_round:
            r[0] = sum(r[1:]) / (n - 1)
            rd[0] = rd0_pts
        pre = [min(math.sqrt(rd[q] ** 2 + c_pts ** 2), rd0_pts) if m.presence[t][q] else rd[q] for q in range(n)]
        mine = {q: [] for q in range(n)}
        for k in np.nonzero(m.rnd == t)[0]:
            i, j, w = int(m.i[k]), int(m.j[k]), int(m.win[k])
            mine[i].append((r[j], pre[j], float(w)))
            mine[j].append((r[i], pre[i], float(1 - w)))
        new_r, new_rd = list(r), list(pre)
        for q in range(n):
            if mine[q]:
                new_r[q], new_rd[q] = glicko_pts(r[q], pre[q], mine[q])
        r, rd = new_r, new_rd
    return r, rd


@pytest.mark.parametrize("kw", SCENARIOS)
def test_glicko_run_equals_an_independent_run_in_points(kw):
    for seed in range(4):
        m = matches(seed, **kw)
        run = G.run_glicko(m, c=8.0 / C.SCALE, phi0=350.0 / C.SCALE)
        r, rd = glicko_run_pts(m, 8.0, 350.0)
        assert 1500 + C.SCALE * run.final == pytest.approx(r, abs=1e-6)
        assert C.SCALE * run.sd[-1][m.presence[-1]] == pytest.approx(np.array(rd)[m.presence[-1]], abs=1e-6)


# --- Elo nach der Lehrbuchformel ---------------------------------------------------------------------------------------------------------------

def elo_run_pts(m, K, K0=None, g0=10.0):
    n = m.n
    r = [1500.0] * n
    played = [0] * n
    for t in range(m.rounds):
        if m.join_round and t == m.join_round:
            r[0] = sum(r[1:]) / (n - 1)
        for k in np.nonzero(m.rnd == t)[0]:
            i, j, w = int(m.i[k]), int(m.j[k]), float(m.win[k])
            e = 1.0 / (1.0 + 10.0 ** ((r[j] - r[i]) / 400.0))
            ki = K if K0 is None else K + (K0 - K) / (1.0 + played[i] / g0)
            kj = K if K0 is None else K + (K0 - K) / (1.0 + played[j] / g0)
            r[i] += ki * (w - e)
            r[j] -= kj * (w - e)
            played[i] += 1
            played[j] += 1
    return r


@pytest.mark.parametrize("kw", SCENARIOS)
def test_elo_run_equals_the_textbook_formulation_in_points(kw):
    for seed in range(4):
        m = matches(seed, **kw)
        for K, K0 in ((20.0, None), (28.0, 60.0)):
            run = E.run_elo(m, K, K0)
            assert 1500 + C.SCALE * run.final == pytest.approx(elo_run_pts(m, K, K0), abs=1e-6)


# --- TrueSkill gegen das Paket ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(), dict(p_rare=0.3), dict(join_round=15, gap=1.0, sigma=0.0), dict(pause_start=10, pause_len=15)])
def test_trueskill_run_equals_the_trueskill_package_in_the_original_units(kw):
    ts = pytest.importorskip("trueskill")
    tau_orig = 0.5
    env = ts.TrueSkill(mu=25.0, sigma=25.0 / 3.0, beta=25.0 / 6.0, tau=tau_orig, draw_probability=0.0)
    for seed in range(3):
        m = matches(seed, **kw)
        run = T.run_trueskill(m, tau=tau_orig * SCALE_ORIG, sigma0=(25.0 / 3.0) * SCALE_ORIG)
        ratings = [env.create_rating() for _ in range(m.n)]
        for t in range(m.rounds):
            if m.join_round and t == m.join_round:
                mean = sum(r.mu for r in ratings[1:]) / (m.n - 1)
                ratings[0] = env.create_rating(mu=mean, sigma=25.0 / 3.0)
            for k in np.nonzero(m.rnd == t)[0]:
                i, j, w = int(m.i[k]), int(m.j[k]), int(m.win[k])
                win, lose = (i, j) if w else (j, i)
                ratings[win], ratings[lose] = ts.rate_1vs1(ratings[win], ratings[lose], env=env)
        # Zentren: das Paket startet bei 25; die θ-Skala bei 0 und mit µ_θ = (µ − 25) · SCALE_ORIG
        want_mu = np.array([(r.mu - 25.0) * SCALE_ORIG for r in ratings])
        want_sd = np.array([r.sigma * SCALE_ORIG for r in ratings])
        ok = m.presence[-1]
        assert run.final[ok] == pytest.approx(want_mu[ok], abs=1e-6)
        assert run.sd[-1][ok] == pytest.approx(want_sd[ok], abs=1e-6)


def test_trueskill_update_with_a_draw_margin_equals_the_package():
    ts = pytest.importorskip("trueskill")
    rng = random.Random(7)
    for p_d in (0.0, 0.05, 0.10, 0.25):
        env = ts.TrueSkill(mu=25.0, sigma=25.0 / 3.0, beta=25.0 / 6.0, tau=0.0, draw_probability=p_d)
        eps = T.draw_margin(p_d)
        for _ in range(40):
            a = ts.Rating(rng.uniform(10, 40), rng.uniform(1, 9))
            b = ts.Rating(rng.uniform(10, 40), rng.uniform(1, 9))
            wa, lb = ts.rate_1vs1(a, b, env=env)
            got = T.trueskill_update((a.mu - 25) * SCALE_ORIG, a.sigma * SCALE_ORIG, (b.mu - 25) * SCALE_ORIG, b.sigma * SCALE_ORIG, C.BETA, eps)
            assert got[0] == pytest.approx((wa.mu - 25) * SCALE_ORIG, abs=1e-6) and got[1] == pytest.approx(wa.sigma * SCALE_ORIG, abs=1e-6)
            assert got[2] == pytest.approx((lb.mu - 25) * SCALE_ORIG, abs=1e-6) and got[3] == pytest.approx(lb.sigma * SCALE_ORIG, abs=1e-6)


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------

def test_glicko_with_certain_ratings_reduces_to_an_elo_like_step():
    """φ klein und c = 0: die Aktualisierung ist ein Elo-Schritt mit k = φ'² g, g ≈ 1."""
    mu, phi = G.glicko_update(0.0, 0.05, [(0.0, 1.0e-9, 1.0)])
    assert mu == pytest.approx(phi ** 2 * 0.5, rel=1e-6) and phi == pytest.approx(0.05, rel=1e-2)


def test_equal_inputs_give_equal_outputs_for_both_players():
    """Symmetrie: zwei gleiche Spieler, einer gewinnt: Wertungen spiegeln einander, Unsicherheiten sind gleich."""
    for run in (G.run_glicko(matches(0, n=2, rounds=1), 0.05, 2.0), T.run_trueskill(matches(0, n=2, rounds=1), 0.05, 2.0)):
        assert run.mu[0, 0] == pytest.approx(-run.mu[0, 1]) and run.sd[0, 0] == pytest.approx(run.sd[0, 1])
