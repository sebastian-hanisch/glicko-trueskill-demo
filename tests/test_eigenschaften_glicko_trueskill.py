"""Eigenschaftstests (metamorph) für Glicko, TrueSkill und Elo dieser Demo (ORACLE-PLAYBOOK §5.2): Man verändert die Eingabe auf eine Weise, deren Wirkung auf die
Ausgabe aus dem Verfahren folgt, und prüft die Beziehung auf Zufallsinstanzen mit festem Seed. Belegt sind die Aussagen durch Glickman (Text „The Glicko system“, am
2026-10-17 eingesehen: Spielausgänge senken die RD immer, Zeit ohne Partien erhöht sie immer; die Perioden-Partien gelten als gleichzeitig; die Änderungen der beiden
Spieler sind nicht ausgeglichen, sondern hängen an beiden RD) und durch die Formeln der Demo (README „Modell“).

Eigenschaften:
- Unsicherheit sinkt nach einer Partie (Glicko φ, TrueSkill σ) und wächst bis zum Deckel bei Spielern, die im Feld sind, aber nicht spielen;
- Spiegeln aller Ergebnisse spiegelt die Wertungen und lässt die Unsicherheiten unverändert (Elo, Glicko, TrueSkill, auch mit Neuling);
- Umnummerieren der Spieler permutiert nur die Ausgabe (Spieler 0, der Neuling bzw. Pausierende, bleibt);
- die Reihenfolge der Partien innerhalb einer Runde bzw. einer Periode ändert nichts (Glicko: gleichzeitige Aktualisierung);
- nur Differenzen der Wertungen zählen (gemeinsame Verschiebung);
- Gewinnerwartung von Elo symmetrisch; bei gleicher Unsicherheit sind Glicko und TrueSkill nullsummig, bei ungleicher bewegt sich der Unsicherere stärker;
- die Unsicherheit eines Pausierenden wächst (Glickman, Schritt 1): dieser Test deckt eine Abweichung der Demo auf, siehe dort."""

import dataclasses
import math

import numpy as np
import pytest

import glt_constants as C
import glt_drift as D
import glt_elo as E
import glt_glicko as G
import glt_trueskill as T
from conftest import build

N_INSTANCES = 24
RNG_SEED = 20261017
SCENARIOS = ("equal", "rare", "newcomer", "pause")


def _rng(salt: int) -> np.random.Generator:
    return np.random.default_rng(RNG_SEED + salt)


def _matches(rng, scenario: str = "equal") -> D.Matches:
    n = int(rng.integers(4, 10))
    rounds = int(rng.integers(30, 60))
    kw = dict(delta=float(rng.choice([0.1, 0.3, 0.6])), sigma=float(rng.choice([0.0, 0.05, 0.1])), p_rare=1.0, join_round=0, gap=0.0, pause_start=0, pause_len=0)
    if scenario == "rare":
        kw["p_rare"] = float(rng.choice([0.2, 0.5]))
    elif scenario == "newcomer":
        kw.update(join_round=int(rng.integers(5, 20)), gap=float(rng.choice([-1.0, 0.0, 2.0])), sigma=0.0)
    elif scenario == "pause":
        kw.update(pause_start=int(rng.integers(5, 15)), pause_len=int(rng.integers(5, 15)))
    return D.simulate_matches(n, kw["delta"], rounds, kw["sigma"], kw["p_rare"], kw["join_round"], kw["gap"], kw["pause_start"], kw["pause_len"], int(rng.integers(1, 10 ** 6)))


def _runs(m: D.Matches) -> dict:
    return {"Elo": E.run_elo(m, 28.0), "Elo mit Anlauf-K": E.run_elo(m, 20.0, 100.0), "Glicko": G.run_glicko(m, 0.05, 1.2), "TrueSkill": T.run_trueskill(m, 0.05, 1.2)}


def _same(a, b, atol=1e-9):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=atol, rtol=0, equal_nan=True)


# --- 1. Die Unsicherheit sinkt nach einer Partie ---------------------------------------------------------------------------------------------------

def test_a_glicko_period_with_games_always_lowers_the_uncertainty_and_more_games_lower_it_more():
    rng = _rng(1)
    for _ in range(N_INSTANCES):
        mu, phi = float(rng.normal(0, 1)), float(rng.uniform(0.1, 2.5))
        games = [(float(rng.normal(0, 1.5)), float(rng.uniform(0.05, 2.5)), float(rng.choice([0.0, 0.5, 1.0]))) for _ in range(int(rng.integers(1, 6)))]
        _, p1 = G.glicko_update(mu, phi, games[:1])
        _, pall = G.glicko_update(mu, phi, games)
        assert 0.0 < pall <= p1 < phi
    mu, phi = G.glicko_update(0.0, 1.0, [(0.0, 1.0, 1.0)])
    assert phi < 1.0
    assert G.glicko_update(0.0, 2.0, [(8.0, 1.0, 1.0)])[1] < 2.0                                   # Randfall: großer Abstand (1390 Punkte), die RD sinkt trotzdem


def test_a_trueskill_game_always_lowers_the_uncertainty_of_both_players():
    rng = _rng(2)
    for _ in range(N_INSTANCES):
        mu_w, mu_l = rng.normal(0, 1.5, 2)
        sd_w, sd_l = rng.uniform(0.05, 2.5, 2)
        new = T.trueskill_update(float(mu_w), float(sd_w), float(mu_l), float(sd_l))
        assert 0.0 < new[1] < sd_w and 0.0 < new[3] < sd_l
        assert new[0] > mu_w and new[2] < mu_l                                                      # Sieger steigt, Verlierer fällt
    new = T.trueskill_update(30.0, 1.0, 0.0, 1.0)                                                   # Randfall: erwarteter Sieg mit riesigem Abstand: fast keine Änderung
    assert new[1] <= 1.0 and new[3] <= 1.0 and new[0] == pytest.approx(30.0, abs=1e-6) and new[2] == pytest.approx(0.0, abs=1e-6)


def test_in_a_run_the_uncertainty_shrinks_with_the_games_and_grows_only_by_the_inflation_up_to_the_cap():
    rng = _rng(3)
    for _ in range(N_INSTANCES):
        m = _matches(rng, "rare")
        c, phi0 = 0.05, 1.2
        run = G.run_glicko(m, c, phi0)
        sd = run.sd
        assert (sd[~np.isnan(sd)] <= phi0 + 1e-12).all()                                            # nie über der Unsicherheit eines Unbewerteten
        played = np.zeros((m.rounds, m.n), dtype=bool)
        played[m.rnd, m.i] = True
        played[m.rnd, m.j] = True
        for t in range(1, m.rounds):
            for q in range(m.n):
                if played[t, q]:
                    assert sd[t, q] < min(math.sqrt(sd[t - 1, q] ** 2 + c * c), phi0) + 1e-12      # nach einer Partie kleiner als nach der Inflation allein
                else:                                                                                # im Feld, aber ohne Partie: nur das Wachstum
                    assert sd[t, q] == pytest.approx(min(math.sqrt(sd[t - 1, q] ** 2 + c * c), phi0), abs=1e-12)
                    assert sd[t, q] >= sd[t - 1, q] - 1e-12


def test_in_a_trueskill_run_every_game_shrinks_the_uncertainty_net_of_the_dynamics():
    rng = _rng(4)
    for _ in range(N_INSTANCES):
        m = _matches(rng, "rare")
        tau, sigma0 = 0.03, 1.2
        run = T.run_trueskill(m, tau, sigma0)
        played = np.zeros((m.rounds, m.n), dtype=bool)
        played[m.rnd, m.i] = True
        played[m.rnd, m.j] = True
        for t in range(1, m.rounds):
            for q in range(m.n):
                if played[t, q]:
                    assert run.sd[t, q] < math.sqrt(run.sd[t - 1, q] ** 2 + tau * tau) + 1e-12
                else:
                    assert run.sd[t, q] == pytest.approx(run.sd[t - 1, q], abs=1e-12)           # ohne Partie: unverändert (Dynamik nur je Partie, wie das Paket)


def test_a_player_in_the_field_who_does_not_play_loses_certainty_by_exactly_the_inflation():
    games = [[(0, 1, True)] for _ in range(20)]                                                    # Spieler 2 ist im Feld, spielt nie
    m = build(3, 20, games)
    run = G.run_glicko(m, 0.1, 1.0)
    for t in range(20):
        assert run.sd[t, 2] == pytest.approx(1.0)                                                  # beginnt am Deckel und bleibt dort
    run2 = G.run_glicko(m, 0.1, 2.0)
    assert (np.diff(run2.sd[:, 2]) >= -1e-12).all() and run2.sd[0, 2] == pytest.approx(min(math.sqrt(4.0 + 0.01), 2.0))


# --- 2. Spiegeln der Ergebnisse -----------------------------------------------------------------------------------------------------------------

def test_mirroring_every_result_negates_the_ratings_and_keeps_the_uncertainties_for_all_three_methods():
    rng = _rng(5)
    for idx in range(N_INSTANCES):
        m = _matches(rng, SCENARIOS[idx % 4])
        flipped = dataclasses.replace(m, win=1 - m.win)
        r, rf = _runs(m), _runs(flipped)
        for name in r:
            _same(rf[name].mu, -r[name].mu)
            _same(rf[name].final, -r[name].final)
            if name != "TrueSkill":                                                                 # TrueSkills Kennzahl misst nur den Sieger und ist deshalb nicht spiegelsymmetrisch
                assert rf[name].mean_k == pytest.approx(r[name].mean_k, rel=1e-9, abs=1e-9)
            if r[name].sd is not None:
                _same(rf[name].sd, r[name].sd)


def test_exchanging_the_two_players_of_every_game_and_flipping_the_result_changes_nothing():
    rng = _rng(6)
    for idx in range(N_INSTANCES):
        m = _matches(rng, SCENARIOS[idx % 4])
        sw = dataclasses.replace(m, i=m.j.copy(), j=m.i.copy(), win=1 - m.win)
        r, rs = _runs(m), _runs(sw)
        for name in r:
            _same(rs[name].mu, r[name].mu)
            if r[name].sd is not None:
                _same(rs[name].sd, r[name].sd)


# --- 3. Umnummerieren -------------------------------------------------------------------------------------------------------------------------------

def _renumbered(m: D.Matches, perm) -> D.Matches:
    inv = np.argsort(perm)
    return dataclasses.replace(m, truth=m.truth[:, inv], i=perm[m.i], j=perm[m.j], rare=tuple(sorted(int(perm[q]) for q in m.rare)))


def test_renumbering_the_players_only_permutes_the_output():
    rng = _rng(7)
    for idx in range(N_INSTANCES):
        scenario = SCENARIOS[idx % 4]
        m = _matches(rng, scenario)
        if scenario in ("newcomer", "pause"):
            perm = np.concatenate([[0], 1 + rng.permutation(m.n - 1)])                              # Spieler 0 ist der Neuling bzw. Pausierende
        else:
            perm = rng.permutation(m.n)
        mp = _renumbered(m, perm)
        r, rp = _runs(m), _runs(mp)
        for name in r:
            _same(rp[name].mu[:, perm], r[name].mu)
            _same(rp[name].final[perm], r[name].final)
            if r[name].sd is not None:
                _same(rp[name].sd[:, perm], r[name].sd)
            assert rp[name].mean_k == pytest.approx(r[name].mean_k, rel=1e-9)


# --- 4. Reihenfolge innerhalb einer Periode -----------------------------------------------------------------------------------------------------

def test_the_order_of_the_games_of_one_glicko_period_does_not_matter():
    rng = _rng(8)
    for _ in range(N_INSTANCES):
        mu, phi = float(rng.normal(0, 1)), float(rng.uniform(0.1, 2.5))
        games = [(float(rng.normal(0, 1.5)), float(rng.uniform(0.05, 2.5)), float(rng.choice([0.0, 0.5, 1.0]))) for _ in range(int(rng.integers(2, 7)))]
        base = G.glicko_update(mu, phi, games)
        for _ in range(3):
            perm = rng.permutation(len(games))
            assert G.glicko_update(mu, phi, [games[q] for q in perm]) == (pytest.approx(base[0], abs=1e-12), pytest.approx(base[1], abs=1e-12))
        # dieselbe Partie doppelt zählt als zwei Gegner mit denselben Werten (Glickman) und macht die Wertung sicherer
        assert G.glicko_update(mu, phi, games[:1] * 2)[1] < G.glicko_update(mu, phi, games[:1])[1]


def test_the_order_of_the_games_inside_a_round_does_not_matter_for_any_method():
    rng = _rng(9)
    for idx in range(N_INSTANCES):
        m = _matches(rng, SCENARIOS[idx % 4])
        perm = np.arange(len(m.i))
        for t in range(m.rounds):
            ids = np.flatnonzero(m.rnd == t)
            perm[ids] = rng.permutation(ids)
        shuffled = dataclasses.replace(m, i=m.i[perm], j=m.j[perm], win=m.win[perm])
        r, rs = _runs(m), _runs(shuffled)
        for name in r:
            _same(rs[name].mu, r[name].mu, atol=1e-12)
            if r[name].sd is not None:
                _same(rs[name].sd, r[name].sd, atol=1e-12)


# --- 5. Gemeinsame Verschiebung der Wertungen --------------------------------------------------------------------------------------------------

def test_only_rating_differences_count_for_all_three_update_rules():
    rng = _rng(10)
    for _ in range(N_INSTANCES):
        c = float(rng.uniform(-4, 4))
        mu, phi = float(rng.normal(0, 1)), float(rng.uniform(0.1, 2.5))
        games = [(float(rng.normal(0, 1.5)), float(rng.uniform(0.05, 2.5)), float(rng.choice([0.0, 0.5, 1.0]))) for _ in range(int(rng.integers(1, 5)))]
        a = G.glicko_update(mu, phi, games)
        b = G.glicko_update(mu + c, phi, [(x + c, y, s) for x, y, s in games])
        assert b[0] - c == pytest.approx(a[0], abs=1e-9) and b[1] == pytest.approx(a[1], abs=1e-12)
        mw, ml = rng.normal(0, 1.5, 2)
        sw, sl = rng.uniform(0.05, 2.5, 2)
        x = T.trueskill_update(float(mw), float(sw), float(ml), float(sl))
        y = T.trueskill_update(float(mw) + c, float(sw), float(ml) + c, float(sl))
        assert (y[0] - c, y[1], y[2] - c, y[3]) == pytest.approx(x, abs=1e-9)
        p, q = float(rng.normal(0, 2)), float(rng.normal(0, 2))
        assert E.expected_score(p + c, q + c) == pytest.approx(E.expected_score(p, q), abs=1e-12)


# --- 6. Symmetrie der Gewinnerwartung und Ausgleich der Änderungen -------------------------------------------------------------------------------

def test_the_elo_win_probability_is_symmetric():
    rng = _rng(11)
    for _ in range(N_INSTANCES):
        a, b = rng.normal(0, 3, 2)
        assert E.expected_score(float(a), float(b)) + E.expected_score(float(b), float(a)) == pytest.approx(1.0, abs=1e-12)
    for d in (800.0, 1e4):
        assert E.expected_score(d, 0.0) + E.expected_score(0.0, d) == pytest.approx(1.0, abs=1e-12)
    assert E.expected_score(0.7, 0.7) == 0.5


def test_with_equal_uncertainty_a_game_is_zero_sum_and_with_unequal_uncertainty_the_more_uncertain_player_moves_more():
    rng = _rng(12)
    for _ in range(N_INSTANCES):
        a, b = float(rng.normal(0, 1.5)), float(rng.normal(0, 1.5))
        s = float(rng.uniform(0.1, 2.5))
        # Glicko, i schlägt j
        wi, _ = G.glicko_update(a, s, [(b, s, 1.0)])
        wj, _ = G.glicko_update(b, s, [(a, s, 0.0)])
        assert (wi - a) + (wj - b) == pytest.approx(0.0, abs=1e-12) and wi > a and wj < b
        # TrueSkill
        x = T.trueskill_update(a, s, b, s)
        assert (x[0] - a) + (x[2] - b) == pytest.approx(0.0, abs=1e-12)
        # ungleich: der Unsicherere (Faktor 3) bewegt sich stärker, egal ob er gewinnt oder verliert
        s_big, s_small = 3.0 * s, s
        gi, _ = G.glicko_update(a, s_big, [(b, s_small, 1.0)])
        gj, _ = G.glicko_update(b, s_small, [(a, s_big, 0.0)])
        assert abs(gi - a) > abs(gj - b)
        y = T.trueskill_update(a, s_big, b, s_small)
        assert abs(y[0] - a) > abs(y[2] - b)
        z = T.trueskill_update(b, s_small, a, s_big)                                                # der Sichere gewinnt, der Unsichere verliert
        assert abs(z[2] - a) > abs(z[0] - b)


# --- 7. Die Unsicherheit eines Pausierenden wächst (Glickman, Schritt 1) -----------------------------------------------------------------------

def test_for_glicko_a_pause_is_the_same_as_staying_in_the_field_without_games():
    """Glickman: „time passing without competing in rated games always increases a player's RD“ (Schritt 1 gilt für jede Periode ohne Partien). Der Pausierende hat in
    der Pause keine Partien; sein Zustand bei der Rückkehr muss also derselbe sein wie bei einem Spieler, der im Feld bleibt und in denselben Runden nicht spielt (dieselben
    Partien der anderen). Die Demo lässt die Unsicherheit in der Pause stehen (die Inflation läuft nur für Spieler im Feld), die README sagt „Glicko wächst in der Unsicherheit
    mit“ und die Modellzeile „Je Runde: φ ← min(√(φ² + c²), φ₀)“.

    BEFUND (Stand 2026-10-17): fällt durch. `run_glicko` inflationiert nur Spieler, die laut `presence` im Feld sind (`pre = [inflate(...) if pres[t][q] else phi[q]]`); in der Pause
    bleibt φ stehen, und Glickmans Schritt 1 läuft erst wieder ab der Rückkehr. Der Orakel-Lauf in `test_oracle_glicko_trueskill.py` übernimmt dieselbe Regel (`if m.presence[t][q]`)."""
    rng = _rng(13)
    for _ in range(10):
        m = _matches(rng, "pause")
        idle = dataclasses.replace(m, pause=(0, 0))                                               # dieselben Partien, aber Spieler 0 bleibt im Feld
        c, phi0 = 0.08, 1.2
        paused, stayed = G.run_glicko(m, c, phi0), G.run_glicko(idle, c, phi0)
        end = m.pause[1]
        assert paused.sd[end, 0] == pytest.approx(stayed.sd[end, 0], rel=1e-9), (m.pause, paused.sd[end, 0], stayed.sd[end, 0])


def test_for_glicko_the_uncertainty_at_the_return_by_hand():
    """Handinstanz: Spieler 0 spielt 2 Runden gegen 1, pausiert 4 Runden (1 gegen 2), kehrt zurück. Ohne Partien wächst φ je Runde um c (Deckel φ₀)."""
    pre = [[(0, 1, True)], [(0, 1, False)]]
    mid = [[(1, 2, True)] for _ in range(4)]
    back = [[(0, 1, True)]]
    games = pre + mid + back
    stay = build(3, 7, games)
    paused = build(3, 7, games, pause=(2, 6))
    c, phi0 = 0.2, 1.5
    rs, rp = G.run_glicko(stay, c, phi0), G.run_glicko(paused, c, phi0)
    assert rs.sd[5, 0] == pytest.approx(min(math.sqrt(rs.sd[1, 0] ** 2 + 4 * c * c), phi0))          # vier Runden ohne Partien
    assert rp.sd[6, 0] == pytest.approx(rs.sd[6, 0], rel=1e-9)
