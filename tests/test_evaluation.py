"""Maße (von Hand gerechnete Läufe), Analyse des gezeigten Laufs, Verdikt-Zweige und die Struktur der Experimente."""

import math

import numpy as np
import pytest

import glt_constants as C
import glt_drift as D
import glt_evaluation as ev
from conftest import build
from glt_scenario import Settings


def run_of(mu, sd=None):
    mu = np.asarray(mu, dtype=float)
    return D.Run(mu, None if sd is None else np.asarray(sd, dtype=float), mu[-1], 0.0)


# --- Maße gegen die Handrechnung --------------------------------------------------------------------------------------------------------------

def test_tracking_error_by_hand():
    truth = [[0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]
    m = build(3, 2, [[], []], truth=truth)
    run = run_of([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    # Runde 0: Abweichungen (0,5; 0; −0,5) -> RMSE √(0,5/3); Runde 1: (1; 0; −1) -> √(2/3)
    assert ev.tracking_error(run, m, burn=0) == pytest.approx((math.sqrt(0.5 / 3) + math.sqrt(2.0 / 3)) / 2)
    assert ev.tracking_error(run, m, burn=1) == pytest.approx(math.sqrt(2.0 / 3))
    assert ev.round_error(run, m, 0) == pytest.approx(math.sqrt(0.5 / 3))


def test_tracking_error_is_invariant_to_a_common_shift_of_the_ratings():
    m = build(3, 2, [[], []], truth=[[0.5, 0.0, -0.5], [1.0, 0.0, -1.0]])
    a = run_of([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    b = run_of([[11.0, 10.0, 9.0], [12.0, 10.0, 8.0]])
    assert ev.tracking_error(a, m, burn=0) == pytest.approx(ev.tracking_error(b, m, burn=0))


def test_tracking_error_over_a_subset_centres_inside_the_subset():
    m = build(4, 1, [[]], truth=[[0.0, 1.0, 2.0, 3.0]])
    run = run_of([[0.0, 5.0, 2.0, 3.0]])
    # Teilmenge {0, 2}: Wertungen (0, 2), Stärken (0, 2): zentriert gleich -> Fehler 0
    assert ev.tracking_error(run, m, burn=0, subset=(0, 2)) == pytest.approx(0.0)
    assert ev.tracking_error(run, m, burn=0) > 0.5


def test_only_players_in_the_field_count():
    m = build(3, 2, [[], []], join_round=1, truth=[[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]])
    mu = [[np.nan, 1.0, -1.0], [0.0, 1.0, -1.0]]
    assert ev.round_error(run_of(mu), m, 0) == pytest.approx(0.0)           # Spieler 0 fehlt: nur 1 und 2 zählen, beide exakt
    assert math.isnan(ev.round_error(run_of([[np.nan, np.nan, 0.0]] * 2), m, 0))     # ein einzelner Spieler: nichts zu zentrieren


def test_coverage_by_hand():
    truth = [[0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]
    m = build(3, 2, [[], []], truth=truth)
    mu = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]]
    sd = [[0.3] * 3, [0.3] * 3]
    # 1,96 · 0,3 = 0,588: Runde 0 Abweichungen (0,5; 0; 0,5) -> 3 von 3; Runde 1 (1; 0; 1) -> 1 von 3
    assert ev.coverage(run_of(mu, sd), m, burn=0) == pytest.approx(4.0 / 6.0)
    assert ev.coverage(run_of(mu, sd), m, burn=1) == pytest.approx(1.0 / 3.0)
    assert math.isnan(ev.coverage(run_of(mu), m, burn=0))                  # Elo hat keine Unsicherheit


def test_player_rounds_and_error_after_games():
    games = [[(0, 1, True)], [(2, 3, True)], [(0, 2, False)], [(0, 3, True)]]
    m = build(4, 4, games, truth=np.zeros((4, 4)))
    assert ev.player_rounds(m, 0) == [0, 2, 3]
    assert ev.player_rounds(m, 0, after=1) == [2, 3]
    mu = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
    run = run_of(mu)
    assert ev.error_after_games(run, m, 1) == pytest.approx(0.0)
    assert ev.error_after_games(run, m, 2) == pytest.approx(1.0 - 0.25)    # nach der zweiten Partie (Runde 2): µ0 = 1, Feldmittel 0,25
    assert math.isnan(ev.error_after_games(run, m, 4))


def test_kendall_tau_values():
    assert ev.kendall_tau([3, 2, 1], [30, 20, 10]) == 1.0
    assert ev.kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0
    assert ev.kendall_tau([1, 1, 2], [1, 2, 3]) == pytest.approx(1.0)      # Gleichstände zählen nicht


def test_run_method_dispatch_and_unknown_method():
    m = D.simulate_matches(4, 0.3, 10, 0.0, 1.0, 0, 0.0, 0, 0, 1)
    assert ev.run_method(m, "Elo", 20.0).sd is None
    assert ev.run_method(m, "Glicko", 0.05, 1.0).sd is not None
    assert ev.run_method(m, "TrueSkill", 0.05, 1.0).sd is not None
    with pytest.raises(ValueError):
        ev.run_method(m, "Bradley-Terry", 1.0)


# --- Analyse des gezeigten Laufs --------------------------------------------------------------------------------------------------------------

def test_analyse_is_deterministic_and_fills_the_fields_for_each_scenario():
    base = Settings(n=6, rounds=60, seed=3)
    a = ev.analyse(base)
    assert repr(a.error) == repr(ev.analyse(base).error) and set(a.error) == set(C.METHODS)
    assert a.error_rare == {} and a.player0 == {} and math.isnan(a.cover["Elo"]) and 0.0 < a.cover["Glicko"] <= 1.0
    b = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[1], p_rare=0.3, seed=3))
    assert set(b.error_rare) == set(C.METHODS) and set(b.error_regular) == set(C.METHODS)
    c = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[2], join=20, gap=1.0, seed=3))
    assert set(c.player0) == set(C.METHODS) and not any(math.isnan(v) for v in c.player0["Elo"])
    d = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[3], pause_start=10, pause_len=20, seed=3))
    assert set(d.player0) == set(C.METHODS) and len(d.player0["Glicko"]) == len(C.PAUSE_GAMES)


def test_the_scenario_parameters_are_clamped_to_valid_values():
    s = Settings(rounds=20, scenario=C.SCENARIOS[2], join=100)
    assert s.join_round == 18
    s = Settings(rounds=20, scenario=C.SCENARIOS[2], join=0)
    assert s.join_round == C.JOIN_MIN
    assert Settings(scenario=C.SCENARIOS[0], join=60).join_round == 0
    s = Settings(rounds=20, scenario=C.SCENARIOS[3], pause_start=100, pause_len=100)
    start, length = s.pause
    assert C.PAUSE_START_MIN <= start <= C.pause_start_max(20) and length >= C.PAUSE_LEN_MIN and start + length <= 20 - 5
    assert Settings(scenario=C.SCENARIOS[1]).pause == (0, 0) and Settings(scenario=C.SCENARIOS[0]).p_rare_eff == 1.0
    assert Settings(scenario=C.SCENARIOS[1], p_rare=0.3).p_rare_eff == 0.3


def test_settings_convert_points_to_theta_units():
    s = Settings(phi0=350.0, c=8.0)
    assert s.phi0_theta == pytest.approx(350.0 / C.SCALE) and s.c_theta == pytest.approx(8.0 / C.SCALE)


# --- Verdikte -------------------------------------------------------------------------------------------------------------------------------

def test_verdict_analysis_branches():
    a = ev.analyse(Settings(n=6, rounds=60, seed=3))
    level, text = ev.verdict_analysis(a)
    assert level == "info" and "Verfolgungsfehler: Elo" in text
    b = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[1], seed=3))
    assert "selten spielenden" in ev.verdict_analysis(b)[1]
    c = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[2], join=20, seed=3))
    assert "Fehler des Neulings nach 5 Partien" in ev.verdict_analysis(c)[1]
    late = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[2], join=58, seed=3))      # zu spät für fünf Partien: nur der Verfolgungsfehler
    assert "Neulings" not in ev.verdict_analysis(late)[1]
    d = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[3], pause_start=10, pause_len=20, seed=3))
    assert "Fehler des Rückkehrers nach 5 Partien" in ev.verdict_analysis(d)[1]


def test_verdict_equal_names_a_close_race_as_no_gain_and_a_gap_as_best():
    near = {"Elo": 0.300, "Glicko": 0.305, "TrueSkill": 0.310}
    a = ev.analyse(Settings(n=6, rounds=60, seed=3))
    import dataclasses
    a2 = dataclasses.replace(a, error=near)
    assert "gleichauf" in ev.verdict_analysis(a2)[1]
    far = dataclasses.replace(a, error={"Elo": 0.30, "Glicko": 0.20, "TrueSkill": 0.35})
    assert "Am besten liegt **Glicko**" in ev.verdict_analysis(far)[1]


def test_number_formatting_uses_the_decimal_comma():
    assert ev.de(3.14159, 3) == "3,142" and ev.pct(0.594) == "59" and ev.pct(0.5946, 1) == "59,5"


# --- Experimente: Struktur und Auswahl des besten Parameters -----------------------------------------------------------------------------------

def test_best_picks_the_parameter_with_the_smallest_mean_error():
    seeds = C.SWEEP_SEEDS[:6]
    param, sc = ev._best("equal", 0.06, "Elo", C.PHI0_DEFAULT, seeds=seeds)
    means = {p: ev._mean([ev._score("equal", 0.06, sd, "Elo", p, C.PHI0_DEFAULT).total for sd in seeds]) for p in C.K_GRID}
    assert param == min(means, key=means.get) and sc.total == pytest.approx(means[param])


def test_scores_are_cached_and_deterministic():
    a = ev._score("equal", 0.06, C.SWEEP_SEEDS[0], "Glicko", 0.06, C.PHI0_DEFAULT)
    b = ev._score("equal", 0.06, C.SWEEP_SEEDS[0], "Glicko", 0.06, C.PHI0_DEFAULT)
    assert a is b and 0.0 < a.cover <= 1.0 and math.isnan(a.regular)


def test_unknown_instance_kind_is_rejected():
    with pytest.raises(ValueError):
        ev._matches("hockey", 0.0, 1)


def test_the_pause_bounds_always_leave_a_valid_slider_range():
    for rounds in range(C.ROUNDS_MIN, C.ROUNDS_MAX + 1):
        hi = C.pause_start_max(rounds)
        assert hi > C.PAUSE_START_MIN
        for start in (C.PAUSE_START_MIN, hi):
            assert C.pause_len_max(rounds, start) > C.PAUSE_LEN_MIN


# --- Grenzfälle der Maße (aus dem Mutationstest) -------------------------------------------------------------------------------------------

def test_a_round_with_a_single_player_in_the_field_counts_for_nothing():
    m = build(2, 2, [[], [(0, 1, True)]], join_round=1, truth=[[0.0, 0.0], [0.0, 0.0]])
    run = run_of([[np.nan, 5.0], [3.0, 0.0]])
    assert math.isnan(ev.round_error(run, m, 0))                               # nur Spieler 1 im Feld: nichts zu zentrieren
    assert ev.tracking_error(run, m, burn=0) == pytest.approx(1.5)             # nur Runde 1 zählt: Abweichungen (1,5; −1,5)
    assert math.isnan(ev.tracking_error(run_of([[np.nan, 5.0]]), build(2, 1, [[]], join_round=1), burn=0))
    sd = [[np.nan, 0.1], [0.1, 0.1]]
    assert ev.coverage(run_of(run.mu, sd), m, burn=0) == pytest.approx(0.0)    # nur die Zweierrunde: beide außerhalb, die Einzelrunde würde einen Treffer zählen


def test_coverage_counts_a_value_exactly_on_the_interval_edge():
    m = build(3, 1, [[]], truth=[[0.0, 0.0, 0.0]])
    run = run_of([[1.96, 0.0, -1.96]], [[1.0, 1.0, 1.0]])
    assert ev.coverage(run, m, burn=0) == pytest.approx(1.0)


def test_error_after_games_is_defined_for_exactly_that_many_games():
    games = [[(0, 1, True)], [(0, 1, True)], [(0, 1, False)]]
    m = build(2, 3, games, truth=np.zeros((3, 2)))
    run = run_of([[0.0, 0.0], [0.0, 0.0], [1.0, -1.0]])
    assert ev.error_after_games(run, m, 3) == pytest.approx(1.0)
    assert math.isnan(ev.error_after_games(run, m, 4))


def test_final_errors_and_ranks_of_the_analysis_are_recomputed_independently():
    a = ev.analyse(Settings(n=7, rounds=60, scenario=C.SCENARIOS[1], p_rare=0.3, seed=11))
    m = a.matches
    truth = m.truth[-1] - m.truth[-1].mean()
    for name, run in a.runs.items():
        fin = run.final - run.final.mean()
        assert a.final_err[name] == pytest.approx(float(np.sqrt(np.mean((fin - truth) ** 2))))
        assert a.tau[name] == pytest.approx(ev.kendall_tau(fin, truth))
    regular = tuple(q for q in range(7) if q % 2 == 0)                            # n = 7: die letzte Nummer (6) gehört zu den oft spielenden
    assert regular[-1] == 6 and m.rare == (1, 3, 5)
    for name, run in a.runs.items():
        assert a.error_regular[name] == pytest.approx(ev.tracking_error(run, m, subset=regular))
        assert a.error_rare[name] == pytest.approx(ev.tracking_error(run, m, subset=(1, 3, 5)))


def test_the_verdict_names_the_best_method_for_the_rare_players_and_the_newcomer():
    a = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[1], seed=3))
    best = min(a.error_rare, key=a.error_rare.get)
    assert f"am besten **{best}**" in ev.verdict_analysis(a)[1]
    c = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[2], join=20, seed=3))
    best = min(C.METHODS, key=lambda k: c.player0[k][1])
    assert f"am besten **{best}**" in ev.verdict_analysis(c)[1]


def test_the_pause_is_clamped_from_below_as_well():
    s = Settings(rounds=100, scenario=C.SCENARIOS[3], pause_start=1, pause_len=1)
    assert s.pause == (C.PAUSE_START_MIN, C.PAUSE_LEN_MIN)
