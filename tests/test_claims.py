"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import glt_constants as C
import glt_evaluation as ev


@pytest.fixture(scope="module")
def equal():
    return {(r["sigma"], r["method"]): r for r in ev.sweep_equal()}


@pytest.fixture(scope="module")
def calibration():
    return {(r["c"], r["method"]): r for r in ev.sweep_calibration()}


@pytest.fixture(scope="module")
def activity():
    return {r["label"]: r for r in ev.sweep_activity()}


@pytest.fixture(scope="module")
def newcomer():
    return {(r["gap"], r["label"]): r["error"] for r in ev.sweep_newcomer()}


@pytest.fixture(scope="module")
def pause():
    return {(r["sigma"], r["label"]): r for r in ev.sweep_pause()}


@pytest.fixture(scope="module")
def compare():
    return ev.compare_glicko_trueskill()


# --- Experiment 1: alle spielen gleich oft -----------------------------------------------------------------------------------------------------

def test_with_equal_activity_the_three_methods_are_level(equal):
    expected = {0.0: (0.239, 0.229, 0.245), 0.03: (0.269, 0.270, 0.282), 0.06: (0.334, 0.339, 0.348), 0.12: (0.486, 0.483, 0.486)}
    for sg, (e, g, t) in expected.items():
        assert equal[(sg, "Elo")]["error"] == pytest.approx(e, abs=0.001), sg
        assert equal[(sg, "Glicko")]["error"] == pytest.approx(g, abs=0.001), sg
        assert equal[(sg, "TrueSkill")]["error"] == pytest.approx(t, abs=0.001), sg
    assert max(abs(equal[(sg, "Elo")]["error"] - equal[(sg, "Glicko")]["error"]) for sg in expected) < 0.012
    assert max(abs(equal[(sg, "Elo")]["error"] - equal[(sg, "TrueSkill")]["error"]) for sg in expected) < 0.015
    assert max(abs(equal[(sg, "Glicko")]["error"] - equal[(sg, "TrueSkill")]["error"]) for sg in expected) < 0.018      # README: Glicko–TrueSkill bis 0,017


def test_the_best_parameters_follow_the_drift(equal):
    assert [equal[(sg, "Elo")]["param"] for sg in C.SIGMA_SWEEP] == [20.0, 20.0, 28.0, 56.0]
    assert [equal[(sg, "Glicko")]["param"] for sg in C.SIGMA_SWEEP] == [0.01, 0.02, 0.06, 0.12]
    assert [equal[(sg, "TrueSkill")]["param"] for sg in C.SIGMA_SWEEP] == [0.01, 0.02, 0.06, 0.12]


def test_the_effective_gain_of_the_methods_with_uncertainty_is_larger_than_elos_fixed_k(equal):
    assert equal[(0.06, "Elo")]["mean_k"] == 28.0
    assert equal[(0.06, "Glicko")]["mean_k"] == pytest.approx(41.8, abs=0.1)
    assert equal[(0.06, "TrueSkill")]["mean_k"] == pytest.approx(51.0, abs=0.1)


# --- Experiment 2: Kalibrierung --------------------------------------------------------------------------------------------------------------

def test_the_coverage_depends_on_c(calibration):
    cov = {c: calibration[(c, "Glicko")]["cover"] for c in C.CAL_C}
    assert [round(cov[c], 3) for c in C.CAL_C] == [0.815, 0.829, 0.909, 0.974, 0.993, 0.995]
    assert calibration[(0.06, "TrueSkill")]["cover"] == pytest.approx(0.974, abs=0.001)
    err = {c: calibration[(c, "Glicko")]["error"] for c in C.CAL_C}
    assert min(err, key=err.get) == 0.06 and err[0.24] > 1.5 * err[0.06]


# --- Experiment 3: ungleiche Aktivität ---------------------------------------------------------------------------------------------------

def test_the_default_prior_hurts_and_an_informed_prior_helps(activity):
    assert activity["Elo"]["total"] == pytest.approx(0.474, abs=0.001) and activity["Elo"]["param"] == 40.0
    assert activity["Glicko, 350 Punkte"]["total"] == pytest.approx(0.500, abs=0.001)
    assert activity["Glicko, informiert"]["total"] == pytest.approx(0.445, abs=0.001)
    assert activity["TrueSkill, 350 Punkte"]["total"] == pytest.approx(0.533, abs=0.001)
    assert activity["TrueSkill, informiert"]["total"] == pytest.approx(0.454, abs=0.001)
    assert activity["Glicko, 350 Punkte"]["total"] > activity["Elo"]["total"] > activity["Glicko, informiert"]["total"]


def test_the_rare_players_are_the_hard_ones(activity):
    for label, row in activity.items():
        assert row["rare"] > row["regular"], label
    assert activity["Elo"]["rare"] == pytest.approx(0.518, abs=0.001) and activity["Glicko, informiert"]["rare"] == pytest.approx(0.476, abs=0.001)
    assert activity["Elo"]["regular"] == pytest.approx(0.311, abs=0.001) and activity["Glicko, informiert"]["regular"] == pytest.approx(0.293, abs=0.001)


def test_the_activity_coverage_is_close_to_nominal(activity):
    assert activity["Glicko, 350 Punkte"]["cover"] == pytest.approx(0.951, abs=0.001)
    assert activity["Glicko, informiert"]["cover"] == pytest.approx(0.964, abs=0.001)
    assert activity["TrueSkill, informiert"]["cover"] == pytest.approx(0.938, abs=0.001)


# --- Experiment 4: Neuling -------------------------------------------------------------------------------------------------------------------

def test_the_newcomer_errors(newcomer):
    exp = {(0.0, "Elo, K = 20 fest"): 0.159, (0.0, "Elo, K = 20 mit Anlauf-K 60"): 0.251, (0.0, "Glicko, 350 Punkte"): 0.348, (0.0, "Glicko, informiert"): 0.250,
           (0.0, "TrueSkill, 350 Punkte"): 0.383, (1.0, "Elo, K = 20 fest"): 0.531, (1.0, "Elo, K = 20 mit Anlauf-K 60"): 0.352, (1.0, "Glicko, 350 Punkte"): 0.338,
           (1.0, "Glicko, informiert"): 0.353, (1.0, "TrueSkill, 350 Punkte"): 0.375, (2.0, "Elo, K = 20 fest"): 1.110, (2.0, "Elo, K = 20 mit Anlauf-K 60"): 0.716,
           (2.0, "Glicko, 350 Punkte"): 0.376, (2.0, "Glicko, informiert"): 0.712, (2.0, "TrueSkill, 350 Punkte"): 0.450}
    for key, value in exp.items():
        assert newcomer[key] == pytest.approx(value, abs=0.002), key


def test_the_newcomer_far_from_the_field_is_found_fast_and_the_equal_one_is_made_noisy(newcomer):
    assert newcomer[(2.0, "Glicko, 350 Punkte")] < 0.55 * newcomer[(2.0, "Elo, K = 20 mit Anlauf-K 60")]
    assert newcomer[(2.0, "Glicko, 350 Punkte")] < 0.4 * newcomer[(2.0, "Elo, K = 20 fest")]
    assert newcomer[(0.0, "Glicko, 350 Punkte")] > 2.0 * newcomer[(0.0, "Elo, K = 20 fest")]


# --- Experiment 5: Pause ---------------------------------------------------------------------------------------------------------------------

def test_the_pause_gain_is_small_and_only_with_strong_drift(pause):
    assert pause[(0.12, "Elo")]["after20"] == pytest.approx(0.460, abs=0.001) and pause[(0.12, "Elo")]["param"] == 112.0
    assert pause[(0.12, "Glicko, informiert")]["after20"] == pytest.approx(0.430, abs=0.001)
    assert pause[(0.12, "Glicko, 350 Punkte")]["after20"] == pytest.approx(0.430, abs=0.001)
    assert pause[(0.12, "TrueSkill, informiert")]["after20"] == pytest.approx(0.441, abs=0.001)
    assert pause[(0.06, "Elo")]["after20"] == pytest.approx(0.288, abs=0.001)
    assert pause[(0.06, "Glicko, informiert")]["after20"] == pytest.approx(0.282, abs=0.001)
    assert pause[(0.06, "Glicko, 350 Punkte")]["after20"] == pytest.approx(0.288, abs=0.001)
    # nach der Rückkehr ist die Unsicherheit gewachsen: gleich nach der ersten Partie liegt Glicko deutlich vorn (0,431 gegen 0,454), nach 20 Partien nur noch um 0,01
    assert pause[(0.06, "Elo")]["after1"] == pytest.approx(0.454, abs=0.001) and pause[(0.06, "Glicko, informiert")]["after1"] == pytest.approx(0.431, abs=0.001)
    assert pause[(0.06, "Elo")]["after20"] - pause[(0.06, "Glicko, informiert")]["after20"] == pytest.approx(0.006, abs=0.002)


# --- Experiment 6: Glicko gegen TrueSkill -------------------------------------------------------------------------------------------------

def test_glicko_and_trueskill_are_nearly_the_same_method(compare):
    assert compare["runs"] == 60
    assert compare["diff_mean"] == pytest.approx(-0.0086, abs=0.0005) and compare["diff_sd"] == pytest.approx(0.0111, abs=0.0005)
    assert compare["tau"] == pytest.approx(0.987, abs=0.001)


# --- Texte -------------------------------------------------------------------------------------------------------------------------------------

def test_verdict_texts_quote_the_measured_numbers(equal, calibration, activity, newcomer, pause, compare):
    level, text = ev.verdict_equal(list(equal.values()))
    assert "Elo 0,334, Glicko 0,339, TrueSkill 0,348" in text and "0,010" in text
    text = ev.verdict_calibration([{"c": c, "method": m, "cover": r["cover"], "error": r["error"]} for (c, m), r in calibration.items()])[1]
    assert "**81 %** bei c = 0,00" in text and "**97 %** bei c = 0,06" in text and "**99,5 %** bei c = 0,24" in text
    text = ev.verdict_activity(list(activity.values()))[1]
    assert "Elo mit K = 40 bei 0,474" in text and "bei 0,500 (schlechter)" in text and "bei 0,445 (besser)" in text
    text = ev.verdict_newcomer([{"gap": g, "label": lab, "error": e} for (g, lab), e in newcomer.items()])[1]
    assert "Abstand 2 vom Feld" in text and "festem K 1,11" in text and "Anlauf-K 0,72" in text and "Glicko 0,38**" in text and "Elo 0,16, Glicko 0,35" in text
    text = ev.verdict_pause([{"sigma": s, "label": lab, **{k: r[k] for k in ("after1", "after5", "after20")}} for (s, lab), r in pause.items()])[1]
    assert "Glicko 0,43**" in text and "Elo 0,46" in text and "(Vorsprung 0,03)" in text and "0,28 gegen 0,29 (Vorsprung 0,01)" in text
    text = ev.verdict_compare(compare)[1]
    assert "-0,009 ± 0,011" in text and "τ = 0,99" in text
