"""Literatur-Referenzwerte für Glicko und TrueSkill (ORACLE-PLAYBOOK §5.1). Nur Werte, die in einer Quelle eingesehen wurden (Stand der Einsicht: 2026-10-17).

Umrechnung: Die Demo rechnet in θ-Einheiten (1 Einheit = SCALE = 400/ln 10 ≈ 173,72 Elo-Punkte). Glicko: µ = (r − 1500)/SCALE, φ = RD/SCALE, c in θ = c in Punkten/SCALE; das
ist Glickmans q = ln 10/400 = 1/SCALE. TrueSkill: Die Demo rechnet mit β = 1,702/√2 statt β = 25/6; die Originaleinheiten x (µ = 25, σ = 25/3, β = 25/6) gehen über den Faktor
SCALE_ORIG = β_Demo/(25/6) in θ-Einheiten, θ = (x − 25)·SCALE_ORIG für µ und θ = σ_x·SCALE_ORIG für σ, zurück x = 25 + θ/SCALE_ORIG bzw. σ_x = σ_θ/SCALE_ORIG.

Werte
-----
1. Glicko-Rechenbeispiel: Spieler 1500 mit RD 200 gegen 1400 (RD 30, Sieg), 1550 (RD 100, Niederlage), 1700 (RD 300, Niederlage).
   Quelle: Glickman, M. E., „The Glicko system“ (Harvard University), http://www.glicko.net/glicko/glicko.pdf, Abschnitt „Example calculation“ (Seite 4), per WebFetch
   abgerufen und im Text der PDF-Seite gelesen am 2026-10-17. Dort: g(RD_j) = 0,9955 / 0,9531 / 0,7242; E = 0,639 / 0,432 / 0,303; d² = 231,67² (= 53670,85); r' = 1500 + 131,9·(−0,272)
   = 1464; RD' = √22918,9 = 151,4. Die Quelle rechnet mit gerundeten Zwischenwerten (d = 231,67 statt 231,70 der ungerundeten Rechnung); verglichen wird mit den Toleranzen der Rundung.
2. Glicko Schritt 1: Zuwachs der Unsicherheit je Periode und Deckel. Quelle: dieselbe PDF, „Step 1 (b)“ und „Implementation issues“ (Seiten 3, 5 und 6): RD = min(√(RD_old² + c²), 350)
   bzw. nach t Perioden ohne Partien RD = min(√(RD_old² + c² t), 350); Beispiel mit RD 50, 30 Perioden, Ziel 350: c = 63,2. Der Wert 63,2 und die Formel stehen in der Quelle.
3. Erwartete Werte von TrueSkill für zwei neue Spieler: Quelle: Dokumentation des Pakets `trueskill`, https://trueskill.org/ (eingesehen am 2026-10-17): Voreinstellung µ = 25, σ = 8,333,
   β = 4,167, τ = 0,0833, Remis-Wahrscheinlichkeit 0,10; `rate_1vs1` zweier neuer Spieler ohne Remis ergibt Sieger Rating(µ=29,396, σ=7,171) und Verlierer Rating(µ=20,604, σ=7,171).
   Das gilt mit der Voreinstellung der Remis-Wahrscheinlichkeit 0,10 (README der Demo: ohne Remis 29,205/7,194). Die Dokumentation nennt auch das Remis-Beispiel (25,000/6,458); die Demo
   kennt keine Remis-Partien und prüft es nicht.

Nicht eingesehen oder nicht anwendbar (siehe Bericht „offen“): Herbrich, Minka, Graepel (2006) im Original (der Text ließ sich nicht sauber lesen; keine nachgerechneten Zahlenwerte
übernommen); Glickman (1999) in Applied Statistics (nicht eingesehen); Glickmans Formel für die erwartete Spielausgang-Wahrscheinlichkeit mit √(RD_i² + RD_j²) (Beispiel 0,376) und die
Match-Qualität des Pakets (44,7 %) hat die Demo nicht umgesetzt; Glicko-2 ist nicht Teil der Demo."""

import math

import pytest

import glt_constants as C
import glt_glicko as G
import glt_trueskill as T
from conftest import build

S = C.SCALE
SCALE_ORIG = C.BETA / (25.0 / 6.0)


def test_glickman_worked_example_weights_expectations_rating_and_deviation():
    games = [(1400.0, 30.0, 1.0), (1550.0, 100.0, 0.0), (1700.0, 300.0, 0.0)]
    for (r_j, rd_j, _), g_paper, e_paper in zip(games, (0.9955, 0.9531, 0.7242), (0.639, 0.432, 0.303)):
        g = G.g_factor(rd_j / S)
        assert g == pytest.approx(g_paper, abs=5e-5)
        assert 1.0 / (1.0 + math.exp(-g * (0.0 - (r_j - 1500.0) / S))) == pytest.approx(e_paper, abs=5e-4)
    mu, phi = G.glicko_update(0.0, 200.0 / S, [((r_j - 1500.0) / S, rd_j / S, s) for r_j, rd_j, s in games])
    assert 1500.0 + S * mu == pytest.approx(1464.0, abs=0.5)                              # Quelle: 1464 (ganzzahlig gerundet)
    assert S * phi == pytest.approx(151.4, abs=0.05)
    # d² der Quelle: 1/RD'² = 1/RD² + 1/d²
    d = 1.0 / math.sqrt(1.0 / (S * phi) ** 2 - 1.0 / 200.0 ** 2)
    assert d == pytest.approx(231.67, abs=0.1)


def test_glickman_step_1_uncertainty_growth_and_the_cap_at_350():
    c = 63.2 / S
    cap = 350.0 / S
    rd = 50.0 / S
    for t in range(1, 31):
        rd = G.inflate(rd, c, cap)                                                        # eine Periode ohne Partien
        assert S * rd == pytest.approx(min(math.sqrt(50.0 ** 2 + 63.2 ** 2 * t), 350.0), abs=1e-9)
    assert S * rd == pytest.approx(350.0, abs=0.5)                                       # nach 30 Perioden fast wieder so unsicher wie ein Unbewerteter
    assert math.sqrt((350.0 ** 2 - 50.0 ** 2) / 30.0) == pytest.approx(63.2, abs=0.05)   # die Herleitung der Quelle
    for _ in range(200):
        rd = G.inflate(rd, c, cap)
    assert S * rd == pytest.approx(350.0, abs=1e-9) and G.inflate(cap, c, cap) == cap     # nie über 350


def _two_new_players(tau_orig: float):
    """Zwei neue Spieler (µ 25, σ 25/3, β 25/6), Remis-Wahrscheinlichkeit 0,10, Spieler 0 gewinnt; Ergebnis in Originaleinheiten."""
    m = build(2, 1, [[(0, 1, True)]])
    run = T.run_trueskill(m, tau_orig * SCALE_ORIG, (25.0 / 3.0) * SCALE_ORIG, draw_probability=0.10, beta=C.BETA)
    return 25.0 + run.mu[0, 0] / SCALE_ORIG, run.sd[0, 0] / SCALE_ORIG, 25.0 + run.mu[0, 1] / SCALE_ORIG, run.sd[0, 1] / SCALE_ORIG


@pytest.mark.parametrize("tau", [0.0, 25.0 / 300.0])
def test_trueskill_documentation_rate_1vs1_of_two_new_players(tau):
    # die Dokumentation rechnet mit τ = 0,0833; das ändert die Werte erst in der fünften Stelle, beide Fassungen runden auf dieselben Zahlen
    mu_w, sd_w, mu_l, sd_l = _two_new_players(tau)
    assert (round(mu_w, 3), round(sd_w, 3)) == (29.396, 7.171)
    assert (round(mu_l, 3), round(sd_l, 3)) == (20.604, 7.171)
